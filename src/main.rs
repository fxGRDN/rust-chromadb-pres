use anyhow::Result;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{ChromaCollection, CollectionEntries, QueryResult};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::env;
use std::fs;
use std::sync::{Arc, Mutex};
use tokio::task;

#[derive(Debug, Serialize, Deserialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

pub struct GeminiClient {
    api_key: String,
    client: reqwest::Client,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    pub async fn generate_content(&self, prompt: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}",
            self.api_key
        );

        let request = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart {
                    text: prompt.to_string(),
                }],
            }],
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let gemini_response: GeminiResponse = response.json().await?;
            if let Some(candidate) = gemini_response.candidates.first() {
                if let Some(part) = candidate.content.parts.first() {
                    return Ok(part.text.clone());
                }
            }
            anyhow::bail!("No content in Gemini response")
        } else {
            let error_text = response.text().await?;
            anyhow::bail!("Gemini API error: {}", error_text)
        }
    }

    pub async fn generate_embedding_prompt(&self, documents: &[String]) -> Result<String> {
        let docs_text = documents.join("\n---\n");
        let prompt = format!(
            "Analyze the following documents and create a concise summary that captures the key themes and concepts:\n\n{}",
            docs_text
        );
        
        self.generate_content(&prompt).await
    }
}

pub struct EmbeddingEngine {
    model: Arc<Mutex<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel>>,
}

impl EmbeddingEngine {
    pub async fn new() -> Result<Self> {
        println!("Loading BERT model for embeddings...");
        
        let model = task::spawn_blocking(|| {
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
                .create_model()
        })
        .await??;
        
        println!("BERT model loaded successfully!");
        Ok(Self { 
            model: Arc::new(Mutex::new(model)) 
        })
    }

    pub async fn generate_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let model = Arc::clone(&self.model);
        
        let embeddings = task::spawn_blocking(move || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let model_guard = model.lock().unwrap();
            model_guard.encode(&text_refs)
        })
        .await??;
        
        Ok(embeddings)
    }
    pub async fn generate_single_embedding(&self, text: String) -> Result<Vec<f32>> {
        let model = Arc::clone(&self.model);
        
        let embeddings = task::spawn_blocking(move || {
            let model_guard = model.lock().unwrap();
            model_guard.encode(&[text.as_str()])
        })
        .await??;
        
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }
    }

pub struct ChromaGeminiApp {
    chroma: ChromaClient,
    gemini: GeminiClient,
    embedding_engine: EmbeddingEngine,
}

impl ChromaGeminiApp {
    pub async fn new(chroma_url: String, gemini_api_key: String) -> Result<Self> {
        let mut options = ChromaClientOptions::default();
        options.url = Some(chroma_url);
        let chroma = ChromaClient::new(options).await?;
        
        let embedding_engine = EmbeddingEngine::new().await?;
        
        Ok(Self {
            chroma,
            gemini: GeminiClient::new(gemini_api_key),
            embedding_engine,
        })
    }

    pub async fn setup_collection(&self, collection_name: &str) -> Result<ChromaCollection> {
        let collection = self.chroma
            .get_or_create_collection(collection_name, None)
            .await?;
        
        println!("Collection '{}' ready", collection_name);
        Ok(collection)
    }

    pub async fn check_existing_documents(&self, collection: &ChromaCollection, file_paths: &[String]) -> Result<Vec<String>> {
        let mut missing_files = vec![];
        
        for file_path in file_paths {
            let file_id = self.path_to_id(file_path);
            
            use chromadb::collection::GetOptions;
            
            let get_options = GetOptions {
                ids: vec![file_id.clone()],
                limit: None,
                offset: None,
                where_document: None,
                where_metadata: None,
                include: None,
            };
            
            let get_result = collection.get(get_options).await;

            match get_result {
                Ok(result) => {
                    if result.ids.is_empty() {
                        missing_files.push(file_path.clone());
                    } else {
                        println!("✅ File already embedded: {}", file_path);
                    }
                }
                Err(_) => {
                    missing_files.push(file_path.clone());
                }
            }
        }
        
        Ok(missing_files)
    }

    fn path_to_id(&self, file_path: &str) -> String {
        format!("file_{}", file_path.replace(['/', '\\', '.'], "_"))
    }

    pub async fn add_documents_with_bert_embeddings(
        &self,
        collection: &ChromaCollection,
        file_paths: Vec<String>,
    ) -> Result<()> {
        let missing_files = self.check_existing_documents(collection, &file_paths).await?;
        
        if missing_files.is_empty() {
            println!("All files are already embedded!");
            return Ok(());
        }

        println!("Embedding {} new files...", missing_files.len());

        let mut documents = vec![];
        let mut valid_paths = vec![];
        
        for file_path in &missing_files {
            if let Ok(content) = fs::read_to_string(file_path) {
                documents.push(content);
                valid_paths.push(file_path.clone());
            } else {
                println!("⚠️ Could not read file: {}", file_path);
            }
        }

        if documents.is_empty() {
            println!("No valid documents to embed");
            return Ok(());
        }

        let summary = self.gemini.generate_embedding_prompt(&documents).await?;
        
        println!("Generating BERT embeddings for {} documents...", documents.len());
        let embeddings = self.embedding_engine.generate_embeddings(documents.clone()).await?;
        println!("✅ Embeddings generated successfully!");
        
        let mut metadatas = vec![];
        let mut ids = vec![];
        
        for (i, (doc, file_path)) in documents.iter().zip(valid_paths.iter()).enumerate() {
            let mut metadata = Map::new();
            metadata.insert("summary".to_string(), Value::String(summary.clone()));
            metadata.insert("file_path".to_string(), Value::String(file_path.clone()));
            metadata.insert("doc_index".to_string(), Value::String(i.to_string()));
            metadata.insert("length".to_string(), Value::String(doc.len().to_string()));
            metadata.insert("word_count".to_string(), Value::String(doc.split_whitespace().count().to_string()));
            metadata.insert("embedding_model".to_string(), Value::String("all-MiniLM-L12-v2".to_string()));
            metadatas.push(metadata);
            ids.push(self.path_to_id(file_path));
        }

        let doc_count = documents.len();

        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

        let entries = CollectionEntries {
            ids: id_refs,
            metadatas: Some(metadatas),
            documents: Some(doc_refs),
            embeddings: Some(embeddings),
        };

        collection.add(entries, None).await?;

        println!("✅ Added {} documents with BERT embeddings", doc_count);
        Ok(())
    }

    pub async fn search_with_bert_embeddings(
        &self,
        collection: &ChromaCollection,
        query: &str,
        n_results: Option<usize>,
    ) -> Result<Vec<(String, f32, String, String)>> {
        println!("Generating query embedding...");
        let query_embedding = self.embedding_engine.generate_single_embedding(query.to_string()).await?;

        use chromadb::collection::QueryOptions;
        
        let query_options = QueryOptions {
            query_texts: None,
            query_embeddings: Some(vec![query_embedding]),
            n_results: Some(n_results.unwrap_or(5)),
            where_document: None,
            where_metadata: None,
            include: None,
        };

        let results: QueryResult = collection.query(query_options, None).await?;

        let mut relevant_docs = vec![];

        if let Some(documents) = &results.documents {
            if let Some(distances) = &results.distances {
                if let Some(metadatas) = &results.metadatas {
                    for (i, doc_vec) in documents.iter().enumerate() {
                        if let (Some(distance_vec), Some(metadata_vec)) = 
                            (distances.get(i), metadatas.get(i)) {
                            
                            for (j, doc) in doc_vec.iter().enumerate() {
                                if let (Some(&distance), Some(metadata)) = 
                                    (distance_vec.get(j), metadata_vec.get(j)) {
                                    
                                    let file_path = metadata
                                        .as_ref()
                                        .and_then(|m| m.get("file_path"))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("unknown");
                                    
                                    let doc_index = metadata
                                        .as_ref()
                                        .and_then(|m| m.get("doc_index"))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("unknown");
                                    
                                    let similarity_score = 1.0 - distance.min(1.0);
                                    
                                    relevant_docs.push((
                                        doc.clone(),
                                        similarity_score,
                                        doc_index.to_string(),
                                        file_path.to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        relevant_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(relevant_docs)
    }

    // Helper function to safely truncate strings at character boundaries
    fn safe_truncate(text: &str, max_chars: usize) -> String {
        if text.chars().count() <= max_chars {
            text.to_string()
        } else {
            text.chars().take(max_chars).collect::<String>() + "..."
        }
    }

    pub async fn interactive_bert_search(&self, collection: &ChromaCollection) -> Result<()> {
        use std::io::{self, Write};

        loop {
            print!("Enter search query for BERT similarity (or 'quit' to exit): ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            let query = input.trim();
            if query.is_empty() {
                continue;
            }
            
            if query.to_lowercase() == "quit" {
                break;
            }

            println!("\n🔍 BERT similarity search for: {}", query);
            println!("{}", "─".repeat(60));

            match self.search_with_bert_embeddings(collection, query, Some(5)).await {
                Ok(results) => {
                    if results.is_empty() {
                        println!("❌ No similar documents found.");
                    } else {
                        for (i, (document, similarity, doc_index, file_path)) in results.iter().enumerate() {
                            println!("📄 Result {} (Similarity: {:.3})", i + 1, similarity);
                            println!("📁 File: {}", file_path);
                            println!("🔢 Document Index: {}", doc_index);
                            
                            // Use safe truncation instead of byte slicing
                            let preview = Self::safe_truncate(document, 300);
                            println!("📖 Content Preview: {}", preview);
                            
                            println!("{}", "─".repeat(40));
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Search error: {}", e);
                }
            }
            
            println!();
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();

    let chroma_url = env::var("CHROMA_URL")
        .unwrap_or_else(|_| "http://localhost:8000".to_string());
    let gemini_api_key = env::var("GEMINI_API_KEY")
        .expect("GEMINI_API_KEY environment variable is required");

    println!("🚀 Initializing ChromaDB + Gemini + BERT application...");
    let app = ChromaGeminiApp::new(chroma_url, gemini_api_key).await?;

    let collection_name = "faktury_pizza_bert";
    
    let collection = app.setup_collection(collection_name).await?;

    let mut file_paths = vec![];
    let dir_path = "faktury";
    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Some(path_str) = path.to_str() {
                    file_paths.push(path_str.to_string());
                }
            }
        }
    }

    if !file_paths.is_empty() {
        println!("📂 Found {} txt files", file_paths.len());
        
        app.add_documents_with_bert_embeddings(&collection, file_paths).await?;
        println!("✅ Document processing complete!");
    } else {
        println!("❌ No txt files found in 'faktury' directory.");
    }

    println!("\n🔍 Starting interactive BERT similarity search...");
    app.interactive_bert_search(&collection).await?;

    Ok(())
}
