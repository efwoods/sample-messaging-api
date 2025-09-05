from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid
import json
import time
import logging
import asyncio
import shutil
from typing import Optional, List, Dict, Union, Any
from contextlib import asynccontextmanager
from threading import Lock
from datetime import datetime
import psutil
import gc
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom embedding function wrapper for ChromaDB
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

# Enhanced Model Manager with memory optimization and caching
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_adapter = None
        self.current_avatar = "base"
        self.model_lock = Lock()
        self._model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self._adapters_dir = "./adapters"
        self._training_data_dir = "./training_data"
        self._cache_dir = "./model_cache"
        self.adapter_cache = {}
        self.training_sessions = {}
        
    def get_memory_usage(self) -> Dict:
        """Get detailed memory usage statistics"""
        if torch.cuda.is_available():
            return {
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "system_memory_gb": psutil.virtual_memory().used / 1024**3,
                "system_memory_percent": psutil.virtual_memory().percent
            }
        return {
            "system_memory_gb": psutil.virtual_memory().used / 1024**3,
            "system_memory_percent": psutil.virtual_memory().percent
        }
    
    def optimize_memory(self):
        """Perform memory optimization"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def load_base_model(self):
        """Load base model with enhanced error handling"""
        with self.model_lock:
            try:
                # Clear existing model
                if self.model is not None:
                    del self.model
                    self.optimize_memory()
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                    torch_dtype=torch.bfloat16,
                    cache_dir=self._cache_dir
                )
                
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name, 
                        trust_remote_code=True, 
                        token=os.getenv("HUGGINGFACE_TOKEN"),
                        cache_dir=self._cache_dir
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.current_adapter = None
                logger.info("Base model loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load base model: {str(e)}")
                raise RuntimeError(f"Failed to load base model: {str(e)}")
    
    def load_adapter(self, avatar: str) -> bool:
        """Load adapter with caching and improved error handling"""
        with self.model_lock:
            if self.current_adapter == avatar:
                return True
                
            # Check cache first
            if avatar in self.adapter_cache:
                logger.info(f"Loading {avatar} from cache")
                self.model = self.adapter_cache[avatar]
                self.current_adapter = avatar
                return True
            
            if self.current_adapter is not None:
                self.load_base_model()
            
            adapter_path = os.path.join(self._adapters_dir, avatar)
            
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                try:
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                    self.current_adapter = avatar
                    logger.info(f"QLoRA adapter for {avatar} loaded successfully")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load adapter for {avatar}: {str(e)}")
                    self.load_base_model()
                    return False
            else:
                logger.info(f"No adapter found for {avatar}, using base model")
                return False

    def get_available_avatars(self) -> List[Dict]:
        """Get list of available avatars with metadata"""
        avatars = []
        
        # Add base avatar
        avatars.append({
            "name": "base",
            "has_adapter": False,
            "has_training_data": False,
            "created_at": None,
            "size_mb": 0
        })
        
        # Scan adapter directory
        if os.path.exists(self._adapters_dir):
            for avatar_dir in os.listdir(self._adapters_dir):
                adapter_path = os.path.join(self._adapters_dir, avatar_dir)
                if os.path.isdir(adapter_path):
                    config_path = os.path.join(adapter_path, "adapter_config.json")
                    has_adapter = os.path.exists(config_path)
                    
                    training_data_path = os.path.join(self._training_data_dir, avatar_dir, "dataset.jsonl")
                    has_training_data = os.path.exists(training_data_path)
                    
                    created_at = None
                    size_mb = 0
                    if has_adapter:
                        created_at = os.path.getctime(config_path)
                        size_mb = sum(os.path.getsize(os.path.join(adapter_path, f)) 
                                    for f in os.listdir(adapter_path)) / 1024 / 1024
                    
                    avatars.append({
                        "name": avatar_dir,
                        "has_adapter": has_adapter,
                        "has_training_data": has_training_data,
                        "created_at": created_at,
                        "size_mb": round(size_mb, 2)
                    })
        
        return avatars


    def ensure_base_model_loaded(self):
        """Ensure base model is loaded, load if not"""
        if self.model is None or self.tokenizer is None:
            logger.info("Base model not loaded, loading now...")
            self.load_base_model()
            return True
        return False
    
    def create_empty_adapter(self, avatar: str) -> bool:
        """Create an empty adapter for avatar without training"""
        try:
            adapter_path = os.path.join(self._adapters_dir, avatar)
            os.makedirs(adapter_path, exist_ok=True)
            
            # Ensure base model is loaded
            if self.current_adapter is not None:
                self.load_base_model()
            
            # Create LoRA configuration with default parameters
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                task_type="CAUSAL_LM"
            )
            
            with self.model_lock:
                # Apply PEFT model
                self.model = get_peft_model(self.model, lora_config)
                
                # Save the adapter (even though untrained)
                self.model.save_pretrained(adapter_path)
                self.tokenizer.save_pretrained(adapter_path)
                
                # Load the adapter back to ensure it's properly attached
                self.model = PeftModel.from_pretrained(self.model.base_model, adapter_path)
                self.current_adapter = avatar
            
            logger.info(f"Empty adapter created and attached for {avatar}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create empty adapter for {avatar}: {str(e)}")
            # Fallback to base model
            self.load_base_model()
            return False
    
    def attach_existing_adapter(self, avatar: str) -> bool:
        """Attach existing adapter to base model"""
        try:
            adapter_path = os.path.join(self._adapters_dir, avatar)
            
            # Ensure we start with base model
            if self.current_adapter is not None:
                self.load_base_model()
            
            with self.model_lock:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.current_adapter = avatar
            
            logger.info(f"Existing adapter attached for {avatar}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to attach existing adapter for {avatar}: {str(e)}")
            return False



# Enhanced ChromaDB management with better search capabilities
class ChromaDBManager:
    def __init__(self, db_path: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = SentenceTransformerEmbeddingFunction(embedding_model)
    
    def get_collection(self, avatar: str):
        """Get or create collection with proper error handling"""
        collection_name = f"documents_{avatar}"
        try:
            return self.client.get_collection(name=collection_name)
        except:
            return self.client.create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
    
    def semantic_search(self, avatar: str, query: str, max_results: int = 3, 
                       similarity_threshold: float = 0.7) -> Dict:
        """Enhanced semantic search with similarity scoring"""
        try:
            collection = self.get_collection(avatar)
            results = collection.query(
                query_texts=[query], 
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            if (results and results["documents"] and results["metadatas"] and 
                len(results["documents"]) > 0 and len(results["documents"][0]) > 0):
                
                context_parts = []
                docs = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results.get("distances", [[]])[0] if "distances" in results else []
                
                for i, (doc, metadata) in enumerate(zip(docs, metadatas)):
                    similarity = 1 - distances[i] if i < len(distances) else 1.0
                    
                    if similarity >= similarity_threshold:
                        context_parts.append({
                            "content": doc,
                            "source": metadata.get('source', 'unknown'),
                            "similarity": round(similarity, 3),
                            "metadata": metadata
                        })
                
                return {
                    "context_parts": context_parts,
                    "total_results": len(context_parts),
                    "query": query,
                    "has_results": len(context_parts) > 0
                }
            
            return {"context_parts": [], "total_results": 0, "query": query, "has_results": False}
            
        except Exception as e:
            logger.warning(f"Error retrieving context for {avatar}: {str(e)}")
            return {"context_parts": [], "total_results": 0, "query": query, "has_results": False}

    def get_collection_stats(self, avatar: str) -> Dict:
        """Get statistics about a collection"""
        try:
            collection = self.get_collection(avatar)
            count = collection.count()
            return {
                "document_count": count,
                "avatar": avatar,
                "collection_name": f"documents_{avatar}"
            }
        except Exception as e:
            return {
                "document_count": 0,
                "avatar": avatar,
                "error": str(e)
            }
        
    def ensure_collection_exists(self, avatar: str) -> tuple[bool, dict]:
        """Ensure collection exists for avatar, return (exists, stats)"""
        try:
            collection = self.get_collection(avatar)
            count = collection.count()
            stats = {
                "collection_exists": True,
                "document_count": count,
                "has_documents": count > 0,
                "avatar": avatar
            }
            logger.info(f"ChromaDB collection for {avatar}: {count} documents")
            return True, stats
        except Exception as e:
            logger.warning(f"Issue with ChromaDB collection for {avatar}: {str(e)}")
            # Try to create collection
            try:
                collection = self.client.create_collection(
                    name=f"documents_{avatar}",
                    embedding_function=self.embedding_function
                )
                stats = {
                    "collection_exists": True,
                    "document_count": 0,
                    "has_documents": False,
                    "avatar": avatar,
                    "created_new": True
                }
                return True, stats
            except Exception as create_error:
                logger.error(f"Failed to create collection for {avatar}: {str(create_error)}")
                return False, {"collection_exists": False, "error": str(create_error)}


# Conversation Memory Manager
class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.conversations = {}
        self.max_history = max_history
    
    def add_message(self, avatar: str, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        if avatar not in self.conversations:
            self.conversations[avatar] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.conversations[avatar].append(message)
        
        # Keep only recent messages
        if len(self.conversations[avatar]) > self.max_history * 2:
            self.conversations[avatar] = self.conversations[avatar][-self.max_history * 2:]
    
    def get_conversation_context(self, avatar: str, include_system: bool = True) -> str:
        """Get formatted conversation context"""
        if avatar not in self.conversations:
            return ""
        
        context_parts = []
        for msg in self.conversations[avatar][-self.max_history:]:
            if not include_system and msg["role"] == "system":
                continue
            context_parts.append(f"{msg['role'].title()}: {msg['content']}")
        
        return "\n".join(context_parts)

# WebSocket Manager for real-time updates
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict):
        if self.active_connections:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_json(message)
                except:
                    self.disconnect(connection)

# Initialize managers
model_manager = ModelManager()
chroma_manager = ChromaDBManager()
conversation_manager = ConversationManager()
websocket_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Enhanced Avatar Management System...")
    try:
        # Create necessary directories
        os.makedirs(model_manager._adapters_dir, exist_ok=True)
        os.makedirs(model_manager._training_data_dir, exist_ok=True)
        os.makedirs(model_manager._cache_dir, exist_ok=True)
        
        model_manager.load_base_model()
        logger.info("System initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Avatar Management System...")
    model_manager.optimize_memory()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Llama-3.2-1B Avatar System",
    description="Advanced Avatar-specific QLoRA adapters with ChromaDB RAG integration",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(3, ge=1, le=10)
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    use_base_model: bool = Field(False)
    include_conversation_history: bool = Field(True)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)

class StreamQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(3, ge=1, le=10)
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.1, le=2.0)

class TrainRequest(BaseModel):
    epochs: int = Field(1, ge=1, le=10)
    batch_size: int = Field(4, ge=1, le=16)
    learning_rate: float = Field(2e-4, ge=1e-5, le=1e-3)
    lora_r: int = Field(8, ge=4, le=64)
    lora_alpha: int = Field(16, ge=8, le=128)
    lora_dropout: float = Field(0.1, ge=0.0, le=0.5)

class DocumentRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=10000)
    source: str = Field("user_upload", max_length=200)
    metadata: Optional[Dict] = Field(None)

class BulkDocumentRequest(BaseModel):
    documents: List[DocumentRequest] = Field(..., min_items=1, max_items=100)

class TrainingDocumentRequest(BaseModel):
    instruction: str = Field(..., min_length=5, max_length=2000)
    response: str = Field(..., min_length=5, max_length=2000)
    source: str = Field("user_upload", max_length=200)
    category: Optional[str] = Field(None)

class SelectAvatarRequest(BaseModel):
    avatar: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_\\-\\s]+$")

class AvatarComparisonRequest(BaseModel):
    avatars: List[str] = Field(..., min_items=2, max_items=5)
    query: str = Field(..., min_length=1, max_length=1000)

class ExportRequest(BaseModel):
    avatar: str = Field(..., min_length=1, max_length=50)
    include_training_data: bool = Field(True)
    include_documents: bool = Field(True)

# Utility functions
def format_training_document(instruction: str, response: str, avatar: str) -> str:
    """Format training document with consistent structure"""
    return f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant in the style of {avatar}.<|end_header_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}<|end_header_id|>
<|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>"""

def normalize_avatar_name(avatar: str) -> str:
    """Normalize avatar name for consistency"""
    return avatar.lower().replace(" ", "_").replace("-", "_")

async def create_adapter_async(avatar: str, background_tasks: BackgroundTasks, 
                             train_config: Optional[TrainRequest] = None) -> Dict:
    """Create adapter asynchronously with configurable parameters"""
    try:
        adapter_path = os.path.join(model_manager._adapters_dir, avatar)
        
        # Ensure base model is loaded
        if model_manager.current_adapter is not None:
            model_manager.load_base_model()
        
        # Configure LoRA with custom parameters
        config = train_config or TrainRequest()
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=config.lora_dropout,
            task_type="CAUSAL_LM"
        )
        
        with model_manager.model_lock:
            model_manager.model = get_peft_model(model_manager.model, lora_config)
        
        # Check for training data
        dataset_file = os.path.join(model_manager._training_data_dir, avatar, "dataset.jsonl")
        
        if os.path.exists(dataset_file):
            # Load real training data
            dataset = load_dataset("json", data_files=dataset_file, split="train")
            
            def format_dataset(examples):
                formatted_texts = []
                for instruction, response in zip(examples["instruction"], examples["response"]):
                    text = format_training_document(instruction, response, avatar)
                    formatted_texts.append(text)
                return {"text": formatted_texts}
            
            dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
            training_message = "trained with existing data"
            epochs = config.epochs
        else:
            # Create minimal dummy dataset
            dummy_text = format_training_document(
                "Hello, can you introduce yourself?",
                f"Hello! I'm a helpful assistant ready to assist you in the style of {avatar}. How can I help you today?",
                avatar
            )
            
            from datasets import Dataset
            dataset = Dataset.from_dict({"text": [dummy_text]})
            training_message = "initialized with dummy data"
            epochs = 1
        
        os.makedirs(adapter_path, exist_ok=True)
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir=adapter_path,
            per_device_train_batch_size=config.batch_size if os.path.exists(dataset_file) else 2,
            gradient_accumulation_steps=max(1, 8 // config.batch_size),
            learning_rate=config.learning_rate,
            num_train_epochs=epochs,
            max_steps=1 if not os.path.exists(dataset_file) else -1,
            save_strategy="epoch" if os.path.exists(dataset_file) else "steps",
            save_steps=1 if not os.path.exists(dataset_file) else None,
            logging_steps=1,
            warmup_steps=max(1, len(dataset) // 10) if os.path.exists(dataset_file) else 0,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=512 if not os.path.exists(dataset_file) else 2048,
            dataset_text_field="text",
        )
        
        # Store training session info
        session_id = str(uuid.uuid4())
        model_manager.training_sessions[session_id] = {
            "avatar": avatar,
            "status": "training",
            "start_time": time.time(),
            "dataset_size": len(dataset)
        }
        
        # Broadcast training start
        await websocket_manager.broadcast({
            "type": "training_started",
            "avatar": avatar,
            "session_id": session_id
        })
        
        # Train the model
        trainer.train()
        
        # Save adapter
        with model_manager.model_lock:
            model_manager.model.save_pretrained(adapter_path)
            model_manager.tokenizer.save_pretrained(adapter_path)
        
        # Update training session
        model_manager.training_sessions[session_id]["status"] = "completed"
        model_manager.training_sessions[session_id]["end_time"] = time.time()
        
        # Load the newly created adapter
        model_manager.load_adapter(avatar)
        
        # Broadcast training completion
        await websocket_manager.broadcast({
            "type": "training_completed",
            "avatar": avatar,
            "session_id": session_id
        })
        
        return {
            "success": True,
            "message": f"Adapter created and {training_message}",
            "had_training_data": os.path.exists(dataset_file),
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to create adapter for {avatar}: {str(e)}")
        model_manager.load_base_model()
        
        # Broadcast training failure
        await websocket_manager.broadcast({
            "type": "training_failed",
            "avatar": avatar,
            "error": str(e)
        })
        
        return {
            "success": False,
            "message": f"Adapter creation failed: {str(e)}",
            "had_training_data": False
        }

# Enhanced API Endpoints

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now, can be enhanced for specific commands
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Updated select_avatar endpoint
@app.post("/select_avatar", tags=["Avatar Management"])
async def select_avatar(request: SelectAvatarRequest, background_tasks: BackgroundTasks):
    """
    Enhanced avatar selection with guaranteed base model loading,
    adapter creation/attachment, and ChromaDB initialization
    """
    avatar = normalize_avatar_name(request.avatar)
    
    try:
        # Step 1: Ensure base model is loaded
        logger.info(f"Selecting avatar: {avatar}")
        base_loaded = model_manager.ensure_base_model_loaded()
        
        # Step 2: Check if adapter exists
        adapter_path = os.path.join(model_manager._adapters_dir, avatar)
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        adapter_exists = os.path.exists(adapter_config_path)
        
        adapter_result = {
            "adapter_exists": adapter_exists,
            "adapter_attached": False,
            "adapter_created": False,
            "adapter_trained": False
        }
        
        if adapter_exists:
            # Step 3a: Attach existing adapter
            success = model_manager.load_adapter(avatar)
            adapter_result["adapter_attached"] = success
            
            # Check if adapter was trained (has training data)
            training_data_path = os.path.join(model_manager._training_data_dir, avatar, "dataset.jsonl")
            adapter_result["adapter_trained"] = os.path.exists(training_data_path)
            
        else:
            # Step 3b: Create new empty adapter
            success = model_manager.create_empty_adapter(avatar)
            adapter_result["adapter_created"] = success
            adapter_result["adapter_attached"] = success
            adapter_result["adapter_trained"] = False
        
        # Step 4: Ensure ChromaDB collection exists (even if empty)
        chroma_success, chroma_stats = chroma_manager.ensure_collection_exists(avatar)
        
        # Step 5: Update current avatar
        model_manager.current_avatar = avatar
        
        # Step 6: Broadcast status update
        await websocket_manager.broadcast({
            "type": "avatar_selected",
            "avatar": avatar,
            "base_model_loaded": model_manager.model is not None,
            "adapter_result": adapter_result,
            "chroma_stats": chroma_stats
        })
        
        # Prepare response
        status_messages = []
        if base_loaded:
            status_messages.append("base model loaded")
        
        if adapter_result["adapter_created"]:
            status_messages.append("new adapter created and attached")
        elif adapter_result["adapter_attached"]:
            trained_status = "trained" if adapter_result["adapter_trained"] else "untrained"
            status_messages.append(f"existing {trained_status} adapter attached")
        else:
            status_messages.append("adapter attachment failed, using base model")
        
        if chroma_success:
            doc_status = f"with {chroma_stats['document_count']} documents" if chroma_stats['has_documents'] else "empty"
            status_messages.append(f"ChromaDB collection ready ({doc_status})")
        else:
            status_messages.append("ChromaDB collection unavailable")
        
        return {
            "status": f"Avatar {avatar} selected successfully",
            "details": ", ".join(status_messages),
            "avatar": avatar,
            "base_model_loaded": model_manager.model is not None,
            "adapter_result": adapter_result,
            "chroma_stats": chroma_stats,
            "ready_for_queries": True
        }
        
    except Exception as e:
        logger.error(f"Error selecting avatar {avatar}: {str(e)}")
        
        # Broadcast error
        await websocket_manager.broadcast({
            "type": "avatar_selection_failed",
            "avatar": avatar,
            "error": str(e)
        })
        
        # Try to fallback to base model
        try:
            model_manager.load_base_model()
            model_manager.current_avatar = "base"
            fallback_message = "Fell back to base model"
        except:
            fallback_message = "System in unstable state"
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to select avatar {avatar}: {str(e)}. {fallback_message}"
        )


@app.delete("/avatars/{avatar}", tags=["Avatar Management"])
async def delete_avatar(avatar: str):
    """Delete an avatar and all associated data"""
    avatar = normalize_avatar_name(avatar)
    
    if avatar == "base":
        raise HTTPException(status_code=400, detail="Cannot delete base avatar")
    
    if avatar == model_manager.current_avatar:
        model_manager.current_avatar = "base"
        model_manager.load_base_model()
    
    try:
        # Delete adapter
        adapter_path = os.path.join(model_manager._adapters_dir, avatar)
        if os.path.exists(adapter_path):
            shutil.rmtree(adapter_path)
        
        # Delete training data
        training_path = os.path.join(model_manager._training_data_dir, avatar)
        if os.path.exists(training_path):
            shutil.rmtree(training_path)
        
        # Delete ChromaDB collection
        try:
            collection_name = f"documents_{avatar}"
            chroma_manager.client.delete_collection(collection_name)
        except:
            pass
        
        await websocket_manager.broadcast({
            "type": "avatar_deleted",
            "avatar": avatar
        })
        
        return {"status": f"Avatar {avatar} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete avatar: {str(e)}")

# Enhanced query method to handle untrained adapters gracefully
@app.post("/query", tags=["Model Operations"])
async def query_model(request: QueryRequest):
    """Enhanced query with better handling of untrained adapters and empty ChromaDB"""
    if not model_manager.model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Ensure correct adapter is loaded if not using base model explicitly
        if not request.use_base_model and model_manager.current_avatar != "base":
            if model_manager.current_adapter != model_manager.current_avatar:
                # Try to load the adapter, create if doesn't exist
                adapter_path = os.path.join(model_manager._adapters_dir, model_manager.current_avatar)
                if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                    logger.info(f"Adapter not found for {model_manager.current_avatar}, creating empty adapter")
                    model_manager.create_empty_adapter(model_manager.current_avatar)
                else:
                    model_manager.load_adapter(model_manager.current_avatar)
        
        # Enhanced context retrieval (works even with empty ChromaDB)
        context_start = time.time()
        search_results = chroma_manager.semantic_search(
            model_manager.current_avatar,
            request.query,
            request.max_results,
            request.similarity_threshold
        )
        context_time = time.time() - context_start
        
        # Build context from search results (gracefully handles empty results)
        context_parts = []
        if search_results["has_results"]:
            for part in search_results["context_parts"]:
                context_parts.append(f"[{part['source']} - Similarity: {part['similarity']}]\n{part['content']}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Get conversation history if requested
        conversation_context = ""
        if request.include_conversation_history:
            conversation_context = conversation_manager.get_conversation_context(model_manager.current_avatar)
        
        # Build enhanced system message
        system_message = "You are a helpful assistant"
        if not request.use_base_model and model_manager.current_avatar != "base":
            system_message += f" in the style of {model_manager.current_avatar}"
        
        # Adapt message based on available context
        if context:
            system_message += ". Use the following context to answer accurately."
        elif model_manager.current_adapter:
            # Adapter exists but no context from ChromaDB
            system_message += ". Draw upon your specialized knowledge and training."
        else:
            system_message += ". Rely on your general knowledge to be helpful."
        
        if conversation_context:
            system_message += " Consider the conversation history when appropriate."
        
        # Construct enhanced prompt
        prompt_parts = [f"<|start_header_id|>system<|end_header_id|>\n{system_message}"]
        
        if context:
            prompt_parts.append(f"Context:\n{context}")
        
        if conversation_context:
            prompt_parts.append(f"Recent Conversation:\n{conversation_context}")
        
        prompt_parts.extend([
            "<|end_header_id|>",
            f"<|start_header_id|>user<|end_header_id|>\n{request.query}<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Tokenization with error handling
        tokenize_start = time.time()
        try:
            inputs = model_manager.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            device = next(model_manager.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")
        
        tokenize_time = time.time() - tokenize_start
        
        # Generation with improved error handling
        gen_start = time.time()
        try:
            with torch.no_grad():
                outputs = model_manager.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=model_manager.tokenizer.eos_token_id,
                    eos_token_id=model_manager.tokenizer.eos_token_id
                )
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
        
        gen_time = time.time() - gen_start
        
        # Decode response
        decode_start = time.time()
        response = model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        try:
            assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        except IndexError:
            assistant_response = response[len(prompt):].strip()
        
        decode_time = time.time() - decode_start
        total_time = time.time() - start_time
        
        # Add to conversation memory
        conversation_manager.add_message(model_manager.current_avatar, "user", request.query)
        conversation_manager.add_message(model_manager.current_avatar, "assistant", assistant_response)
        
        # Calculate metrics
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(outputs[0]) - input_tokens
        tokens_per_second = output_tokens / gen_time if gen_time > 0 else 0
        
        # Determine adapter status
        adapter_status = "none"
        if model_manager.current_adapter:
            training_data_path = os.path.join(model_manager._training_data_dir, model_manager.current_avatar, "dataset.jsonl")
            adapter_status = "trained" if os.path.exists(training_data_path) else "untrained"
        
        response_data = {
            "response": assistant_response,
            "avatar": model_manager.current_avatar,
            "adapter_status": adapter_status,
            "context_used": search_results["has_results"],
            "context_source_count": len(search_results["context_parts"]),
            "using_base_model_only": request.use_base_model,
            "search_results": search_results,
            "conversation_history_used": request.include_conversation_history,
            "metrics": {
                "total_time_seconds": round(total_time, 4),
                "context_retrieval_time_seconds": round(context_time, 4),
                "tokenization_time_seconds": round(tokenize_time, 4),
                "generation_time_seconds": round(gen_time, 4),
                "decoding_time_seconds": round(decode_time, 4),
                "tokens_per_second": round(tokens_per_second, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
        
        # Broadcast query completion
        await websocket_manager.broadcast({
            "type": "query_completed",
            "avatar": model_manager.current_avatar,
            "adapter_status": adapter_status,
            "context_used": search_results["has_results"],
            "metrics": response_data["metrics"]
        })
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/train_adapter", tags=["QLoRA Adapter Management"])
async def train_adapter(request: TrainRequest):
    """Enhanced training with configurable LoRA parameters"""
    if not model_manager.model:
        raise HTTPException(status_code=500, detail="Base model not loaded")
    
    avatar = model_manager.current_avatar
    adapter_path = os.path.join(model_manager._adapters_dir, avatar)
    dataset_file = os.path.join(model_manager._training_data_dir, avatar, "dataset.jsonl")
    
    if not os.path.exists(dataset_file):
        raise HTTPException(status_code=404, detail=f"No training dataset found for {avatar}")
    
    try:
        # Reload base model if adapter is loaded
        if model_manager.current_adapter is not None:
            model_manager.load_base_model()
        
        # Configure LoRA with custom parameters
        lora_config = LoraConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=request.lora_dropout,
            task_type="CAUSAL_LM"
        )
        
        with model_manager.model_lock:
            model_manager.model = get_peft_model(model_manager.model, lora_config)
        
        # Load and format dataset
        dataset = load_dataset("json", data_files=dataset_file, split="train")
        
        def format_dataset(examples):
            formatted_texts = []
            for instruction, response in zip(examples["instruction"], examples["response"]):
                text = format_training_document(instruction, response, avatar)
                formatted_texts.append(text)
            return {"text": formatted_texts}
        
        dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names)
        
        os.makedirs(adapter_path, exist_ok=True)
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=adapter_path,
            per_device_train_batch_size=request.batch_size,
            gradient_accumulation_steps=max(1, 8 // request.batch_size),
            learning_rate=request.learning_rate,
            num_train_epochs=request.epochs,
            save_strategy="epoch",
            evaluation_strategy="no",
            logging_steps=max(1, len(dataset) // (request.batch_size * 4)),
            warmup_steps=max(1, len(dataset) // (request.batch_size * 10)),
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
            save_total_limit=2,
        )
        
        trainer = SFTTrainer(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=2048,
            dataset_text_field="text",
        )
        
        # Store training session
        session_id = str(uuid.uuid4())
        model_manager.training_sessions[session_id] = {
            "avatar": avatar,
            "status": "training",
            "start_time": time.time(),
            "dataset_size": len(dataset),
            "config": request.dict()
        }
        
        # Broadcast training start
        await websocket_manager.broadcast({
            "type": "training_started",
            "avatar": avatar,
            "session_id": session_id,
            "config": request.dict()
        })
        
        # Train and save
        train_result = trainer.train()
        
        with model_manager.model_lock:
            model_manager.model.save_pretrained(adapter_path)
            model_manager.tokenizer.save_pretrained(adapter_path)
        
        # Update session
        model_manager.training_sessions[session_id]["status"] = "completed"
        model_manager.training_sessions[session_id]["end_time"] = time.time()
        
        # Broadcast completion
        await websocket_manager.broadcast({
            "type": "training_completed",
            "avatar": avatar,
            "session_id": session_id
        })
        
        return {
            "status": f"Training completed for {avatar}",
            "training_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "epochs_completed": request.epochs,
            "dataset_size": len(dataset),
            "session_id": session_id,
            "lora_config": {
                "r": request.lora_r,
                "alpha": request.lora_alpha,
                "dropout": request.lora_dropout
            }
        }
        
    except Exception as e:
        logger.error(f"Training error for {avatar}: {str(e)}")
        model_manager.load_base_model()
        
        await websocket_manager.broadcast({
            "type": "training_failed",
            "avatar": avatar,
            "error": str(e)
        })
        
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/training_sessions", tags=["QLoRA Adapter Management"])
async def get_training_sessions():
    """Get all training sessions with their status"""
    return {
        "sessions": model_manager.training_sessions,
        "active_sessions": [
            session_id for session_id, session in model_manager.training_sessions.items()
            if session["status"] == "training"
        ]
    }

# from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
# from fastapi.responses import HTMLResponse, JSONResponse
# from typing import List, Optional, Dict, Any
# from pydantic import BaseModel
# import uuid
# import time
# import json
# import logging
# from pathlib import Path
# import asyncio
# from datetime import datetime

# # Pydantic Models
# class DocumentMetadata(BaseModel):
#     source: Optional[str] = None
#     title: Optional[str] = None
#     author: Optional[str] = None
#     category: Optional[str] = None
#     tags: Optional[List[str]] = []
#     custom_fields: Optional[Dict[str, Any]] = {}

# class DocumentRequest(BaseModel):
#     content: str
#     source: str
#     metadata: Optional[Dict[str, Any]] = {}

# class BulkDocumentRequest(BaseModel):
#     documents: List[DocumentRequest]

# class DocumentResponse(BaseModel):
#     status: str
#     document_id: str
#     avatar: str
#     filename: Optional[str] = None
#     file_size: Optional[int] = None
#     content_preview: Optional[str] = None

# class BulkDocumentResponse(BaseModel):
#     status: str
#     document_ids: List[str]
#     avatar: str
#     processed_count: int
#     failed_count: int
#     errors: Optional[List[str]] = []

# # File processing utilities
# class DocumentProcessor:
#     SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.json', '.csv', '.xml', '.html'}
#     MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
#     @staticmethod
#     async def read_file_content(file: UploadFile) -> str:
#         """Read and extract text content from uploaded file"""
#         content = await file.read()
#         file_extension = Path(file.filename).suffix.lower()
        
#         if file_extension == '.txt' or file_extension == '.md':
#             return content.decode('utf-8')
#         elif file_extension == '.json':
#             return content.decode('utf-8')
#         elif file_extension == '.csv':
#             return content.decode('utf-8')
#         elif file_extension == '.html' or file_extension == '.xml':
#             return content.decode('utf-8')
#         elif file_extension == '.pdf':
#             # For PDF, you'd need additional libraries like PyPDF2 or pdfplumber
#             # This is a placeholder - implement PDF extraction based on your needs
#             return f"PDF content extraction needed for: {file.filename}"
#         elif file_extension == '.docx':
#             # For DOCX, you'd need python-docx library
#             # This is a placeholder - implement DOCX extraction based on your needs
#             return f"DOCX content extraction needed for: {file.filename}"
#         else:
#             raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
    
#     @staticmethod
#     def validate_file(file: UploadFile) -> bool:
#         """Validate file type and size"""
#         if not file.filename:
#             return False
            
#         file_extension = Path(file.filename).suffix.lower()
#         if file_extension not in DocumentProcessor.SUPPORTED_EXTENSIONS:
#             return False
            
#         return True
    
#     @staticmethod
#     def create_metadata(file: UploadFile, custom_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
#         """Create comprehensive metadata for document"""
#         metadata = {
#             "filename": file.filename,
#             "file_size": file.size if hasattr(file, 'size') else 0,
#             "content_type": file.content_type,
#             "upload_timestamp": datetime.now().isoformat(),
#             "file_extension": Path(file.filename).suffix.lower() if file.filename else "",
#         }
        
#         if custom_metadata:
#             metadata.update(custom_metadata)
            
#         return metadata

# # FastAPI Endpoints
# @app.post("/upload_document", 
#           response_model=DocumentResponse,
#           tags=["ChromaDB Document Management"])
# async def upload_single_document(
#     file: UploadFile = File(...),
#     source: Optional[str] = Form(None),
#     title: Optional[str] = Form(None),
#     author: Optional[str] = Form(None),
#     category: Optional[str] = Form(None),
#     tags: Optional[str] = Form(None),
#     custom_metadata: Optional[str] = Form(None)
# ):
#     """Upload a single document to ChromaDB with metadata"""
#     try:
#         # Validate file
#         if not DocumentProcessor.validate_file(file):
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Unsupported file type. Supported types: {', '.join(DocumentProcessor.SUPPORTED_EXTENSIONS)}"
#             )
        
#         # Check file size
#         if hasattr(file, 'size') and file.size > DocumentProcessor.MAX_FILE_SIZE:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"File too large. Maximum size: {DocumentProcessor.MAX_FILE_SIZE / (1024*1024)}MB"
#             )
        
#         # Read file content
#         content = await DocumentProcessor.read_file_content(file)
        
#         # Parse custom metadata
#         parsed_custom_metadata = {}
#         if custom_metadata:
#             try:
#                 parsed_custom_metadata = json.loads(custom_metadata)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
#         # Parse tags
#         parsed_tags = []
#         if tags:
#             parsed_tags = [tag.strip() for tag in tags.split(',')]
        
#         # Create comprehensive metadata
#         file_metadata = DocumentProcessor.create_metadata(file, parsed_custom_metadata)
        
#         # Add form data to metadata
#         if source:
#             file_metadata["source"] = source
#         if title:
#             file_metadata["title"] = title
#         if author:
#             file_metadata["author"] = author
#         if category:
#             file_metadata["category"] = category
#         if parsed_tags:
#             file_metadata["tags"] = parsed_tags
        
#         # Add to ChromaDB
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
#         doc_id = str(uuid.uuid4())
        
#         chroma_metadata = {
#             "source": source or file.filename,
#             "avatar": model_manager.current_avatar,
#             "created_at": time.time(),
#             "document_id": doc_id,
#             **file_metadata
#         }
        
#         collection.add(
#             documents=[content],
#             ids=[doc_id],
#             metadatas=[chroma_metadata]
#         )
        
#         # Broadcast update
#         await websocket_manager.broadcast({
#             "type": "document_uploaded",
#             "avatar": model_manager.current_avatar,
#             "document_id": doc_id,
#             "filename": file.filename
#         })
        
#         return DocumentResponse(
#             status="Document uploaded successfully",
#             document_id=doc_id,
#             avatar=model_manager.current_avatar,
#             filename=file.filename,
#             file_size=file.size if hasattr(file, 'size') else 0,
#             content_preview=content[:200] + "..." if len(content) > 200 else content
#         )
        
#     except Exception as e:
#         logger.error(f"Error uploading document: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

# @app.post("/upload_documents_bulk",
#           response_model=BulkDocumentResponse,
#           tags=["ChromaDB Document Management"])
# async def upload_multiple_documents(
#     files: List[UploadFile] = File(...),
#     source: Optional[str] = Form(None),
#     category: Optional[str] = Form(None),
#     tags: Optional[str] = Form(None),
#     custom_metadata: Optional[str] = Form(None)
# ):
#     """Upload multiple documents to ChromaDB in bulk"""
#     try:
#         if len(files) > 50:  # Limit bulk uploads
#             raise HTTPException(status_code=400, detail="Maximum 50 files allowed per bulk upload")
        
#         # Parse common metadata
#         parsed_custom_metadata = {}
#         if custom_metadata:
#             try:
#                 parsed_custom_metadata = json.loads(custom_metadata)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
#         parsed_tags = []
#         if tags:
#             parsed_tags = [tag.strip() for tag in tags.split(',')]
        
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
        
#         documents = []
#         doc_ids = []
#         metadatas = []
#         processed_count = 0
#         failed_count = 0
#         errors = []
        
#         for file in files:
#             try:
#                 # Validate each file
#                 if not DocumentProcessor.validate_file(file):
#                     errors.append(f"Unsupported file type: {file.filename}")
#                     failed_count += 1
#                     continue
                
#                 # Read content
#                 content = await DocumentProcessor.read_file_content(file)
                
#                 # Create metadata
#                 file_metadata = DocumentProcessor.create_metadata(file, parsed_custom_metadata)
                
#                 # Add form data to metadata
#                 if source:
#                     file_metadata["source"] = source
#                 if category:
#                     file_metadata["category"] = category
#                 if parsed_tags:
#                     file_metadata["tags"] = parsed_tags
                
#                 doc_id = str(uuid.uuid4())
                
#                 chroma_metadata = {
#                     "source": source or file.filename,
#                     "avatar": model_manager.current_avatar,
#                     "created_at": time.time(),
#                     "document_id": doc_id,
#                     **file_metadata
#                 }
                
#                 documents.append(content)
#                 doc_ids.append(doc_id)
#                 metadatas.append(chroma_metadata)
#                 processed_count += 1
                
#             except Exception as e:
#                 errors.append(f"Error processing {file.filename}: {str(e)}")
#                 failed_count += 1
#                 logger.error(f"Error processing file {file.filename}: {str(e)}")
        
#         # Add all valid documents to ChromaDB
#         if documents:
#             collection.add(
#                 documents=documents,
#                 ids=doc_ids,
#                 metadatas=metadatas
#             )
            
#             # Broadcast update
#             await websocket_manager.broadcast({
#                 "type": "documents_bulk_uploaded",
#                 "avatar": model_manager.current_avatar,
#                 "count": processed_count,
#                 "failed_count": failed_count
#             })
        
#         return BulkDocumentResponse(
#             status=f"Bulk upload completed. {processed_count} successful, {failed_count} failed",
#             document_ids=doc_ids,
#             avatar=model_manager.current_avatar,
#             processed_count=processed_count,
#             failed_count=failed_count,
#             errors=errors if errors else None
#         )
        
#     except Exception as e:
#         logger.error(f"Error in bulk document upload: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error in bulk upload: {str(e)}")

# @app.get("/supported_formats", tags=["ChromaDB Document Management"])
# async def get_supported_formats():
#     """Get list of supported file formats"""
#     return {
#         "supported_extensions": list(DocumentProcessor.SUPPORTED_EXTENSIONS),
#         "max_file_size_mb": DocumentProcessor.MAX_FILE_SIZE / (1024*1024),
#         "max_bulk_files": 50
#     }

# @app.post("/upload_text_content",
#           response_model=DocumentResponse,
#           tags=["ChromaDB Document Management"])
# async def upload_text_content(
#     content: str = Form(...),
#     source: str = Form(...),
#     title: Optional[str] = Form(None),
#     author: Optional[str] = Form(None),
#     category: Optional[str] = Form(None),
#     tags: Optional[str] = Form(None),
#     custom_metadata: Optional[str] = Form(None)
# ):
#     """Upload raw text content directly (no file)"""
#     try:
#         # Parse custom metadata
#         parsed_custom_metadata = {}
#         if custom_metadata:
#             try:
#                 parsed_custom_metadata = json.loads(custom_metadata)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
#         # Parse tags
#         parsed_tags = []
#         if tags:
#             parsed_tags = [tag.strip() for tag in tags.split(',')]
        
#         # Create metadata
#         doc_metadata = {
#             "upload_timestamp": datetime.now().isoformat(),
#             "content_type": "text/plain",
#             "content_length": len(content),
#             **parsed_custom_metadata
#         }
        
#         # Add form data to metadata
#         if title:
#             doc_metadata["title"] = title
#         if author:
#             doc_metadata["author"] = author
#         if category:
#             doc_metadata["category"] = category
#         if parsed_tags:
#             doc_metadata["tags"] = parsed_tags
        
#         # Add to ChromaDB
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
#         doc_id = str(uuid.uuid4())
        
#         chroma_metadata = {
#             "source": source,
#             "avatar": model_manager.current_avatar,
#             "created_at": time.time(),
#             "document_id": doc_id,
#             **doc_metadata
#         }
        
#         collection.add(
#             documents=[content],
#             ids=[doc_id],
#             metadatas=[chroma_metadata]
#         )
        
#         # Broadcast update
#         await websocket_manager.broadcast({
#             "type": "text_content_uploaded",
#             "avatar": model_manager.current_avatar,
#             "document_id": doc_id,
#             "source": source
#         })
        
#         return DocumentResponse(
#             status="Text content uploaded successfully",
#             document_id=doc_id,
#             avatar=model_manager.current_avatar,
#             content_preview=content[:200] + "..." if len(content) > 200 else content
#         )
        
#     except Exception as e:
#         logger.error(f"Error uploading text content: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error uploading text content: {str(e)}")

# @app.get("/upload_form", response_class=HTMLResponse, tags=["ChromaDB Document Management"])
# async def get_upload_form():
#     """Serve HTML form for document upload"""
#     html_content = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>ChromaDB Document Upload</title>
#         <style>
#             body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
#             .form-group { margin-bottom: 15px; }
#             label { display: block; margin-bottom: 5px; font-weight: bold; }
#             input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
#             button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
#             button:hover { background: #45a049; }
#             .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 4px; }
#             .file-list { margin-top: 10px; }
#             .file-item { background: #f5f5f5; padding: 5px 10px; margin: 5px 0; border-radius: 3px; }
#         </style>
#     </head>
#     <body>
#         <h1>ChromaDB Document Upload</h1>
        
#         <h2>Single Document Upload</h2>
#         <form id="singleForm" enctype="multipart/form-data">
#             <div class="form-group">
#                 <label>File:</label>
#                 <input type="file" name="file" required accept=".txt,.md,.pdf,.docx,.json,.csv,.xml,.html">
#             </div>
#             <div class="form-group">
#                 <label>Source:</label>
#                 <input type="text" name="source" placeholder="Document source">
#             </div>
#             <div class="form-group">
#                 <label>Title:</label>
#                 <input type="text" name="title" placeholder="Document title">
#             </div>
#             <div class="form-group">
#                 <label>Author:</label>
#                 <input type="text" name="author" placeholder="Document author">
#             </div>
#             <div class="form-group">
#                 <label>Category:</label>
#                 <input type="text" name="category" placeholder="Document category">
#             </div>
#             <div class="form-group">
#                 <label>Tags (comma-separated):</label>
#                 <input type="text" name="tags" placeholder="tag1, tag2, tag3">
#             </div>
#             <button type="submit">Upload Single Document</button>
#         </form>
        
#         <hr style="margin: 40px 0;">
        
#         <h2>Bulk Document Upload</h2>
#         <form id="bulkForm" enctype="multipart/form-data">
#             <div class="form-group">
#                 <label>Files (multiple):</label>
#                 <input type="file" name="files" multiple required accept=".txt,.md,.pdf,.docx,.json,.csv,.xml,.html">
#             </div>
#             <div class="form-group">
#                 <label>Source:</label>
#                 <input type="text" name="source" placeholder="Common source for all documents">
#             </div>
#             <div class="form-group">
#                 <label>Category:</label>
#                 <input type="text" name="category" placeholder="Common category">
#             </div>
#             <div class="form-group">
#                 <label>Tags (comma-separated):</label>
#                 <input type="text" name="tags" placeholder="tag1, tag2, tag3">
#             </div>
#             <button type="submit">Upload Multiple Documents</button>
#         </form>
        
#         <hr style="margin: 40px 0;">
        
#         <h2>Text Content Upload</h2>
#         <form id="textForm">
#             <div class="form-group">
#                 <label>Content:</label>
#                 <textarea name="content" rows="8" required placeholder="Enter your text content here..."></textarea>
#             </div>
#             <div class="form-group">
#                 <label>Source:</label>
#                 <input type="text" name="source" required placeholder="Content source">
#             </div>
#             <div class="form-group">
#                 <label>Title:</label>
#                 <input type="text" name="title" placeholder="Content title">
#             </div>
#             <div class="form-group">
#                 <label>Author:</label>
#                 <input type="text" name="author" placeholder="Content author">
#             </div>
#             <button type="submit">Upload Text Content</button>
#         </form>
        
#         <div id="results" style="margin-top: 20px;"></div>
        
#         <script>
#             function showResult(message, isError = false) {
#                 const results = document.getElementById('results');
#                 results.innerHTML = `<div style="padding: 10px; border-radius: 4px; background: ${isError ? '#ffebee' : '#e8f5e9'}; color: ${isError ? '#c62828' : '#2e7d32'};">${message}</div>`;
#             }
            
#             document.getElementById('singleForm').addEventListener('submit', async (e) => {
#                 e.preventDefault();
#                 const formData = new FormData(e.target);
#                 try {
#                     const response = await fetch('/upload_document', { method: 'POST', body: formData });
#                     const result = await response.json();
#                     showResult(`Success: ${result.status} (ID: ${result.document_id})`);
#                 } catch (error) {
#                     showResult(`Error: ${error.message}`, true);
#                 }
#             });
            
#             document.getElementById('bulkForm').addEventListener('submit', async (e) => {
#                 e.preventDefault();
#                 const formData = new FormData(e.target);
#                 try {
#                     const response = await fetch('/upload_documents_bulk', { method: 'POST', body: formData });
#                     const result = await response.json();
#                     showResult(`Success: ${result.status}`);
#                 } catch (error) {
#                     showResult(`Error: ${error.message}`, true);
#                 }
#             });
            
#             document.getElementById('textForm').addEventListener('submit', async (e) => {
#                 e.preventDefault();
#                 const formData = new FormData(e.target);
#                 try {
#                     const response = await fetch('/upload_text_content', { method: 'POST', body: formData });
#                     const result = await response.json();
#                     showResult(`Success: ${result.status} (ID: ${result.document_id})`);
#                 } catch (error) {
#                     showResult(`Error: ${error.message}`, true);
#                 }
#             });
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

# # Additional utility endpoints
# @app.get("/documents/search", tags=["ChromaDB Document Management"])
# async def search_documents(
#     query: str,
#     limit: int = 10,
#     category: Optional[str] = None,
#     author: Optional[str] = None
# ):
#     """Search documents in ChromaDB with optional filters"""
#     try:
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
        
#         # Build where clause for filtering
#         where_clause = {"avatar": model_manager.current_avatar}
#         if category:
#             where_clause["category"] = category
#         if author:
#             where_clause["author"] = author
        
#         results = collection.query(
#             query_texts=[query],
#             n_results=limit,
#             where=where_clause if len(where_clause) > 1 else None
#         )
        
#         return {
#             "query": query,
#             "results": results,
#             "count": len(results.get('documents', []))
#         }
        
#     except Exception as e:
#         logger.error(f"Error searching documents: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

# @app.get("/documents/stats", tags=["ChromaDB Document Management"])
# async def get_document_stats():
#     """Get statistics about documents in ChromaDB"""
#     try:
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
        
#         # Get all documents for this avatar
#         all_docs = collection.get(where={"avatar": model_manager.current_avatar})
        
#         total_docs = len(all_docs.get('documents', []))
        
#         # Analyze metadata
#         categories = {}
#         authors = {}
#         file_types = {}
        
#         for metadata in all_docs.get('metadatas', []):
#             if 'category' in metadata:
#                 categories[metadata['category']] = categories.get(metadata['category'], 0) + 1
#             if 'author' in metadata:
#                 authors[metadata['author']] = authors.get(metadata['author'], 0) + 1
#             if 'file_extension' in metadata:
#                 file_types[metadata['file_extension']] = file_types.get(metadata['file_extension'], 0) + 1
        
#         return {
#             "avatar": model_manager.current_avatar,
#             "total_documents": total_docs,
#             "categories": categories,
#             "authors": authors,
#             "file_types": file_types
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting document stats: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error getting document stats: {str(e)}")
    
@app.get("/documents", tags=["ChromaDB Document Management"])
async def list_documents(limit: int = 10, offset: int = 0):
    """List documents with pagination"""
    try:
        collection = chroma_manager.get_collection(model_manager.current_avatar)
        
        # Get all documents (ChromaDB doesn't have built-in pagination)
        results = collection.get(include=['documents', 'metadatas'])
        
        if not results or not results.get('documents'):
            return {
                "documents": [],
                "total": 0,
                "avatar": model_manager.current_avatar,
                "page_info": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": False
                }
            }
        
        # Manual pagination
        total = len(results['documents'])
        start_idx = offset
        end_idx = min(offset + limit, total)
        
        paginated_docs = []
        for i in range(start_idx, end_idx):
            paginated_docs.append({
                "id": results['ids'][i] if 'ids' in results else f"doc_{i}",
                "content": results['documents'][i][:200] + "..." if len(results['documents'][i]) > 200 else results['documents'][i],
                "metadata": results['metadatas'][i] if 'metadatas' in results else {}
            })
        
        return {
            "documents": paginated_docs,
            "total": total,
            "avatar": model_manager.current_avatar,
            "page_info": {
                "limit": limit,
                "offset": offset,
                "has_more": end_idx < total
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/documents/stats", tags=["ChromaDB Document Management"])
async def get_document_stats():
    """Get document statistics for current avatar"""
    try:
        stats = chroma_manager.get_collection_stats(model_manager.current_avatar)
        return stats
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.delete("/documents/{document_id}", tags=["ChromaDB Document Management"])
async def delete_document(document_id: str):
    """Delete a specific document"""
    try:
        collection = chroma_manager.get_collection(model_manager.current_avatar)
        collection.delete(ids=[document_id])
        
        await websocket_manager.broadcast({
            "type": "document_deleted",
            "avatar": model_manager.current_avatar,
            "document_id": document_id
        })
        
        return {"status": "Document deleted successfully", "id": document_id}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Training Data Management
@app.post("/add_training_data", tags=["Training Data Management"])
async def add_training_data(request: TrainingDocumentRequest):
    """Add training data with enhanced metadata"""
    avatar = model_manager.current_avatar
    training_dir = os.path.join(model_manager._training_data_dir, avatar)
    os.makedirs(training_dir, exist_ok=True)
    
    dataset_file = os.path.join(training_dir, "dataset.jsonl")
    
    training_entry = {
        "instruction": request.instruction,
        "response": request.response,
        "source": request.source,
        "category": request.category,
        "created_at": time.time(),
        "id": str(uuid.uuid4())
    }
    
    try:
        with open(dataset_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_entry, ensure_ascii=False) + "\n")
        
        await websocket_manager.broadcast({
            "type": "training_data_added",
            "avatar": avatar,
            "entry_id": training_entry["id"]
        })
        
        return {
            "status": "Training data added successfully",
            "id": training_entry["id"],
            "avatar": avatar
        }
        
    except Exception as e:
        logger.error(f"Error adding training data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding training data: {str(e)}")

@app.get("/training_data", tags=["Training Data Management"])
async def list_training_data(limit: int = 10, offset: int = 0):
    """List training data with pagination"""
    avatar = model_manager.current_avatar
    dataset_file = os.path.join(model_manager._training_data_dir, avatar, "dataset.jsonl")
    
    if not os.path.exists(dataset_file):
        return {
            "training_data": [],
            "total": 0,
            "avatar": avatar,
            "page_info": {
                "limit": limit,
                "offset": offset,
                "has_more": False
            }
        }
    
    try:
        training_entries = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    training_entries.append(json.loads(line))
        
        total = len(training_entries)
        start_idx = offset
        end_idx = min(offset + limit, total)
        paginated_data = training_entries[start_idx:end_idx]
        
        return {
            "training_data": paginated_data,
            "total": total,
            "avatar": avatar,
            "page_info": {
                "limit": limit,
                "offset": offset,
                "has_more": end_idx < total
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing training data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing training data: {str(e)}")

@app.get("/system/memory", tags=["System"])
async def get_memory_usage():
    """Get detailed system memory usage"""
    return {
        "memory_usage": model_manager.get_memory_usage(),
        "model_loaded": model_manager.model is not None,
        "current_avatar": model_manager.current_avatar,
        "adapter_loaded": model_manager.current_adapter
    }

@app.post("/system/optimize", tags=["System"])
async def optimize_system():
    """Perform system optimization"""
    try:
        memory_before = model_manager.get_memory_usage()
        model_manager.optimize_memory()
        memory_after = model_manager.get_memory_usage()
        
        return {
            "status": "System optimization completed",
            "memory_before": memory_before,
            "memory_after": memory_after
        }
        
    except Exception as e:
        logger.error(f"Error optimizing system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/status", tags=["System"])
async def get_status():
    """Enhanced system status with comprehensive information"""
    return {
        "current_avatar": model_manager.current_avatar,
        "adapter_loaded": model_manager.current_adapter,
        "model_loaded": model_manager.model is not None,
        "tokenizer_loaded": model_manager.tokenizer is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_usage": model_manager.get_memory_usage(),
        "available_avatars": len(model_manager.get_available_avatars()),
        "active_connections": len(websocket_manager.active_connections),
        "training_sessions": {
            "total": len(model_manager.training_sessions),
            "active": len([s for s in model_manager.training_sessions.values() if s["status"] == "training"])
        },
        "system_info": {
            "python_version": os.sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Enhanced health check with system validation"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {
                "model_loaded": model_manager.model is not None,
                "tokenizer_loaded": model_manager.tokenizer is not None,
                "chroma_accessible": True,
                "memory_ok": True
            }
        }
        
        # Check ChromaDB
        try:
            chroma_manager.client.heartbeat()
        except:
            health_status["checks"]["chroma_accessible"] = False
            health_status["status"] = "degraded"
        
        # Check memory usage
        memory_usage = model_manager.get_memory_usage()
        if memory_usage.get("system_memory_percent", 0) > 95:
            health_status["checks"]["memory_ok"] = False
            health_status["status"] = "degraded"
        
        if not all(health_status["checks"].values()):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

    
# Additional utility endpoint to check avatar readiness
@app.get("/avatar_status/{avatar}", tags=["Avatar Management"])
async def get_avatar_status(avatar: str):
    """Get detailed status of a specific avatar"""
    avatar = normalize_avatar_name(avatar)
    
    # Check adapter status
    adapter_path = os.path.join(model_manager._adapters_dir, avatar)
    adapter_exists = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    
    # Check training data
    training_data_path = os.path.join(model_manager._training_data_dir, avatar, "dataset.jsonl")
    training_data_exists = os.path.exists(training_data_path)
    training_data_count = 0
    
    if training_data_exists:
        try:
            with open(training_data_path, "r", encoding="utf-8") as f:
                training_data_count = sum(1 for line in f if line.strip())
        except:
            training_data_count = 0
    
    # Check ChromaDB collection
    chroma_success, chroma_stats = chroma_manager.ensure_collection_exists(avatar)
    
    return {
        "avatar": avatar,
        "is_current": avatar == model_manager.current_avatar,
        "adapter": {
            "exists": adapter_exists,
            "loaded": model_manager.current_adapter == avatar,
            "trained": training_data_exists,
            "training_examples": training_data_count
        },
        "chromadb": chroma_stats,
        "ready_for_use": True,  # Always ready with this new system
        "recommended_action": (
            "Ready to use" if adapter_exists and training_data_exists 
            else "Consider adding training data" if adapter_exists 
            else "Will create adapter on selection"
        )
    }