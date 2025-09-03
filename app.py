from fastapi import FastAPI, HTTPException, BackgroundTasks
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
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import asyncio
from threading import Lock

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

# Thread-safe model manager
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
        
    def load_base_model(self):
        """Load base model with error handling and validation"""
        with self.model_lock:
            try:
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
                    torch_dtype=torch.bfloat16
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name, 
                    trust_remote_code=True, 
                    token=os.getenv("HUGGINGFACE_TOKEN")
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
        """Load adapter with improved error handling"""
        with self.model_lock:
            if self.current_adapter == avatar:
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
                    self.load_base_model()  # Fallback to base model
                    return False
            else:
                logger.info(f"No adapter found for {avatar}, using base model")
                return False

# Global model manager instance
model_manager = ModelManager()

# Enhanced ChromaDB management
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
    
    def search_with_fallback(self, avatar: str, query: str, max_results: int = 3) -> tuple[str, bool]:
        """Search with fallback handling"""
        try:
            collection = self.get_collection(avatar)
            results = collection.query(query_texts=[query], n_results=max_results)
            
            if (results and results["documents"] and results["metadatas"] and 
                len(results["documents"]) > 0 and len(results["documents"][0]) > 0):
                
                context = ""
                docs = results["documents"][0]
                metadatas = results["metadatas"][0]
                
                for doc, metadata in zip(docs, metadatas):
                    context += f"Source: {metadata.get('source', 'unknown')}\n{doc}\n\n"
                
                return context, True
            return "", False
            
        except Exception as e:
            logger.warning(f"Error retrieving context for {avatar}: {str(e)}")
            return "", False

# Initialize managers
chroma_manager = ChromaDBManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Avatar Management System...")
    try:
        model_manager.load_base_model()
        logger.info("System initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Avatar Management System...")
    # Add cleanup logic here if needed

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Enhanced Llama-3.2-1B Avatar System",
    description="Avatar-specific QLoRA adapters with ChromaDB RAG integration",
    version="2.0.0",
    lifespan=lifespan
)

# Enhanced Pydantic models with validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The query to process")
    max_results: int = Field(3, ge=1, le=10, description="Maximum context documents to retrieve")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    use_base_model: bool = Field(False, description="Force use of base model only")

class TrainRequest(BaseModel):
    epochs: int = Field(1, ge=1, le=10, description="Number of training epochs")
    batch_size: int = Field(4, ge=1, le=16, description="Training batch size")
    learning_rate: float = Field(2e-4, ge=1e-5, le=1e-3, description="Learning rate")

class DocumentRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=10000)
    source: str = Field("user_upload", max_length=200)
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

class TrainingDocumentRequest(BaseModel):
    instruction: str = Field(..., min_length=5, max_length=2000)
    response: str = Field(..., min_length=5, max_length=2000)
    source: str = Field("user_upload", max_length=200)
    category: Optional[str] = Field(None, description="Training category")

class SelectAvatarRequest(BaseModel):
    avatar: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_\\-\\s]+$")

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

async def create_adapter_async(avatar: str, background_tasks: BackgroundTasks) -> Dict:
    """Create adapter asynchronously with better error handling"""
    try:
        adapter_path = os.path.join(model_manager._adapters_dir, avatar)
        
        # Ensure base model is loaded
        if model_manager.current_adapter is not None:
            model_manager.load_base_model()
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
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
            epochs = 1
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
            per_device_train_batch_size=2 if not os.path.exists(dataset_file) else 4,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            num_train_epochs=epochs,
            max_steps=1 if not os.path.exists(dataset_file) else -1,
            save_strategy="epoch" if os.path.exists(dataset_file) else "steps",
            save_steps=1 if not os.path.exists(dataset_file) else None,
            logging_steps=1,
            warmup_steps=0,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        trainer = SFTTrainer(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=512 if not os.path.exists(dataset_file) else 2048,
            dataset_text_field="text",
        )
        
        # Train the model
        trainer.train()
        
        # Save adapter
        with model_manager.model_lock:
            model_manager.model.save_pretrained(adapter_path)
            model_manager.tokenizer.save_pretrained(adapter_path)
        
        # Load the newly created adapter
        model_manager.load_adapter(avatar)
        
        return {
            "success": True,
            "message": f"Adapter created and {training_message}",
            "had_training_data": os.path.exists(dataset_file)
        }
        
    except Exception as e:
        logger.error(f"Failed to create adapter for {avatar}: {str(e)}")
        # Ensure we reload base model on failure
        model_manager.load_base_model()
        return {
            "success": False,
            "message": f"Adapter creation failed: {str(e)}",
            "had_training_data": False
        }

# API Endpoints

@app.post("/select_avatar", tags=["Avatar Management"])
async def select_avatar(request: SelectAvatarRequest, background_tasks: BackgroundTasks):
    """Select and initialize avatar with improved async handling"""
    avatar = normalize_avatar_name(request.avatar)
    model_manager.current_avatar = avatar
    
    adapter_path = os.path.join(model_manager._adapters_dir, avatar)
    adapter_exists = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    
    if adapter_exists:
        success = model_manager.load_adapter(avatar)
        return {
            "status": f"Avatar {avatar} selected and adapter loaded" if success else f"Avatar {avatar} selected, adapter load failed",
            "avatar": avatar,
            "adapter_loaded": success,
            "adapter_created": False
        }
    else:
        # Create adapter asynchronously
        result = await create_adapter_async(avatar, background_tasks)
        
        return {
            "status": f"Avatar {avatar} selected. {result['message']}",
            "avatar": avatar,
            "adapter_loaded": result['success'],
            "adapter_created": True,
            "had_training_data": result['had_training_data']
        }

@app.post("/query", tags=["Model Operations"])
async def query_model(request: QueryRequest):
    """Enhanced query with better error handling and metrics"""
    if not model_manager.model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Ensure correct adapter is loaded
        if not request.use_base_model and model_manager.current_avatar != "base":
            if model_manager.current_avatar != model_manager.current_adapter:
                model_manager.load_adapter(model_manager.current_avatar)
        
        # Retrieve context
        context_start = time.time()
        context, context_used = chroma_manager.search_with_fallback(
            model_manager.current_avatar, request.query, request.max_results
        )
        context_time = time.time() - context_start
        
        # Build system message
        system_message = "You are a helpful assistant"
        if not request.use_base_model and model_manager.current_avatar != "base":
            system_message += f" in the style of {model_manager.current_avatar}"
        
        if context_used:
            system_message += ". Use the following context to answer accurately."
        else:
            system_message += ". Rely on your general knowledge."
        
        # Construct prompt
        prompt = f"""<|start_header_id|>system<|end_header_id|>
{system_message}

Context:
{context}
<|end_header_id|>
<|start_header_id|>user<|end_header_id|>
{request.query}
<|end_header_id|>
<|start_header_id|>assistant<|end_header_id|>"""

        # Tokenization
        tokenize_start = time.time()
        inputs = model_manager.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        
        device = next(model_manager.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tokenize_time = time.time() - tokenize_start
        
        # Generation
        gen_start = time.time()
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
        
        # Calculate metrics
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(outputs[0]) - input_tokens
        tokens_per_second = output_tokens / gen_time if gen_time > 0 else 0
        
        return {
            "response": assistant_response,
            "avatar": model_manager.current_avatar,
            "context_used": context_used,
            "adapter_loaded": model_manager.current_adapter is not None,
            "using_base_model_only": request.use_base_model,
            "context": context if context_used else "",
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
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/train_adapter", tags=["QLoRA Adapter Management"])
async def train_adapter(request: TrainRequest):
    """Enhanced training with configurable parameters"""
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
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
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
        )
        
        trainer = SFTTrainer(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=2048,
            dataset_text_field="text",
        )
        
        # Train and save
        train_result = trainer.train()
        
        with model_manager.model_lock:
            model_manager.model.save_pretrained(adapter_path)
            model_manager.tokenizer.save_pretrained(adapter_path)
        
        return {
            "status": f"Training completed for {avatar}",
            "training_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "epochs_completed": request.epochs,
            "dataset_size": len(dataset)
        }
        
    except Exception as e:
        logger.error(f"Training error for {avatar}: {str(e)}")
        # Ensure base model is reloaded on failure
        model_manager.load_base_model()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Document management endpoints (enhanced with better error handling)
@app.post("/add_document", tags=["ChromaDB Document Management"])
async def add_document(request: DocumentRequest):
    """Add document with enhanced metadata"""
    try:
        collection = chroma_manager.get_collection(model_manager.current_avatar)
        doc_id = str(uuid.uuid4())
        
        metadata = {
            "source": request.source,
            "avatar": model_manager.current_avatar,
            "created_at": time.time()
        }
        if request.metadata:
            metadata.update(request.metadata)
        
        collection.add(
            documents=[request.content],
            ids=[doc_id],
            metadatas=[metadata]
        )
        
        return {"status": "Document added successfully", "id": doc_id}
        
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.get("/status", tags=["System"])
async def get_status():
    """Enhanced system status"""
    return {
        "current_avatar": model_manager.current_avatar,
        "adapter_loaded": model_manager.current_adapter,
        "model_loaded": model_manager.model is not None,
        "tokenizer_loaded": model_manager.tokenizer is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_memory": {
            "allocated": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "cached": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        } if torch.cuda.is_available() else None,
        "system_info": {
            "python_version": os.sys.version,
            "pytorch_version": torch.__version__
        }
    }

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)