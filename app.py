# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Initialize FastAPI app
app = FastAPI(title="Llama-3.2-1B-Instruct with Avatar-Specific QLoRA and ChromaDB")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_results: int = 3
    max_tokens: int = 512
    temperature: float = 0.7

class TrainRequest(BaseModel):
    epochs: int = 1
    batch_size: int = 4

class DocumentRequest(BaseModel):
    content: str
    source: str = "user_upload"

class DocumentsRequest(BaseModel):
    documents: list[DocumentRequest]

class TrainingDocumentRequest(BaseModel):
    instruction: str
    response: str
    source: str = "user_upload"

class TrainingDocumentsRequest(BaseModel):
    documents: list[TrainingDocumentRequest]

class UpdateDocumentRequest(BaseModel):
    id: str
    content: str
    source: str = "user_upload"

class UpdateTrainingDocumentRequest(BaseModel):
    id: str
    instruction: str
    response: str
    source: str = "user_upload"

class DeleteDocumentRequest(BaseModel):
    id: str

class SelectAvatarRequest(BaseModel):
    avatar: str

# Global variables
model = None
tokenizer = None
current_adapter = None  # Tracks currently loaded adapter (None for base model)
current_avatar = "base"  # Tracks selected avatar
adapters_dir = "./adapters"
training_data_dir = "./training_data"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = SentenceTransformer(model_name="all-MiniLM-L6-v2")

# Load base model and tokenizer
def load_base_model():
    global model, tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=os.getenv("HUGGINGFACE_TOKEN"))
        print("Base model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load base model: {str(e)}")

# Load QLoRA adapter for a specific avatar
def load_adapter(avatar: str):
    global model, current_adapter
    adapter_path = os.path.join(adapters_dir, avatar)
    
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            current_adapter = avatar
            print(f"QLoRA adapter for {avatar} loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load QLoRA adapter for {avatar}: {str(e)}")
    else:
        print(f"No QLoRA adapter found for {avatar}. Using base model.")
        current_adapter = None

# Get or create ChromaDB collection for an avatar
def get_collection(avatar: str):
    collection_name = f"documents_{avatar}"
    try:
        return chroma_client.get_collection(name=collection_name)
    except:
        return chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)

# Initialize base model at startup
load_base_model()

@app.post("/select_avatar")
async def select_avatar(request: SelectAvatarRequest):
    global current_avatar, current_adapter
    current_avatar = request.avatar
    if current_avatar != current_adapter:
        load_adapter(current_avatar)
    return {"status": f"Avatar {current_avatar} selected."}

# Chroma DB 
@app.post("/add_document")
async def add_document(request: DocumentRequest):
    try:
        collection = get_collection(current_avatar)
        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[request.content],
            ids=[doc_id],
            metadatas=[{"source": request.source, "avatar": current_avatar}]
        )
        return {"status": "Document added", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

# Chroma DB 
@app.get("/list_documents")
async def list_documents():
    try:
        collection = get_collection(current_avatar)
        results = collection.get(include=["metadatas", "documents"])
        return {"documents": [
            {"id": id, "content": doc, "metadata": meta}
            for id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

# Chroma DB 
@app.put("/update_document")
async def update_document(request: UpdateDocumentRequest):
    try:
        collection = get_collection(current_avatar)
        collection.update(
            ids=[request.id],
            documents=[request.content],
            metadatas=[{"source": request.source, "avatar": current_avatar}]
        )
        return {"status": f"Document {request.id} updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")

# Chroma DB 
@app.delete("/delete_document")
async def delete_document(request: DeleteDocumentRequest):
    try:
        collection = get_collection(current_avatar)
        collection.delete(ids=[request.id])
        return {"status": f"Document {request.id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# QLoRA Adapter
@app.post("/add_training_document")
async def add_training_document(request: TrainingDocumentRequest):
    try:
        avatar_dir = os.path.join(training_data_dir, current_avatar)
        os.makedirs(avatar_dir, exist_ok=True)
        dataset_file = os.path.join(avatar_dir, "dataset.jsonl")
        
        doc_id = str(uuid.uuid4())
        document = {
            "id": doc_id,
            "instruction": request.instruction,
            "response": request.response,
            "source": request.source
        }
        
        with open(dataset_file, "a") as f:
            f.write(json.dumps(document) + "\n")
        
        return {"status": "Training document added", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding training document: {str(e)}")

# QLoRA Adapter
@app.get("/list_training_documents")
async def list_training_documents():
    try:
        avatar_dir = os.path.join(training_data_dir, current_avatar)
        dataset_file = os.path.join(avatar_dir, "dataset.jsonl")
        
        if not os.path.exists(dataset_file):
            return {"documents": []}
        
        documents = []
        with open(dataset_file, "r") as f:
            for line in f:
                documents.append(json.loads(line.strip()))
        
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing training documents: {str(e)}")

# QLoRA Adapter
@app.put("/update_training_document")
async def update_training_document(request: UpdateTrainingDocumentRequest):
    try:
        avatar_dir = os.path.join(training_data_dir, current_avatar)
        dataset_file = os.path.join(avatar_dir, "dataset.jsonl")
        
        if not os.path.exists(dataset_file):
            raise HTTPException(status_code=404, detail="No training dataset found")
        
        # Read existing documents
        documents = []
        with open(dataset_file, "r") as f:
            documents = [json.loads(line.strip()) for line in f]
        
        # Update the document
        updated = False
        for doc in documents:
            if doc["id"] == request.id:
                doc.update({
                    "instruction": request.instruction,
                    "response": request.response,
                    "source": request.source
                })
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail=f"Document {request.id} not found")
        
        # Write back
        with open(dataset_file, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")
        
        return {"status": f"Training document {request.id} updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating training document: {str(e)}")

# QLoRA Adapter
@app.delete("/delete_training_document")
async def delete_training_document(request: DeleteDocumentRequest):
    try:
        avatar_dir = os.path.join(training_data_dir, current_avatar)
        dataset_file = os.path.join(avatar_dir, "dataset.jsonl")
        
        if not os.path.exists(dataset_file):
            raise HTTPException(status_code=404, detail="No training dataset found")
        
        # Read existing documents
        documents = []
        with open(dataset_file, "r") as f:
            documents = [json.loads(line.strip()) for line in f]
        
        # Filter out the document
        documents = [doc for doc in documents if doc["id"] != request.id]
        
        # Write back
        with open(dataset_file, "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")
        
        return {"status": f"Training document {request.id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting training document: {str(e)}")

@app.post("/query")
async def query_model(request: QueryRequest):
    global current_adapter, current_avatar
    if current_avatar != "base" and current_avatar != current_adapter:
        load_adapter(current_avatar)

    try:
        collection = get_collection(current_avatar)
        results = collection.query(query_texts=[request.query], n_results=request.max_results)
        context = ""
        for doc, metadata in zip(results["documents"] or [], results["metadatas"] or []):
            context += f"Document: {metadata['source']}\nContent: {doc}\n\n"

        prompt = f"""<|start_header_id|>system<|end_header_id>
You are a helpful assistant{' in the style of ' + current_avatar if current_avatar != 'base' else ''}. Use the following context to answer the user's query accurately and concisely. If no context is provided, rely on your general knowledge.

Context:
{context}
<|end_header_id>
<|start_header_id|>user<|end_header_id>
{request.query}
<|end_header_id>
<|start_header_id|>assistant<|end_header_id>"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.9
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("<|start_header_id|>assistant<|end_header_id>")[1].strip()
        return {"response": assistant_response, "context": context, "avatar": current_avatar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# QLoRA Adapter
@app.post("/train_adapter")
async def train_adapter(request: TrainRequest):
    global model, current_avatar
    if not model:
        raise HTTPException(status_code=500, detail="Base model not loaded.")

    adapter_path = os.path.join(adapters_dir, current_avatar)
    dataset_file = os.path.join(training_data_dir, current_avatar, "dataset.jsonl")
    
    try:
        global current_adapter
        current_adapter = None
        load_base_model()

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        if not os.path.exists(dataset_file):
            raise HTTPException(status_code=404, detail=f"No training dataset found for {current_avatar}")

        dataset = load_dataset("json", data_files=dataset_file, split="train")

        training_args = TrainingArguments(
            output_dir=adapter_path,
            per_device_train_batch_size=request.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            num_train_epochs=request.epochs,
            save_strategy="epoch",
            logging_steps=10
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=2048
        )

        trainer.train()
        model.save_pretrained(adapter_path)
        return {"status": f"Training completed for {current_avatar}. Adapter saved to {adapter_path}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training adapter for {current_avatar}: {str(e)}")

# QLoRA Adapter
@app.post("/attach_adapter")
async def attach_adapter():
    try:
        load_adapter(current_avatar)
        return {"status": f"Adapter for {current_avatar} attached successfully." if current_adapter == current_avatar else f"No adapter found for {current_avatar}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error attaching adapter for {current_avatar}: {str(e)}")

