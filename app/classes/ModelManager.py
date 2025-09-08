# ModelManager.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from core.config import settings
from core.logging import logger
from core.monitoring import metrics
from core.s3_instance import get_s3_client, download_s3_to_dir
from core.redis_instance import get_redis_client
import aiohttp
import json
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
from huggingface_hub import login

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_adapter = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path("/tmp/adapters")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_limit = 5  # Max number of adapters to cache
        self.cache_keys = []  # Track order of cached adapters
        
        # Pre-built model path (baked into Docker image)
        self.model_path = Path("/app/models/llama-3.2-1b-instruct")
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"

    def _is_model_available(self) -> bool:
        """Check if the pre-built model is available in the expected location."""
        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        
        if not self.model_path.exists():
            return False
            
        for file_name in required_files:
            if not (self.model_path / file_name).exists():
                return False
                
        return True

    async def load_base_model(self):
        """Load the pre-built base model and tokenizer from the Docker image."""
        if self.model is None:
            if not self._is_model_available():
                error_msg = f"Pre-built model not found at {self.model_path}. Please rebuild the Docker image."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"Loading pre-built model from: {self.model_path}")
            
            try:
                # Load tokenizer from pre-built location
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True  # Only use local files
                )
                logger.info("Tokenizer loaded from pre-built model.")
                
                # Load model from pre-built location
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto",
                    local_files_only=True  # Only use local files
                )
                logger.info("Model loaded from pre-built model.")
                
                logger.info("✅ Base model loaded successfully from Docker image.")
                metrics.model_loads.labels(model="base").inc()
                
            except Exception as e:
                logger.error(f"Failed to load pre-built model: {e}")
                raise e

    async def preload_model_on_startup(self):
        """Preload the model during application startup."""
        try:
            logger.info("Preloading base model on startup...")
            await self.load_base_model()
            logger.info("✅ Base model preloaded successfully and ready for inference.")
        except Exception as e:
            logger.error(f"❌ Failed to preload base model: {e}")
            raise e

    async def _download_model_if_missing(self):
        """Download model if missing (for development mode)."""
        try:
            logger.info(f"Downloading {self.model_name} to {self.model_path}...")
            
            # Create directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Get HF token from environment
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            if hf_token:
                login(token=hf_token)
                logger.info("Logged into Hugging Face Hub for model download.")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token if hf_token else None
            )
            tokenizer.save_pretrained(str(self.model_path))
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                token=hf_token if hf_token else None,
                device_map=None
            )
            model.save_pretrained(str(self.model_path))
            
            logger.info(f"✅ Model downloaded successfully to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise e

    async def load_adapter(self, user_id: str, avatar_id: str) -> bool:
        """Load adapter from cache or S3, return True if loaded, False if not found."""
        adapter_path = self.cache_dir / avatar_id
        s3_prefix = f"users/{user_id}/avatars/{avatar_id}/adapters/"

        # Check cache
        if adapter_path.exists():
            logger.info(f"Loading adapter from cache: {avatar_id}")
            self._attach_adapter(adapter_path)
            self._update_cache_order(avatar_id)
            metrics.adapter_cache_hits.inc()
            return True

        # Check S3
        async with await get_s3_client() as s3_client:
            try:
                await download_s3_to_dir(s3_prefix, adapter_path)
                logger.info(f"Downloaded adapter from S3: {s3_prefix}")
                self._attach_adapter(adapter_path)
                await self._cache_adapter(avatar_id, adapter_path)
                metrics.adapter_s3_downloads.inc()
                return True
            except Exception as e:
                logger.warning(f"Adapter not found in S3 for {avatar_id}: {e}")
                self.current_adapter = None
                metrics.adapter_s3_errors.inc()
                return False

    def _attach_adapter(self, adapter_path: Path):
        """Attach adapter to the model."""
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.current_adapter = str(adapter_path)
        logger.info(f"Attached adapter: {adapter_path}")
        metrics.adapter_loads.inc()

    async def _cache_adapter(self, avatar_id: str, adapter_path: Path):
        """Cache adapter locally and manage cache limit."""
        self._update_cache_order(avatar_id)
        if len(self.cache_keys) > self.cache_limit:
            old_adapter = self.cache_keys.pop(0)
            old_path = self.cache_dir / old_adapter
            if old_path.exists():
                import shutil
                shutil.rmtree(old_path)
                logger.info(f"Evicted adapter from cache: {old_adapter}")
                metrics.adapter_cache_evictions.inc()

    def _update_cache_order(self, avatar_id: str):
        """Update LRU cache order."""
        if avatar_id in self.cache_keys:
            self.cache_keys.remove(avatar_id)
        self.cache_keys.append(avatar_id)

    async def query_vectorstore(self, user_input: str, user_id: str, avatar_id: str, top_k: int = 10) -> str:
        """Query ChromaDB vectorstore for context."""
        async with aiohttp.ClientSession() as session:
            query_vec = self._embed_query(user_input)
            url = f"{settings.CHROMA_DB_URL}/query"  # Placeholder for ChromaDB endpoint
            payload = {
                "query_embeddings": [query_vec],
                "n_results": top_k,
                "collection": f"users/{user_id}/avatars/{avatar_id}/vectorstore_data"
            }
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.warning(f"Vectorstore query failed: {response.status}")
                        return "No relevant context found."
                    results = await response.json()
                    docs = results.get("documents", [[]])[0]
                    context = "\n".join(docs)[:1000] if docs else "No relevant context found."
                    logger.info(f"Retrieved context from vectorstore for {avatar_id}")
                    metrics.vectorstore_queries.inc()
                    return context
            except Exception as e:
                logger.error(f"Vectorstore query error: {e}")
                metrics.vectorstore_errors.inc()
                return "No relevant context found."

    def _embed_query(self, query: str) -> list:
        """Placeholder for query embedding (to be implemented with actual embedder)."""
        # Replace with actual embedding logic (e.g., using sentence-transformers)
        return [0.0] * 768  # Mock embedding vector

    async def generate_response(self, user_input: str, user_id: str, avatar_id: Optional[str] = None, use_context: bool = False, max_new_tokens: int = 50) -> Dict[str, Any]:
        """Generate response with optional adapter and context."""
        # Ensure base model is loaded (should already be loaded from startup)
        if self.model is None:
            await self.load_base_model()

        adapter_used = False
        if avatar_id:
            adapter_used = await self.load_adapter(user_id, avatar_id)

        context = ""
        if use_context:
            context = await self.query_vectorstore(user_input, user_id, avatar_id or "default")

        prompt = f"""You are an assistant. Use the context to answer the question briefly.

Context:
{context}

Q: {user_input}
A:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("A:")[-1].strip()
        if not adapter_used and avatar_id:
            response += " (Adapter not used: not found in S3 or cache)"
        
        return {
            "response": response,
            "adapter_used": adapter_used,
            "context_used": use_context and context != "No relevant context found."
        }