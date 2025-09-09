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
import gc

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_adapter = None
        self.device = self._get_device()
        self.cache_dir = Path("/tmp/adapters")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_limit = 5  # Max number of adapters to cache
        self.cache_keys = []  # Track order of cached adapters
        
        # Pre-built model path (baked into Docker image)
        self.model_path = Path("/app/models/llama-3.2-1b-instruct")
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"

    def _get_device(self) -> str:
        """Determine the appropriate device based on environment variables and hardware."""
        # Check if GPU usage is forced off
        force_cpu = os.getenv("FORCE_CPU", "0") == "1"
        use_gpu = os.getenv("USE_GPU", "0") == "1"
        
        logger.info(f"Device selection - FORCE_CPU: {force_cpu}, USE_GPU: {use_gpu}")
        
        if force_cpu:
            logger.info("CPU mode forced via FORCE_CPU=1")
            return "cpu"
        
        if not use_gpu:
            logger.info("CPU mode selected (USE_GPU not set to 1)")
            return "cpu"
        
        # Check if CUDA is available and GPUs are visible
        if torch.cuda.is_available():
            cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
            logger.info(f"CUDA available, CUDA_VISIBLE_DEVICES: '{cuda_devices}'")
            
            if cuda_devices and cuda_devices.strip() != "":
                device = "cuda:0"
                # Log GPU information
                if torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"Selected GPU device: {device} ({gpu_name}, {gpu_memory:.1f}GB)")
                return device
            else:
                logger.warning("CUDA available but no devices visible (CUDA_VISIBLE_DEVICES is empty)")
        else:
            logger.warning("CUDA not available on this system")
        
        # Fallback to CPU
        logger.info("Falling back to CPU")
        return "cpu"

    def _get_model_kwargs(self) -> Dict[str, Any]:
            """Get model loading kwargs based on device and available memory."""
            kwargs = {
                "torch_dtype": torch.float16 if self.device.startswith("cuda") else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            
            if self.device.startswith("cuda"):
                # GPU-specific optimizations
                kwargs.update({
                    "device_map": "auto",
                })
                
                # Check available GPU memory and adjust accordingly
                if torch.cuda.is_available():
                    try:
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        available_memory = total_memory - torch.cuda.memory_allocated(0)
                        memory_gb = available_memory / 1e9
                        
                        logger.info(f"GPU memory available: {memory_gb:.1f}GB")
                        
                        # If GPU has less than 6GB available, use 8-bit loading
                        if memory_gb < 6:
                            logger.info("GPU has <6GB available memory, enabling 8-bit loading")
                            kwargs["load_in_8bit"] = True
                        # If GPU has less than 4GB available, use 4-bit loading
                        elif memory_gb < 4:
                            logger.info("GPU has <4GB available memory, enabling 4-bit loading")
                            kwargs["load_in_4bit"] = True
                            
                    except Exception as e:
                        logger.warning(f"Could not check GPU memory: {e}")
            else:
                # CPU-specific settings
                kwargs.update({
                    "torch_dtype": torch.float32,  # CPU works better with float32
                    "device_map": None,
                })
            
            return kwargs

    def _cleanup_model(self):
        """Clean up existing model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("Model cleanup completed")

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
            logger.info(f"Target device: {self.device}")
            
            try:
                # Load tokenizer from pre-built location
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True  # Only use local files
                )
                
                # Add padding token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("Tokenizer loaded from pre-built model.")
                
                # Load model from pre-built location with device-appropriate settings
                logger.info("Loading model...")
                model_kwargs = self._get_model_kwargs()
                logger.info(f"Model loading kwargs: {model_kwargs}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,  # Only use local files
                    **model_kwargs
                )
                
                # If using CPU or device_map is None, manually move to device
                if not self.device.startswith("cuda") or model_kwargs.get("device_map") is None:
                    logger.info(f"Moving model to device: {self.device}")
                    self.model = self.model.to(self.device)
                
                logger.info("Model loaded from pre-built model.")
                
                # Log memory usage
                if torch.cuda.is_available() and self.device.startswith("cuda"):
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    cached = torch.cuda.memory_reserved(0) / 1e9
                    logger.info(f"GPU memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
                
                logger.info("✅ Base model loaded successfully from Docker image.")
                metrics.model_loads.inc()
                
            except Exception as e:
                logger.error(f"Failed to load pre-built model: {e}")
                self._cleanup_model()  # Clean up on failure
                raise e

    async def preload_model_on_startup(self):
        """Preload the model during application startup."""
        try:
            logger.info("Preloading base model on startup...")
            logger.info(f"Device configuration: {self.device}")
            
            # Log environment variables for debugging
            logger.info(f"Environment - USE_GPU: {os.getenv('USE_GPU')}, FORCE_CPU: {os.getenv('FORCE_CPU')}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            
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
            
            # Download model with appropriate dtype
            model_kwargs = self._get_model_kwargs()
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token if hf_token else None,
                torch_dtype=model_kwargs["torch_dtype"],
                device_map=None  # Don't map to device during download
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
            try:
                adapter_used = await self.load_adapter(user_id, avatar_id)
                if adapter_used:
                    logger.info(f"Using adapter for avatar {avatar_id}")
                else:
                    logger.info(f"No adapter found for avatar {avatar_id}, using base model")
            except Exception as e:
                logger.warning(f"Failed to load adapter for {avatar_id}: {e}")
                adapter_used = False

        # If no adapter was loaded or avatar_id is None, ensure we're using base model
        if not adapter_used:
            await self._ensure_base_model()

        context = ""
        if use_context:
            context = await self.query_vectorstore(user_input, user_id, avatar_id or "default")

        # Build prompt based on whether we have context
        if context and context != "No relevant context found.":
            prompt = f"""You are an assistant. Use the context to answer the question briefly.

    Context:
    {context}

    Q: {user_input}
    A:"""
        else:
            prompt = f"""You are a helpful assistant. Answer the question briefly.

    Q: {user_input}
    A:"""

        # Prepare inputs and ensure they're on the correct device
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.7,  # Add some temperature for better responses
                    repetition_penalty=1.1  # Prevent repetitive responses
                )

            # Extract only the generated part (after the prompt)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # If response is empty or just whitespace, provide a fallback
            if not response:
                response = "I'm sorry, I couldn't generate a proper response. Could you please rephrase your question?"

            # Log GPU memory usage after inference
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated(0) / 1e9
                logger.debug(f"Post-inference GPU memory allocated: {allocated:.2f}GB")

            # Log the successful inference
            model_type = "adapter" if adapter_used else "base model"
            logger.info(f"Generated response using {model_type} for user {user_id}")

            return {
                "response": response,
                "adapter_used": adapter_used,
                "context_used": use_context and context != "No relevant context found.",
                "device": self.device,
                "model_type": model_type
            }

        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            # Return a proper error response instead of raising
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "adapter_used": adapter_used,
                "context_used": False,
                "device": self.device,
                "error": str(e)
            }

    async def _ensure_base_model(self):
        """Ensure we're using the base model (not wrapped with an adapter)."""
        if self.current_adapter is not None:
            logger.info("Switching back to base model")
            # If we have an adapter currently loaded, we need to reload the base model
            # This is because PEFT wraps the original model
            if hasattr(self, 'base_model') and self.base_model is not None:
                self.model = self.base_model
            else:
                # Fallback: reload the base model
                await self.load_base_model()
            self.current_adapter = None

    def _attach_adapter(self, adapter_path: Path):
        """Attach adapter to the model."""
        try:
            # Save reference to base model if we don't have one
            if not hasattr(self, 'base_model') or self.base_model is None:
                self.base_model = self.model

            # Create PEFT model from base model
            self.model = PeftModel.from_pretrained(self.base_model, str(adapter_path))

            # Ensure adapter is on the same device as the base model
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)

            self.current_adapter = str(adapter_path)
            logger.info(f"Attached adapter: {adapter_path} on device: {self.device}")
            metrics.adapter_loads.inc()

        except Exception as e:
            logger.error(f"Failed to attach adapter {adapter_path}: {e}")
            # Fall back to base model on adapter loading failure
            if hasattr(self, 'base_model') and self.base_model is not None:
                self.model = self.base_model
                self.current_adapter = None
            raise e

        