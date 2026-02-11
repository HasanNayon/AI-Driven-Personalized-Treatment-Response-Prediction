import torch
import numpy as np
from typing import List, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class LLMFeatureExtractor:
    def __init__(self, model_path: Union[str, Path] = None):
        self.model_path = model_path or config.model_paths['llm']
        self.device = torch.device(
            config.get('llm.device') if torch.cuda.is_available() else 'cpu'
        )
        self.max_length = config.get('llm.max_length', 512)
        self.batch_size = config.get('llm.batch_size', 8)
        self.embedding_dim = 2051  # matches XGBoost training dimensions
        
        logger.info(f"Initializing LLM on device: {self.device}")
        self.tokenizer = None
        self.model = None
        self.use_embeddings = False
        
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"Loading fine-tuned model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                timeout=5
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.use_embeddings = True
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load LLM model: {type(e).__name__}: {str(e)[:100]}")
            logger.info(f"Using {self.embedding_dim}-dimensional zero embeddings for compatibility")
            self.use_embeddings = False
            # Model and tokenizer will be None - that's OK, we'll just return zeros
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        # If model not available, return zero embeddings with correct dimension
        if not self.use_embeddings:
            logger.debug(f"Using zero embeddings for {len(texts)} texts")
            return np.zeros((len(texts), self.embedding_dim))
        
        if not texts or all(not t.strip() for t in texts):
            logger.warning(f"Empty texts provided, returning zero embeddings with shape ({len(texts)}, {self.embedding_dim})")
            return np.zeros((len(texts), self.embedding_dim))
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_size = len(batch_texts)
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_state = outputs.last_hidden_state
                    
                    mean_pool = hidden_state.mean(dim=1)
                    max_pool = hidden_state.max(dim=1)[0]
                    cls_token = hidden_state[:, 0, :]
                    
                    # concat pooled representations and trim to match training dims
                    combined = torch.cat([mean_pool, max_pool, cls_token], dim=1)
                    combined = combined[:, :self.embedding_dim]
                    
                    embeddings = combined.cpu().numpy()
                    all_embeddings.append(embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Return zero embeddings for failed batch
                all_embeddings.append(np.zeros((batch_size, self.embedding_dim)))
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def extract_features_from_dataframe(self, df, text_column: str) -> np.ndarray:
        texts = df[text_column].fillna('').astype(str).tolist()
        return self.get_embeddings(texts)
    
    def extract_multimodal_features(self, text_dict: dict) -> np.ndarray:
        embeddings_list = []
        
        for source, text in text_dict.items():
            if text and text.strip():
                emb = self.get_embeddings(text)
                if emb.shape[0] > 0:
                    embeddings_list.append(emb[0])
        
        if not embeddings_list:
            # Return zero embedding if no text available
            return np.zeros((1, self.embedding_dim))
        
        # Average embeddings from different sources
        combined = np.mean(embeddings_list, axis=0, keepdims=True)
        return combined
    
    def __del__(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
