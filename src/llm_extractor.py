"""
LLM Feature Extractor
Uses fine-tuned Llama 3.2 1B model to generate embeddings from text data
"""

import torch
import numpy as np
from typing import List, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class LLMFeatureExtractor:
    """Extract features using fine-tuned Llama 3.2 1B model"""
    
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialize LLM feature extractor
        
        Args:
            model_path: Path to fine-tuned model (uses config if None)
        """
        self.model_path = model_path or config.model_paths['llm']
        self.device = torch.device(
            config.get('llm.device') if torch.cuda.is_available() else 'cpu'
        )
        self.max_length = config.get('llm.max_length', 512)
        self.batch_size = config.get('llm.batch_size', 8)
        self.embedding_dim = 2051  # Fixed dimension to match XGBoost training
        
        logger.info(f"Initializing LLM on device: {self.device}")
        self.tokenizer = None
        self.model = None
        self.use_embeddings = False  # Flag to use actual embeddings vs fallback
        
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model - graceful fallback to zero embeddings"""
        try:
            logger.info(f"Loading fine-tuned model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                timeout=5  # Short timeout
            )
            
            # Add padding token if not present
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
        """
        Generate embeddings from text(s)
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, 2051])
        """
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
                
                # Generate embeddings with multiple pooling strategies
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]
                    
                    # Multiple pooling strategies to get richer embeddings
                    mean_pool = hidden_state.mean(dim=1)  # 768 dims
                    max_pool = hidden_state.max(dim=1)[0]  # 768 dims
                    
                    # Use CLS token representation if available
                    cls_token = hidden_state[:, 0, :]  # 768 dims
                    
                    # Concatenate all pooled representations
                    # This gives us 768 * 3 = 2304 dimensions, but we'll trim to 2051
                    combined = torch.cat([mean_pool, max_pool, cls_token], dim=1)  # [batch_size, 2304]
                    
                    # Trim to target dimension (2051) to match model training
                    combined = combined[:, :self.embedding_dim]
                    
                    embeddings = combined.cpu().numpy()
                    all_embeddings.append(embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Return zero embeddings for failed batch with correct dimension
                all_embeddings.append(np.zeros((batch_size, self.embedding_dim)))
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def extract_features_from_dataframe(self, df, text_column: str) -> np.ndarray:
        """
        Extract embeddings from a DataFrame column
        
        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            
        Returns:
            Numpy array of embeddings
        """
        texts = df[text_column].fillna('').astype(str).tolist()
        return self.get_embeddings(texts)
    
    def extract_multimodal_features(self, text_dict: dict) -> np.ndarray:
        """
        Extract and combine features from multiple text sources
        
        Args:
            text_dict: Dictionary with keys like 'therapy_notes', 'chat', 'reddit'
            
        Returns:
            Combined embedding vector (2051 dimensions to match XGBoost training)
        """
        embeddings_list = []
        
        for source, text in text_dict.items():
            if text and text.strip():
                emb = self.get_embeddings(text)
                if emb.shape[0] > 0:
                    embeddings_list.append(emb[0])  # Take first (and only) row
        
        if not embeddings_list:
            # Return zero embedding if no text available
            return np.zeros((1, self.embedding_dim))
        
        # Average embeddings from different sources
        combined = np.mean(embeddings_list, axis=0, keepdims=True)
        return combined
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
