import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables if present
        config['api']['host'] = os.getenv('API_HOST', config['api']['host'])
        config['api']['port'] = int(os.getenv('API_PORT', config['api']['port']))
        config['api']['debug'] = os.getenv('FLASK_DEBUG', str(config['api']['debug'])).lower() == 'true'
        config['llm']['device'] = os.getenv('MODEL_DEVICE', config['llm']['device'])
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        path_str = self.get(key)
        if path_str is None:
            raise ValueError(f"Path configuration not found: {key}")
        
        return self.base_dir / path_str
    
    @property
    def model_paths(self) -> Dict[str, Path]:
        return {
            'llm': self.get_path('models.llm_path'),
            'xgboost': self.get_path('models.xgboost_path'),
            'checkpoint': self.get_path('models.checkpoint_path')
        }
    
    @property
    def data_paths(self) -> Dict[str, Path]:
        return {
            'patient_profiles': self.get_path('data.patient_profiles'),
            'therapy_notes': self.get_path('data.therapy_notes'),
            'digital_chats': self.get_path('data.digital_chats'),
            'reddit_posts': self.get_path('data.reddit_posts')
        }

# Global configuration instance
config = Config()
