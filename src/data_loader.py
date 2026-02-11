import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    def __init__(self):
        self.data_paths = config.data_paths
        self._validate_paths()
    
    def _validate_paths(self):
        missing_files = []
        for name, path in self.data_paths.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            logger.warning(f"Missing data files:\n" + "\n".join(missing_files))
    
    def load_patient_profiles(self) -> pd.DataFrame:
        path = self.data_paths['patient_profiles']
        logger.info(f"Loading patient profiles from {path}")
        
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = [
            'patient_id', 'age', 'gender', 'baseline_phq9', 'baseline_gad7',
            'baseline_severity', 'treatment_type', 'treatment_duration_weeks',
            'session_attendance_rate', 'outcome_phq9', 'outcome_gad7',
            'treatment_response'
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def load_therapy_notes(self) -> pd.DataFrame:
        path = self.data_paths['therapy_notes']
        
        if not path.exists():
            logger.warning(f"Therapy notes not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        return df
    
    def load_digital_chats(self) -> pd.DataFrame:
        path = self.data_paths['digital_chats']
        
        if not path.exists():
            logger.warning(f"Digital chats not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        return df
    
    def load_reddit_posts(self) -> pd.DataFrame:
        path = self.data_paths['reddit_posts']
        
        if not path.exists():
            logger.warning(f"Reddit posts not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        data = {
            'patient_profiles': self.load_patient_profiles(),
            'therapy_notes': self.load_therapy_notes(),
            'digital_chats': self.load_digital_chats(),
            'reddit_posts': self.load_reddit_posts()
        }
        return data
    
    def get_patient_data(self, patient_id: str) -> Dict[str, pd.DataFrame]:
        data = self.load_all_data()
        
        patient_data = {
            'profile': data['patient_profiles'][
                data['patient_profiles']['patient_id'] == patient_id
            ],
            'therapy_notes': data['therapy_notes'][
                data['therapy_notes']['patient_id'] == patient_id
            ] if not data['therapy_notes'].empty else pd.DataFrame(),
            'digital_chats': data['digital_chats'][
                data['digital_chats']['patient_id'] == patient_id
            ] if not data['digital_chats'].empty else pd.DataFrame(),
            'reddit_posts': data['reddit_posts'][
                data['reddit_posts']['patient_id'] == patient_id
            ] if not data['reddit_posts'].empty else pd.DataFrame()
        }
        
        return patient_data
