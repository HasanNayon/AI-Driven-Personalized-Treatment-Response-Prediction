"""
Data Loader
Handles loading and basic validation of datasets
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and validate mental health datasets"""
    
    def __init__(self):
        """Initialize data loader with configured paths"""
        self.data_paths = config.data_paths
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that all required data files exist"""
        missing_files = []
        for name, path in self.data_paths.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            logger.warning(f"Missing data files:\n" + "\n".join(missing_files))
    
    def load_patient_profiles(self) -> pd.DataFrame:
        """
        Load patient profiles dataset
        
        Returns:
            DataFrame with patient demographics and outcomes
        """
        path = self.data_paths['patient_profiles']
        logger.info(f"Loading patient profiles from {path}")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} patient records")
        
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
        """
        Load therapy session notes
        
        Returns:
            DataFrame with therapy notes and metadata
        """
        path = self.data_paths['therapy_notes']
        
        if not path.exists():
            logger.warning(f"Therapy notes not found: {path}")
            return pd.DataFrame()
        
        logger.info(f"Loading therapy notes from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} therapy note records")
        
        return df
    
    def load_digital_chats(self) -> pd.DataFrame:
        """
        Load digital therapy chat transcripts
        
        Returns:
            DataFrame with chat messages and sentiment
        """
        path = self.data_paths['digital_chats']
        
        if not path.exists():
            logger.warning(f"Digital chats not found: {path}")
            return pd.DataFrame()
        
        logger.info(f"Loading digital chats from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} chat records")
        
        return df
    
    def load_reddit_posts(self) -> pd.DataFrame:
        """
        Load patient Reddit posts (self-reported data)
        
        Returns:
            DataFrame with Reddit posts and features
        """
        path = self.data_paths['reddit_posts']
        
        if not path.exists():
            logger.warning(f"Reddit posts not found: {path}")
            return pd.DataFrame()
        
        logger.info(f"Loading Reddit posts from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} Reddit post records")
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets
        
        Returns:
            Dictionary with all loaded DataFrames
        """
        logger.info("Loading all datasets...")
        
        data = {
            'patient_profiles': self.load_patient_profiles(),
            'therapy_notes': self.load_therapy_notes(),
            'digital_chats': self.load_digital_chats(),
            'reddit_posts': self.load_reddit_posts()
        }
        
        logger.info("All datasets loaded successfully")
        return data
    
    def get_patient_data(self, patient_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific patient
        
        Args:
            patient_id: Patient identifier (e.g., 'P0001')
            
        Returns:
            Dictionary with patient-specific data from all sources
        """
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
