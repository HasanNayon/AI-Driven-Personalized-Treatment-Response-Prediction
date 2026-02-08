"""
Data Preprocessing
Feature engineering and data preparation for model training/inference
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess and engineer features for mental health prediction"""
    
    def __init__(self):
        """Initialize preprocessor with encoders and scalers"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare clinical/demographic features
        
        Args:
            df: Patient profiles DataFrame
            
        Returns:
            DataFrame with engineered clinical features
        """
        logger.info("Preparing clinical features...")
        
        df = df.copy()
        
        # Age group categorization
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 14, 17, 20, 25], 
            labels=['early_teen', 'mid_teen', 'late_teen', 'young_adult']
        )
        
        # Total baseline severity score
        df['baseline_total'] = df['baseline_phq9'] + df['baseline_gad7']
        
        # Outcome improvement
        df['outcome_total'] = df['outcome_phq9'] + df['outcome_gad7']
        df['symptom_reduction'] = df['baseline_total'] - df['outcome_total']
        df['improvement_rate'] = df['symptom_reduction'] / (df['baseline_total'] + 1e-6)
        
        # Treatment intensity features
        df['treatment_intensity'] = df['treatment_duration_weeks'] * df['session_attendance_rate']
        
        # Digital engagement (only for digital therapy)
        df['digital_engagement_score'] = df['digital_engagement_score'].fillna(0)
        df['has_digital'] = (df['digital_engagement_score'] > 0).astype(int)
        
        # Severity categories to numeric
        severity_map = {
            'mild-moderate': 1,
            'moderate': 2,
            'moderate-severe': 3,
            'severe': 4
        }
        df['severity_numeric'] = df['baseline_severity'].map(severity_map)
        
        logger.info(f"Clinical features prepared: {df.shape[1]} columns")
        return df
    
    def aggregate_text_features(
        self, 
        patient_df: pd.DataFrame,
        therapy_notes: pd.DataFrame,
        digital_chats: pd.DataFrame,
        reddit_posts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate text data for each patient
        
        Args:
            patient_df: Patient profiles
            therapy_notes: Therapy session notes
            digital_chats: Digital therapy chats
            reddit_posts: Reddit posts
            
        Returns:
            DataFrame with aggregated text features
        """
        logger.info("Aggregating text features...")
        
        text_features = []
        
        for _, patient in patient_df.iterrows():
            patient_id = patient['patient_id']
            
            # Therapy notes aggregation
            patient_notes = therapy_notes[therapy_notes['patient_id'] == patient_id]
            notes_text = ' '.join(patient_notes['therapist_notes'].astype(str)) if not patient_notes.empty else ""
            notes_count = len(patient_notes)
            
            # Digital chats aggregation (patient messages only)
            patient_chats = digital_chats[
                (digital_chats['patient_id'] == patient_id) & 
                (digital_chats['message_type'] == 'patient')
            ] if not digital_chats.empty else pd.DataFrame()
            
            chat_text = ' '.join(patient_chats['message_text'].astype(str)) if not patient_chats.empty else ""
            chat_count = len(patient_chats)
            avg_sentiment = patient_chats['sentiment_score'].mean() if not patient_chats.empty else 0
            
            # Reddit posts aggregation
            patient_posts = reddit_posts[reddit_posts['patient_id'] == patient_id] if not reddit_posts.empty else pd.DataFrame()
            reddit_text = ' '.join(patient_posts['post_text'].astype(str)) if not patient_posts.empty else ""
            reddit_count = len(patient_posts)
            reddit_sentiment = patient_posts['post_sentiment'].mean() if not patient_posts.empty else 0
            
            # Combined text for LLM embedding
            combined_text = f"{notes_text} {chat_text} {reddit_text}".strip()
            
            text_features.append({
                'patient_id': patient_id,
                'therapy_notes_text': notes_text,
                'digital_chat_text': chat_text,
                'reddit_text': reddit_text,
                'combined_text': combined_text,
                'notes_count': notes_count,
                'chat_count': chat_count,
                'reddit_count': reddit_count,
                'avg_chat_sentiment': avg_sentiment,
                'avg_reddit_sentiment': reddit_sentiment,
                'total_text_length': len(combined_text)
            })
        
        text_df = pd.DataFrame(text_features)
        logger.info(f"Text features aggregated for {len(text_df)} patients")
        
        return text_df
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: DataFrame with categorical columns
            categorical_cols: List of columns to encode
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if fit:
                # Fit new encoder
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Fitted encoder for {col}: {le.classes_}")
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    logger.warning(f"No encoder found for {col}, skipping")
        
        return df
    
    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: DataFrame with numerical columns
            numerical_cols: List of columns to scale
            fit: Whether to fit scaler
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        # Filter to existing columns
        cols_to_scale = [col for col in numerical_cols if col in df.columns]
        
        if not cols_to_scale:
            return df
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            logger.debug(f"Fitted scaler for {len(cols_to_scale)} numerical features")
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def prepare_training_data(
        self,
        patient_df: pd.DataFrame,
        therapy_notes: pd.DataFrame = None,
        digital_chats: pd.DataFrame = None,
        reddit_posts: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline for training
        
        Args:
            patient_df: Patient profiles
            therapy_notes: Therapy session notes (optional)
            digital_chats: Digital therapy chats (optional)
            reddit_posts: Reddit posts (optional)
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Starting training data preparation...")
        
        # Prepare clinical features
        df = self.prepare_clinical_features(patient_df)
        
        # Aggregate text features if available
        if therapy_notes is not None or digital_chats is not None or reddit_posts is not None:
            therapy_notes = therapy_notes if therapy_notes is not None else pd.DataFrame()
            digital_chats = digital_chats if digital_chats is not None else pd.DataFrame()
            reddit_posts = reddit_posts if reddit_posts is not None else pd.DataFrame()
            
            text_features = self.aggregate_text_features(
                df, therapy_notes, digital_chats, reddit_posts
            )
            df = df.merge(text_features, on='patient_id', how='left')
        
        # Encode categorical features
        categorical_cols = ['gender', 'baseline_severity', 'treatment_type', 'age_group']
        df = self.encode_categorical_features(df, categorical_cols, fit=True)
        
        # Prepare target variable
        target = df['treatment_response'].copy()
        
        # Scale numerical features
        numerical_cols = [
            'age', 'baseline_phq9', 'baseline_gad7', 'baseline_total',
            'treatment_duration_weeks', 'session_attendance_rate',
            'digital_engagement_score', 'treatment_intensity', 'severity_numeric'
        ]
        df = self.scale_numerical_features(df, numerical_cols, fit=True)
        
        self.is_fitted = True
        logger.info(f"Training data prepared: {df.shape}")
        
        return df, target
    
    def prepare_inference_data(self, patient_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare data for inference
        
        Args:
            patient_data: Dictionary with patient data from all sources
            
        Returns:
            Preprocessed features DataFrame
        """
        logger.info("Preparing inference data...")
        
        if not self.is_fitted:
            logger.warning("Preprocessor not fitted. Encoders/scalers may not be available.")
        
        # Similar to training preparation but without fitting
        df = self.prepare_clinical_features(patient_data['profile'])
        
        # Aggregate text features
        text_features = self.aggregate_text_features(
            df,
            patient_data.get('therapy_notes', pd.DataFrame()),
            patient_data.get('digital_chats', pd.DataFrame()),
            patient_data.get('reddit_posts', pd.DataFrame())
        )
        df = df.merge(text_features, on='patient_id', how='left')
        
        # Encode and scale
        categorical_cols = ['gender', 'baseline_severity', 'treatment_type', 'age_group']
        df = self.encode_categorical_features(df, categorical_cols, fit=False)
        
        numerical_cols = [
            'age', 'baseline_phq9', 'baseline_gad7', 'baseline_total',
            'treatment_duration_weeks', 'session_attendance_rate',
            'digital_engagement_score', 'treatment_intensity', 'severity_numeric'
        ]
        df = self.scale_numerical_features(df, numerical_cols, fit=False)
        
        logger.info(f"Inference data prepared: {df.shape}")
        return df
