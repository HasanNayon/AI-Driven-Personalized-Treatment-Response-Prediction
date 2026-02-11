import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from src.config import config
from src.logger import get_logger
from src.llm_extractor import LLMFeatureExtractor
from src.preprocessing import DataPreprocessor

logger = get_logger(__name__)


class ModelManager:
    def __init__(self):
        self.xgboost_model = None
        self.llm_extractor = None
        self.preprocessor = DataPreprocessor()
        self.feature_names = None
        self.response_mapping = {
            0: 'non-responder',
            1: 'partial',
            2: 'responder'
        }
        self.load_models()
    
    def load_models(self):
        logger.info("Loading models...")
        
        # Load XGBoost model
        try:
            xgb_path = config.model_paths['xgboost']
            logger.info(f"Loading XGBoost model from {xgb_path}")
            loaded_obj = joblib.load(xgb_path)
            
            if isinstance(loaded_obj, dict):
                logger.info(f"Model dict keys: {list(loaded_obj.keys())}")
                if 'model' in loaded_obj:
                    self.xgboost_model = loaded_obj['model']
                elif 'xgboost_model' in loaded_obj:
                    self.xgboost_model = loaded_obj['xgboost_model']
                elif 'clf' in loaded_obj:
                    self.xgboost_model = loaded_obj['clf']
                else:
                    for key, value in loaded_obj.items():
                        if hasattr(value, 'predict') and hasattr(value, 'predict_proba'):
                            self.xgboost_model = value
                            logger.info(f"Found model in key: {key}")
                            break
                    if self.xgboost_model is None:
                        raise RuntimeError(f"Could not find model in dictionary. Available keys: {list(loaded_obj.keys())}")
                
                # Store feature names if available
                if 'feature_names' in loaded_obj:
                    self.feature_names = loaded_obj['feature_names']
            else:
                self.xgboost_model = loaded_obj
            
            if not hasattr(self.xgboost_model, 'predict'):
                raise RuntimeError(f"Loaded object does not have 'predict' method. Type: {type(self.xgboost_model)}")
            
            logger.info("XGBoost model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            raise
        
        try:
            logger.info("Initializing LLM feature extractor...")
            self.llm_extractor = LLMFeatureExtractor()
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            logger.warning("LLM feature extraction not available")
    
    def prepare_features(
        self,
        patient_data: pd.DataFrame,
        include_llm_features: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        
        # Extract clinical features
        clinical_feature_cols = [
            'age', 'baseline_phq9', 'baseline_gad7', 'baseline_total',
            'treatment_duration_weeks', 'session_attendance_rate',
            'digital_engagement_score', 'treatment_intensity', 'severity_numeric',
            'gender_encoded', 'treatment_type_encoded', 'age_group_encoded'
        ]
        
        # Filter to available columns
        available_cols = [col for col in clinical_feature_cols if col in patient_data.columns]
        clinical_features = patient_data[available_cols].values
        
        feature_names = available_cols.copy()
        
        if include_llm_features and self.llm_extractor is not None:
            if 'combined_text' in patient_data.columns:
                llm_features = self.llm_extractor.extract_features_from_dataframe(
                    patient_data, 'combined_text'
                )
                
                # combine clinical + LLM features
                features = np.hstack([clinical_features, llm_features])
                
                embedding_dim = llm_features.shape[1]
                llm_feature_names = [f'llm_emb_{i}' for i in range(embedding_dim)]
                feature_names.extend(llm_feature_names)
                
                logger.info(f"Features shape: {features.shape}")
            else:
                logger.warning("No text data available, using only clinical features")
                features = clinical_features
        else:
            features = clinical_features
        
        logger.info(f"Feature prep done: {features.shape}")
        return features, feature_names
    
    def predict(
        self,
        patient_data: pd.DataFrame,
        return_probabilities: bool = False
    ) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        if self.xgboost_model is None:
            raise RuntimeError("XGBoost model not loaded")
        
        logger.info(f"Making predictions for {len(patient_data)} patients")
        
        # Prepare features
        features, feature_names = self.prepare_features(patient_data)
        
        predictions = self.xgboost_model.predict(features)
        pred_labels = [self.response_mapping.get(p, 'unknown') for p in predictions]
        
        logger.info(f"Predictions: {dict(pd.Series(pred_labels).value_counts())}")
        
        if return_probabilities:
            probabilities = self.xgboost_model.predict_proba(features)
            return pred_labels, probabilities
        
        return pred_labels
    
    def predict_single_patient(
        self,
        patient_info: Dict,
        therapy_notes: str = "",
        digital_chats: str = "",
        reddit_posts: str = ""
    ) -> Dict:
        logger.info(f"Predicting for patient: {patient_info.get('patient_id', 'unknown')}")
        
        # Create DataFrame from patient info
        patient_df = pd.DataFrame([patient_info])
        
        # Add text data
        combined_text = f"{therapy_notes} {digital_chats} {reddit_posts}".strip()
        patient_df['combined_text'] = combined_text
        
        # Preprocess
        patient_df = self.preprocessor.prepare_clinical_features(patient_df)
        
        # Make prediction
        predictions, probabilities = self.predict(patient_df, return_probabilities=True)
        
        # Get feature importance for explanation
        features, feature_names = self.prepare_features(patient_df)
        feature_importance = self._get_feature_importance(features[0], feature_names)
        
        result = {
            'patient_id': patient_info.get('patient_id', 'unknown'),
            'predicted_response': predictions[0],
            'confidence': float(np.max(probabilities[0])),
            'probabilities': {
                'non-responder': float(probabilities[0][0]),
                'partial': float(probabilities[0][1]),
                'responder': float(probabilities[0][2])
            },
            'top_features': feature_importance[:10],  # Top 10 influential features
            'recommendation': self._generate_recommendation(predictions[0], probabilities[0])
        }
        
        logger.info(f"Prediction: {predictions[0]} (confidence: {result['confidence']:.2%})")
        return result
    
    def _get_feature_importance(
        self,
        feature_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 20
    ) -> List[Dict]:
        if self.xgboost_model is None:
            return []
        
        try:
            # Get feature importance from model
            model_importance = self.xgboost_model.feature_importances_
            
            # weight by feature value magnitude
            combined_importance = np.abs(feature_values) * model_importance
            
            # Get top features
            top_indices = np.argsort(combined_importance)[-top_k:][::-1]
            
            feature_importance = [
                {
                    'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                    'value': float(feature_values[i]),
                    'importance': float(combined_importance[i])
                }
                for i in top_indices
            ]
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return []
    
    def _generate_recommendation(
        self,
        prediction: str,
        probabilities: np.ndarray
    ) -> str:
        confidence = float(np.max(probabilities))
        
        recommendations = {
            'responder': (
                f"Patient likely to respond well to assigned treatment (confidence: {confidence:.1%}). "
                "Continue with current treatment plan and monitor progress regularly."
            ),
            'partial': (
                f"Patient may show partial response (confidence: {confidence:.1%}). "
                "Consider treatment augmentation or combination therapy. "
                "Close monitoring and regular assessment recommended."
            ),
            'non-responder': (
                f"Patient at risk of poor response (confidence: {confidence:.1%}). "
                "Consider alternative treatment modalities or combination approaches. "
                "Frequent monitoring and early intervention adjustments strongly recommended."
            )
        }
        
        base_rec = recommendations.get(prediction, "Prediction uncertain.")
        
        # Add uncertainty note if confidence is low
        if confidence < 0.6:
            base_rec += " Note: Prediction confidence is moderate. Clinical judgment is essential."
        
        return base_rec
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        logger.info("Evaluating model performance...")
        
        # Handle case where model might still be a dict
        model = self.xgboost_model
        if isinstance(model, dict) and not hasattr(model, 'predict'):
            for key, value in model.items():
                if hasattr(value, 'predict'):
                    model = value
                    break
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        results = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logger.info(f"Model accuracy: {accuracy:.2%}")
        return results
