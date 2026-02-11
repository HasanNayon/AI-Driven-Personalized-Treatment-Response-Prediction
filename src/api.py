from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger, swag_from
import pandas as pd
import numpy as np
from typing import Dict, List
import traceback

from src.config import config
from src.logger import get_logger
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_manager import ModelManager

logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Swagger configuration
app.config['SWAGGER'] = {
    'title': 'Mental Health Treatment Response API',
    'uiversion': 3,
    'version': '1.0.0',
    'description': 'API for youth mental health treatment response prediction'
}
swagger = Swagger(app)

# Initialize components
data_loader = DataLoader()
model_manager = ModelManager()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    ---
    responses:
      200:
        description: Service is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            models_loaded:
              type: object
    """
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'xgboost': model_manager.xgboost_model is not None,
            'llm': model_manager.llm_extractor is not None
        },
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict_treatment_response():
    """
    Predict treatment response for a patient
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - patient_id
            - age
            - gender
            - baseline_phq9
            - baseline_gad7
            - baseline_severity
            - treatment_type
            - treatment_duration_weeks
            - session_attendance_rate
          properties:
            patient_id:
              type: string
              example: "P9999"
            age:
              type: integer
              example: 17
            gender:
              type: string
              example: "F"
            baseline_phq9:
              type: integer
              example: 18
            baseline_gad7:
              type: integer
              example: 15
            baseline_severity:
              type: string
              example: "moderate-severe"
            treatment_type:
              type: string
              example: "CBT+Digital"
            treatment_duration_weeks:
              type: integer
              example: 12
            session_attendance_rate:
              type: number
              example: 0.85
            digital_engagement_score:
              type: number
              example: 0.7
            therapy_notes:
              type: string
              example: "Patient shows good engagement..."
            digital_chats:
              type: string
              example: "Patient: I've been feeling better..."
            reddit_posts:
              type: string
              example: "Feeling anxious about school..."
    responses:
      200:
        description: Prediction successful
        schema:
          type: object
          properties:
            patient_id:
              type: string
            predicted_response:
              type: string
            confidence:
              type: number
            probabilities:
              type: object
            recommendation:
              type: string
      400:
        description: Invalid input
      500:
        description: Prediction error
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'patient_id', 'age', 'gender', 'baseline_phq9', 'baseline_gad7',
            'baseline_severity', 'treatment_type', 'treatment_duration_weeks',
            'session_attendance_rate'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Extract patient info
        patient_info = {
            'patient_id': data['patient_id'],
            'age': int(data['age']),
            'gender': data['gender'],
            'baseline_phq9': int(data['baseline_phq9']),
            'baseline_gad7': int(data['baseline_gad7']),
            'baseline_severity': data['baseline_severity'],
            'treatment_type': data['treatment_type'],
            'treatment_duration_weeks': int(data['treatment_duration_weeks']),
            'session_attendance_rate': float(data['session_attendance_rate']),
            'digital_engagement_score': float(data.get('digital_engagement_score', 0)),
            'outcome_phq9': 0,  # Placeholder for preprocessing
            'outcome_gad7': 0   # Placeholder for preprocessing
        }
        
        # Extract text data
        therapy_notes = data.get('therapy_notes', '')
        digital_chats = data.get('digital_chats', '')
        reddit_posts = data.get('reddit_posts', '')
        
        # Make prediction
        result = model_manager.predict_single_patient(
            patient_info,
            therapy_notes=therapy_notes,
            digital_chats=digital_chats,
            reddit_posts=reddit_posts
        )
        
        logger.info(f"Prediction completed for patient {patient_info['patient_id']}")
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict treatment response for multiple patients
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - patients
          properties:
            patients:
              type: array
              items:
                type: object
    responses:
      200:
        description: Batch prediction successful
        schema:
          type: object
          properties:
            predictions:
              type: array
      400:
        description: Invalid input
      500:
        description: Prediction error
    """
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({'error': 'No patients data provided'}), 400
        
        patients = data['patients']
        
        if not isinstance(patients, list) or len(patients) == 0:
            return jsonify({'error': 'Patients must be a non-empty array'}), 400
        
        results = []
        
        for patient_data in patients:
            try:
                # Extract patient info
                patient_info = {
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'age': int(patient_data['age']),
                    'gender': patient_data['gender'],
                    'baseline_phq9': int(patient_data['baseline_phq9']),
                    'baseline_gad7': int(patient_data['baseline_gad7']),
                    'baseline_severity': patient_data['baseline_severity'],
                    'treatment_type': patient_data['treatment_type'],
                    'treatment_duration_weeks': int(patient_data['treatment_duration_weeks']),
                    'session_attendance_rate': float(patient_data['session_attendance_rate']),
                    'digital_engagement_score': float(patient_data.get('digital_engagement_score', 0)),
                    'outcome_phq9': 0,
                    'outcome_gad7': 0
                }
                
                # Make prediction
                result = model_manager.predict_single_patient(
                    patient_info,
                    therapy_notes=patient_data.get('therapy_notes', ''),
                    digital_chats=patient_data.get('digital_chats', ''),
                    reddit_posts=patient_data.get('reddit_posts', '')
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting for patient: {e}")
                results.append({
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'error': str(e)
                })
        
        logger.info(f"Batch prediction completed for {len(results)} patients")
        return jsonify({'predictions': results}), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/patient/<patient_id>', methods=['GET'])
def get_patient_prediction(patient_id: str):
    """
    Get prediction for an existing patient from database
    ---
    parameters:
      - name: patient_id
        in: path
        type: string
        required: true
        description: Patient ID (e.g., P0001)
    responses:
      200:
        description: Prediction retrieved
      404:
        description: Patient not found
      500:
        description: Server error
    """
    try:
        # Load patient data
        patient_data = data_loader.get_patient_data(patient_id)
        
        if patient_data['profile'].empty:
            return jsonify({'error': f'Patient {patient_id} not found'}), 404
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        
        # Load all data for fitting
        all_data = data_loader.load_all_data()
        preprocessor.prepare_training_data(
            all_data['patient_profiles'],
            all_data['therapy_notes'],
            all_data['digital_chats'],
            all_data['reddit_posts']
        )
        
        # Prepare patient data for inference
        processed_data = preprocessor.prepare_inference_data(patient_data)
        
        # Make prediction
        predictions, probabilities = model_manager.predict(
            processed_data,
            return_probabilities=True
        )
        
        result = {
            'patient_id': patient_id,
            'predicted_response': predictions[0],
            'confidence': float(np.max(probabilities[0])),
            'probabilities': {
                'non-responder': float(probabilities[0][0]),
                'partial': float(probabilities[0][1]),
                'responder': float(probabilities[0][2])
            },
            'actual_response': patient_data['profile']['treatment_response'].values[0]
            if 'treatment_response' in patient_data['profile'].columns else None
        }
        
        logger.info(f"Retrieved prediction for patient {patient_id}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error retrieving patient prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/treatments', methods=['GET'])
def get_treatment_types():
    """
    Get available treatment types
    ---
    responses:
      200:
        description: Treatment types retrieved
        schema:
          type: object
          properties:
            treatments:
              type: array
              items:
                type: string
    """
    treatments = config.get('treatments', [])
    return jsonify({'treatments': treatments}), 200


@app.route('/severity_levels', methods=['GET'])
def get_severity_levels():
    """
    Get baseline severity levels
    ---
    responses:
      200:
        description: Severity levels retrieved
        schema:
          type: object
          properties:
            severity_levels:
              type: array
              items:
                type: string
    severity_levels = config.get('severity_levels', [])
    return jsonify({'severity_levels': severity_levels}), 200


@app.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Get dataset statistics
    ---
    responses:
      200:
        description: Statistics retrieved
      500:
        description: Server error
    """
    try:
        # Load patient profiles
        patient_profiles = data_loader.load_patient_profiles()
        
        stats = {
            'total_patients': len(patient_profiles),
            'treatment_distribution': patient_profiles['treatment_type'].value_counts().to_dict(),
            'response_distribution': patient_profiles['treatment_response'].value_counts().to_dict(),
            'severity_distribution': patient_profiles['baseline_severity'].value_counts().to_dict(),
            'gender_distribution': patient_profiles['gender'].value_counts().to_dict(),
            'age_statistics': {
                'mean': float(patient_profiles['age'].mean()),
                'min': int(patient_profiles['age'].min()),
                'max': int(patient_profiles['age'].max()),
                'std': float(patient_profiles['age'].std())
            }
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def run_api():
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 5000)
    debug = config.get('api.debug', True)
    
    logger.info(f"Starting API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api()
