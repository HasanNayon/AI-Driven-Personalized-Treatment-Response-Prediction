"""
Main Application
Command-line interface for the Mental Health Treatment Response Prediction System
"""

import argparse
import sys
from pathlib import Path

from src.config import config
from src.logger import get_logger
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_manager import ModelManager
from src.evaluator import ModelEvaluator
from src.api import run_api

logger = get_logger(__name__)


def predict_patient(args):
    """Predict treatment response for a single patient"""
    logger.info("Starting single patient prediction...")
    
    # Initialize components
    model_manager = ModelManager()
    
    # Build patient info from command line arguments
    patient_info = {
        'patient_id': args.patient_id,
        'age': args.age,
        'gender': args.gender,
        'baseline_phq9': args.baseline_phq9,
        'baseline_gad7': args.baseline_gad7,
        'baseline_severity': args.baseline_severity,
        'treatment_type': args.treatment_type,
        'treatment_duration_weeks': args.treatment_duration_weeks,
        'session_attendance_rate': args.session_attendance_rate,
        'digital_engagement_score': args.digital_engagement_score or 0.0,
        'outcome_phq9': 0,  # Placeholder
        'outcome_gad7': 0   # Placeholder
    }
    
    # Make prediction
    result = model_manager.predict_single_patient(
        patient_info,
        therapy_notes=args.therapy_notes or "",
        digital_chats=args.digital_chats or "",
        reddit_posts=args.reddit_posts or ""
    )
    
    # Display results
    print("\n" + "="*60)
    print(f"TREATMENT RESPONSE PREDICTION REPORT")
    print("="*60)
    print(f"Patient ID: {result['patient_id']}")
    print(f"\nPredicted Response: {result['predicted_response'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nClass Probabilities:")
    for response, prob in result['probabilities'].items():
        print(f"  {response:15s}: {prob:.1%}")
    print(f"\nRecommendation:")
    print(f"  {result['recommendation']}")
    print("="*60 + "\n")
    
    logger.info("Prediction completed successfully")


def predict_from_database(args):
    """Predict for an existing patient in the database"""
    logger.info(f"Predicting for patient {args.patient_id} from database...")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    model_manager = ModelManager()
    
    # Load patient data
    patient_data = data_loader.get_patient_data(args.patient_id)
    
    if patient_data['profile'].empty:
        logger.error(f"Patient {args.patient_id} not found in database")
        print(f"Error: Patient {args.patient_id} not found")
        return
    
    # Load all data for fitting preprocessor
    all_data = data_loader.load_all_data()
    preprocessor.prepare_training_data(
        all_data['patient_profiles'],
        all_data['therapy_notes'],
        all_data['digital_chats'],
        all_data['reddit_posts']
    )
    
    # Prepare patient data
    processed_data = preprocessor.prepare_inference_data(patient_data)
    
    # Make prediction
    predictions, probabilities = model_manager.predict(
        processed_data,
        return_probabilities=True
    )
    
    # Get actual response if available
    actual_response = patient_data['profile']['treatment_response'].values[0]
    
    # Display results
    print("\n" + "="*60)
    print(f"TREATMENT RESPONSE PREDICTION REPORT")
    print("="*60)
    print(f"Patient ID: {args.patient_id}")
    print(f"\nPredicted Response: {predictions[0].upper()}")
    print(f"Actual Response: {actual_response.upper()}")
    print(f"Match: {'✓ YES' if predictions[0] == actual_response else '✗ NO'}")
    print(f"\nConfidence: {probabilities[0].max():.1%}")
    print(f"\nClass Probabilities:")
    prob_dict = {
        'non-responder': probabilities[0][0],
        'partial': probabilities[0][1],
        'responder': probabilities[0][2]
    }
    for response, prob in prob_dict.items():
        print(f"  {response:15s}: {prob:.1%}")
    print("="*60 + "\n")
    
    logger.info("Prediction completed successfully")


def evaluate_model(args):
    """Evaluate model performance on test data"""
    logger.info("Starting model evaluation...")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    model_manager = ModelManager()
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    
    # Load all data
    all_data = data_loader.load_all_data()
    
    # Prepare data
    processed_data, target = preprocessor.prepare_training_data(
        all_data['patient_profiles'],
        all_data['therapy_notes'],
        all_data['digital_chats'],
        all_data['reddit_posts']
    )
    
    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(target)
    
    # Prepare features
    X, feature_names = model_manager.prepare_features(processed_data)
    
    # Split data (simple split for demonstration)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    patient_test = all_data['patient_profiles'].iloc[list(idx_test)]
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        model_manager.xgboost_model,
        X_test,
        y_test,
        patient_test,
        feature_names,
        class_names=['non-responder', 'partial', 'responder']
    )
    
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Set Size: {len(y_test)}")
    print(f"Accuracy: {report['overall_metrics']['accuracy']:.1%}")
    print(f"Balanced Accuracy: {report['overall_metrics']['balanced_accuracy']:.1%}")
    print(f"F1 Score (weighted): {report['overall_metrics']['f1_weighted']:.1%}")
    if report['overall_metrics']['roc_auc_weighted']:
        print(f"ROC AUC (weighted): {report['overall_metrics']['roc_auc_weighted']:.1%}")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("="*60 + "\n")
    
    logger.info("Evaluation completed successfully")


def start_api_server(args):
    """Start the Flask API server"""
    logger.info("Starting API server...")
    run_api()


def batch_predict(args):
    """Batch prediction from CSV file"""
    logger.info(f"Running batch predictions from {args.input_file}...")
    
    import pandas as pd
    
    # Load input CSV
    input_df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(input_df)} patients for prediction")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    model_manager = ModelManager()
    
    # Prepare data
    # First fit preprocessor on existing data
    data_loader = DataLoader()
    all_data = data_loader.load_all_data()
    preprocessor.prepare_training_data(
        all_data['patient_profiles'],
        all_data['therapy_notes'],
        all_data['digital_chats'],
        all_data['reddit_posts']
    )
    
    # Prepare input data
    processed_data = preprocessor.prepare_clinical_features(input_df)
    
    # Add empty text columns if not present
    if 'combined_text' not in processed_data.columns:
        processed_data['combined_text'] = ''
    
    # Make predictions
    predictions, probabilities = model_manager.predict(
        processed_data,
        return_probabilities=True
    )
    
    # Add predictions to dataframe
    input_df['predicted_response'] = predictions
    input_df['confidence'] = probabilities.max(axis=1)
    input_df['prob_non_responder'] = probabilities[:, 0]
    input_df['prob_partial'] = probabilities[:, 1]
    input_df['prob_responder'] = probabilities[:, 2]
    
    # Save results
    output_file = args.output_file or args.input_file.replace('.csv', '_predictions.csv')
    input_df.to_csv(output_file, index=False)
    
    print(f"\nBatch prediction completed!")
    print(f"Results saved to: {output_file}")
    print(f"Total patients: {len(input_df)}")
    print(f"Predictions: {dict(pd.Series(predictions).value_counts())}")
    
    logger.info(f"Batch predictions saved to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Mental Health Treatment Response Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict for a new patient')
    predict_parser.add_argument('--patient-id', required=True, help='Patient ID')
    predict_parser.add_argument('--age', type=int, required=True, help='Patient age')
    predict_parser.add_argument('--gender', required=True, choices=['M', 'F'], help='Gender')
    predict_parser.add_argument('--baseline-phq9', type=int, required=True, help='Baseline PHQ-9 score (0-27)')
    predict_parser.add_argument('--baseline-gad7', type=int, required=True, help='Baseline GAD-7 score (0-21)')
    predict_parser.add_argument('--baseline-severity', required=True, 
                               choices=['mild-moderate', 'moderate', 'moderate-severe', 'severe'],
                               help='Baseline severity')
    predict_parser.add_argument('--treatment-type', required=True,
                               choices=['CBT', 'Medication', 'Digital_Therapy', 'CBT+Medication', 'CBT+Digital'],
                               help='Treatment type')
    predict_parser.add_argument('--treatment-duration-weeks', type=int, required=True,
                               help='Treatment duration in weeks')
    predict_parser.add_argument('--session-attendance-rate', type=float, required=True,
                               help='Session attendance rate (0.0-1.0)')
    predict_parser.add_argument('--digital-engagement-score', type=float, default=0.0,
                               help='Digital engagement score (0.0-1.0)')
    predict_parser.add_argument('--therapy-notes', help='Therapy session notes')
    predict_parser.add_argument('--digital-chats', help='Digital therapy chat text')
    predict_parser.add_argument('--reddit-posts', help='Reddit post text')
    
    # Database predict command
    db_predict_parser = subparsers.add_parser('predict-db', help='Predict for existing patient')
    db_predict_parser.add_argument('--patient-id', required=True, help='Patient ID from database')
    
    # Batch predict command
    batch_parser = subparsers.add_parser('batch', help='Batch predictions from CSV')
    batch_parser.add_argument('--input-file', required=True, help='Input CSV file')
    batch_parser.add_argument('--output-file', help='Output CSV file (default: input_predictions.csv)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    try:
        if args.command == 'predict':
            predict_patient(args)
        elif args.command == 'predict-db':
            predict_from_database(args)
        elif args.command == 'batch':
            batch_predict(args)
        elif args.command == 'evaluate':
            evaluate_model(args)
        elif args.command == 'api':
            start_api_server(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
