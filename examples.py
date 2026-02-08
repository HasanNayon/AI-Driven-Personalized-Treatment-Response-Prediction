"""
Example Usage Scripts
Demonstrations of how to use the Mental Health Treatment Response Prediction System
"""

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_manager import ModelManager
from src.evaluator import ModelEvaluator
from src.logger import get_logger

logger = get_logger(__name__)


def example_1_single_patient_prediction():
    """Example 1: Predict treatment response for a single patient"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Patient Prediction")
    print("="*60 + "\n")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Define patient information
    patient_info = {
        'patient_id': 'P_NEW_001',
        'age': 16,
        'gender': 'F',
        'baseline_phq9': 18,
        'baseline_gad7': 16,
        'baseline_severity': 'moderate-severe',
        'treatment_type': 'CBT+Digital',
        'treatment_duration_weeks': 12,
        'session_attendance_rate': 0.85,
        'digital_engagement_score': 0.75,
        'outcome_phq9': 0,  # Placeholder
        'outcome_gad7': 0   # Placeholder
    }
    
    # Optional: Add therapy notes
    therapy_notes = """
    Patient shows good engagement in CBT sessions. 
    Demonstrates understanding of cognitive restructuring techniques.
    Reports increased use of coping strategies between sessions.
    """
    
    # Make prediction
    result = model_manager.predict_single_patient(
        patient_info,
        therapy_notes=therapy_notes
    )
    
    # Display results
    print(f"Patient ID: {result['patient_id']}")
    print(f"Predicted Response: {result['predicted_response']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nProbabilities:")
    for response, prob in result['probabilities'].items():
        print(f"  {response}: {prob:.1%}")
    print(f"\nRecommendation: {result['recommendation']}")
    print("\n" + "="*60 + "\n")


def example_2_batch_prediction():
    """Example 2: Batch prediction for multiple patients"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Prediction from Database")
    print("="*60 + "\n")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    model_manager = ModelManager()
    
    # Load data
    all_data = data_loader.load_all_data()
    
    # Prepare data
    processed_data, target = preprocessor.prepare_training_data(
        all_data['patient_profiles'][:10],  # First 10 patients
        all_data['therapy_notes'],
        all_data['digital_chats'],
        all_data['reddit_posts']
    )
    
    # Make predictions
    predictions, probabilities = model_manager.predict(
        processed_data,
        return_probabilities=True
    )
    
    # Display results
    print(f"Predicted {len(predictions)} patients")
    print(f"\nPrediction distribution:")
    import pandas as pd
    print(pd.Series(predictions).value_counts())
    print("\n" + "="*60 + "\n")


def example_3_model_evaluation():
    """Example 3: Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Evaluation")
    print("="*60 + "\n")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    model_manager = ModelManager()
    evaluator = ModelEvaluator(output_dir="outputs/examples")
    
    # Load and prepare data
    all_data = data_loader.load_all_data()
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
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    # Make predictions
    y_pred = model_manager.xgboost_model.predict(X_test)
    y_pred_proba = model_manager.xgboost_model.predict_proba(X_test)
    
    # Evaluate
    results = evaluator.evaluate_predictions(
        y_test, y_pred, y_pred_proba,
        class_names=['non-responder', 'partial', 'responder']
    )
    
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"F1 Score: {results['f1_weighted']:.1%}")
    print(f"\nDetailed results saved to outputs/examples/")
    print("\n" + "="*60 + "\n")


def example_4_explain_prediction():
    """Example 4: Explain prediction with feature importance"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Prediction Explanation")
    print("="*60 + "\n")
    
    # Initialize components
    data_loader = DataLoader()
    model_manager = ModelManager()
    
    # Get a patient from database
    patient_data = data_loader.get_patient_data('P0001')
    
    if not patient_data['profile'].empty:
        patient_row = patient_data['profile'].iloc[0]
        
        patient_info = {
            'patient_id': patient_row['patient_id'],
            'age': patient_row['age'],
            'gender': patient_row['gender'],
            'baseline_phq9': patient_row['baseline_phq9'],
            'baseline_gad7': patient_row['baseline_gad7'],
            'baseline_severity': patient_row['baseline_severity'],
            'treatment_type': patient_row['treatment_type'],
            'treatment_duration_weeks': patient_row['treatment_duration_weeks'],
            'session_attendance_rate': patient_row['session_attendance_rate'],
            'digital_engagement_score': patient_row['digital_engagement_score'],
            'outcome_phq9': 0,
            'outcome_gad7': 0
        }
        
        # Make prediction with explanation
        result = model_manager.predict_single_patient(patient_info)
        
        print(f"Patient: {result['patient_id']}")
        print(f"Prediction: {result['predicted_response']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nTop Influential Features:")
        for i, feature in enumerate(result['top_features'][:5], 1):
            print(f"  {i}. {feature['feature']}: {feature['value']:.3f} (importance: {feature['importance']:.3f})")
    
    print("\n" + "="*60 + "\n")


def example_5_visualizations():
    """Example 5: Generate visualizations"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Generate Visualizations")
    print("="*60 + "\n")
    
    from src.visualizer import Visualizer
    
    # Initialize
    data_loader = DataLoader()
    visualizer = Visualizer(output_dir="outputs/visualizations")
    
    # Load patient data
    patient_df = data_loader.load_patient_profiles()
    
    # Generate visualizations
    print("Generating treatment distribution plot...")
    visualizer.plot_treatment_distribution(patient_df)
    
    print("Generating clinical scores plot...")
    visualizer.plot_clinical_scores(patient_df)
    
    print("Generating interactive dashboard...")
    visualizer.create_interactive_dashboard(patient_df)
    
    print(f"\nVisualizations saved to outputs/visualizations/")
    print("  - treatment_distribution.png")
    print("  - clinical_scores.png")
    print("  - dashboard.html (open in browser)")
    print("\n" + "="*60 + "\n")


def run_all_examples():
    """Run all examples"""
    try:
        example_1_single_patient_prediction()
        example_2_batch_prediction()
        example_3_model_evaluation()
        example_4_explain_prediction()
        example_5_visualizations()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_single_patient_prediction()
        elif example_num == '2':
            example_2_batch_prediction()
        elif example_num == '3':
            example_3_model_evaluation()
        elif example_num == '4':
            example_4_explain_prediction()
        elif example_num == '5':
            example_5_visualizations()
        else:
            print("Usage: python examples.py [1-5]")
    else:
        run_all_examples()
