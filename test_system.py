import sys
import os
from datetime import datetime

def print_header(title):
    print("\n" + "-"*50)
    print(f"  {title}")
    print("-"*50 + "\n")

def test_imports():
    print("TEST 1: Module Imports")
    print("-" * 40)
    
    modules = [
        ('src.config', 'config'),
        ('src.logger', 'get_logger'),
        ('src.model_manager', 'ModelManager'),
        ('src.preprocessing', 'DataPreprocessor'),
        ('src.llm_extractor', 'LLMFeatureExtractor'),
        ('src.evaluator', 'ModelEvaluator'),
        ('src.visualizer', 'Visualizer'),
    ]
    
    passed = 0
    for module, obj in modules:
        try:
            exec(f"from {module} import {obj}")
            print(f"  [OK] {module}.{obj}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {module}.{obj} - Error: {e}")
    
    print(f"\nResult: {passed}/{len(modules)} modules loaded successfully\n")
    return passed == len(modules)

def test_configuration():
    print("TEST 2: Configuration")
    print("-" * 40)
    
    try:
        from src.config import config
        
        print(f"  [OK] Configuration loaded")
        print(f"    - Model paths defined: {len(config.model_paths)}")
        print(f"    - Data paths defined: {len(config.data_paths)}")
        print(f"    - XGBoost path: {config.model_paths['xgboost']}")
        print(f"    - LLM path: {config.model_paths['llm']}")
        print("\nResult: Configuration test passed\n")
        return True
    except Exception as e:
        print(f"  [X] Configuration error: {e}\n")
        return False

def test_model_loading():
    print("TEST 3: Model Loading")
    print("-" * 40)
    
    try:
        from src.model_manager import ModelManager
        
        print("  Loading models...")
        mm = ModelManager()
        
        xgb_status = "[OK] Loaded" if mm.xgboost_model is not None else "[X] Not loaded"
        llm_status = "[OK] Loaded" if mm.llm_extractor is not None else "[X] Not loaded"
        
        print(f"    - XGBoost model: {xgb_status}")
        print(f"    - LLM extractor: {llm_status}")
        
        if mm.feature_names:
            print(f"    - Features: {len(mm.feature_names)} features")
        
        success = mm.xgboost_model is not None
        print(f"\nResult: Model loading {'passed' if success else 'failed'}\n")
        return success, mm
    except Exception as e:
        print(f"  [X] Model loading error: {e}\n")
        return False, None

def test_prediction(model_manager):
    print("TEST 4: Single Patient Prediction")
    print("-" * 40)
    
    try:
        # Test patient data
        patient_info = {
            'patient_id': 'TEST_PATIENT_001',
            'age': 17,
            'gender': 'Male',
            'baseline_phq9': 20,
            'baseline_gad7': 18,
            'baseline_severity': 'severe',
            'treatment_type': 'CBT',
            'treatment_duration_weeks': 16,
            'session_attendance_rate': 0.90,
            'digital_engagement_score': 0.65,
            'outcome_phq9': 0,
            'outcome_gad7': 0
        }
        
        therapy_notes = """
        Patient presents with severe depression and anxiety symptoms.
        Good engagement in CBT sessions. Demonstrates effort in homework.
        Shows understanding of cognitive restructuring techniques.
        """
        
        print("  Making prediction for test patient...")
        result = model_manager.predict_single_patient(
            patient_info,
            therapy_notes=therapy_notes
        )
        
        print(f"\n  Results:")
        print(f"    Patient ID: {result['patient_id']}")
        print(f"    Predicted Response: {result['predicted_response'].upper()}")
        print(f"    Confidence: {result['confidence']:.1%}")
        
        print(f"\n  Probabilities:")
        for response, prob in result['probabilities'].items():
            print(f"    {response:15s}: {prob:6.1%}")
        
        print(f"\n  Top Contributing Features:")
        for i, feat in enumerate(result.get('top_features', [])[:5], 1):
            print(f"    {i}. {feat['feature']}: {feat['value']:.3f}")
        
        print("\nResult: Prediction test passed\n")
        return True
    except Exception as e:
        print(f"  [X] Prediction error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    print("TEST 5: File Structure")
    print("-" * 40)
    
    essential_files = {
        'Application Files': [
            'app.py',
            'web_interface.py',
            'examples.py',
            'quickstart.py',
        ],
        'Configuration': [
            'config.yaml',
            'requirements.txt',
            '.gitignore',
        ],
        'Documentation': [
            'README.md',
        ],
        'Source Code': [
            'src/__init__.py',
            'src/model_manager.py',
            'src/llm_extractor.py',
            'src/preprocessing.py',
            'src/config.py',
            'src/api.py',
        ],
        'Models': [
            'LLM fine Tuniing xgboost model/xgboost_llm_enhanced.pkl',
        ]
    }
    
    total_files = 0
    found_files = 0
    
    for category, files in essential_files.items():
        print(f"\n  {category}:")
        for file in files:
            total_files += 1
            exists = os.path.exists(file)
            symbol = '[OK]' if exists else '[X]'
            if exists:
                found_files += 1
                size = os.path.getsize(file)
                size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                print(f"    {symbol} {file} ({size_str})")
            else:
                print(f"    {symbol} {file}")
    
    print(f"\nResult: {found_files}/{total_files} files found\n")
    return found_files >= total_files * 0.8  # 80% threshold

def main():
    print_header("System Test")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['configuration'] = test_configuration()
    results['model_loading'], model_manager = test_model_loading()
    
    if model_manager:
        results['prediction'] = test_prediction(model_manager)
    else:
        print("TEST 4: Skipped (model not loaded)\n")
        results['prediction'] = False
    
    results['file_structure'] = test_file_structure()
    
    # Summary
    print_header("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"  {test_name.replace('_', ' ').title():20s}: {status}")
    
    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n  All tests passed.")
    elif passed_tests >= total_tests * 0.8:
        print("\n  Most tests passed. Minor issues present.")
    else:
        print("\n  Multiple failures. Check configuration.")
    
    print()
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
