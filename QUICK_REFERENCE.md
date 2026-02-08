# Quick Reference Guide

## Common Commands

### CLI Usage

```bash
# Predict for a new patient
python app.py predict \
  --patient-id P_NEW_001 \
  --age 17 \
  --gender F \
  --baseline-phq9 18 \
  --baseline-gad7 15 \
  --baseline-severity moderate-severe \
  --treatment-type CBT+Digital \
  --treatment-duration-weeks 12 \
  --session-attendance-rate 0.85

# Predict for existing patient  
python app.py predict-db --patient-id P0001

# Batch predictions
python app.py batch --input-file patients.csv

# Evaluate model
python app.py evaluate --output-dir outputs

# Start API server
python app.py api

# Start web interface
python web_interface.py
```

### Python Usage

```python
from src.model_manager import ModelManager

# Initialize
model = ModelManager()

# Predict
patient = {
    'patient_id': 'P001',
    'age': 17,
    'gender': 'F',
    'baseline_phq9': 18,
    'baseline_gad7': 15,
    'baseline_severity': 'moderate-severe',
    'treatment_type': 'CBT+Digital',
    'treatment_duration_weeks': 12,
    'session_attendance_rate': 0.85,
    'digital_engagement_score': 0.7,
    'outcome_phq9': 0,
    'outcome_gad7': 0
}

result = model.predict_single_patient(patient)
print(f"Response: {result['predicted_response']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "age": 17,
    "gender": "F",
    "baseline_phq9": 18,
    "baseline_gad7": 15,
    "baseline_severity": "moderate-severe",
    "treatment_type": "CBT+Digital",
    "treatment_duration_weeks": 12,
    "session_attendance_rate": 0.85
  }'

# Get patient from database
curl http://localhost:5000/patient/P0001

# Get statistics
curl http://localhost:5000/statistics
```

## Configuration

### Edit config.yaml

```yaml
# Change device (cpu/cuda)
llm:
  device: "cpu"
  
# Change API port
api:
  port: 5000
  
# Change log level
logging:
  level: "INFO"
```

### Edit .env

```env
# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Model device
MODEL_DEVICE=cpu

# API settings
API_PORT=5000
```

## Troubleshooting

### Import errors
```bash
pip install -r requirements.txt
```

### Model not found
Check paths in config.yaml match your directory structure

### CUDA errors
Set device to CPU in config.yaml:
```yaml
llm:
  device: "cpu"
```

### Port in use
Change port in .env:
```env
API_PORT=5001
```

## File Locations

- **Configuration**: `config.yaml`, `.env`
- **Models**: `LLM Fine tuning model/`, `LLM fine Tuniing xgboost model/`
- **Data**: `Dataset/processed/`
- **Logs**: `logs/app.log`
- **Outputs**: `outputs/`
- **Source Code**: `src/`

## Key Modules

- `src/config.py` - Configuration management
- `src/data_loader.py` - Data loading
- `src/preprocessing.py` - Feature engineering
- `src/llm_extractor.py` - LLM embeddings
- `src/model_manager.py` - Model inference
- `src/evaluator.py` - Model evaluation
- `src/visualizer.py` - Visualization
- `src/api.py` - REST API

## Treatment Types

- CBT
- Medication
- Digital_Therapy
- CBT+Medication
- CBT+Digital

## Severity Levels

- mild-moderate
- moderate
- moderate-severe
- severe

## Response Categories

- **responder**: ≥50% symptom reduction
- **partial**: 25-49% symptom reduction
- **non-responder**: <25% symptom reduction

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quickstart
python quickstart.py

# 3. Run examples
python examples.py

# 4. Start web interface
python web_interface.py
```

## Documentation

- `README.md` - Main documentation
- `SETUP_GUIDE.md` - Detailed setup
- `API_DOCUMENTATION.md` - API reference
- `QUICK_REFERENCE.md` - This file
- `Dataset/README.md` - Dataset info
