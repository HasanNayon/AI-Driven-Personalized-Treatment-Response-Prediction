# Mental Health Treatment Response Prediction

Predicts youth mental health treatment responses using a fine-tuned Llama 3.2 1B model combined with XGBoost. Takes in clinical data, therapy notes, and patient self-reports to classify treatment outcomes as responder, partial, or non-responder.

## Setup

```bash
pip install -r requirements.txt
```

Models should be placed in:
- `LLM Fine tuning model/` — fine-tuned Llama 3.2 1B LoRA adapters
- `LLM fine Tuniing xgboost model/` — trained XGBoost model (.pkl)

Dataset CSVs go in `Dataset/processed/`.

## Usage

### CLI

```bash
# predict for an existing patient
python app.py predict-db --patient-id P0001

# predict for a new patient
python app.py predict --patient-id P_NEW_001 --age 17 --gender F \
  --baseline-phq9 18 --baseline-gad7 15 --baseline-severity moderate-severe \
  --treatment-type CBT+Digital --treatment-duration-weeks 12 \
  --session-attendance-rate 0.85 --digital-engagement-score 0.7

# batch predict
python app.py batch --input-file patients.csv --output-file predictions.csv

# evaluate model
python app.py evaluate --output-dir outputs/evaluation

# start api server
python app.py api
```

### Python

```python
from src.model_manager import ModelManager

manager = ModelManager()
result = manager.predict_single_patient(
    patient_info={
        'patient_id': 'P_TEST',
        'age': 16, 'gender': 'F',
        'baseline_phq9': 18, 'baseline_gad7': 16,
        'baseline_severity': 'moderate-severe',
        'treatment_type': 'CBT+Digital',
        'treatment_duration_weeks': 12,
        'session_attendance_rate': 0.85,
        'digital_engagement_score': 0.75,
        'outcome_phq9': 0, 'outcome_gad7': 0
    },
    therapy_notes="Patient shows good engagement in sessions..."
)
print(result['predicted_response'], result['confidence'])
```

### REST API

```bash
python app.py api
# http://localhost:5000/apidocs for Swagger docs
```

Endpoints:
- `GET /health` — health check
- `POST /predict` — single patient prediction
- `POST /predict/batch` — batch prediction
- `GET /patient/<id>` — get patient from database
- `GET /statistics` — dataset statistics

## Project Structure

```
src/                  # core modules
  config.py           # config loading
  data_loader.py      # data loading
  preprocessing.py    # feature engineering
  llm_extractor.py    # LLM feature extraction
  model_manager.py    # model inference
  evaluator.py        # evaluation metrics
  visualizer.py       # plotting
  api.py              # Flask API
notebook/             # training notebooks
Dataset/processed/    # processed CSV data
app.py                # CLI entry point
web_interface.py      # web UI
config.yaml           # configuration
```

## Notebooks

- `notebook/finetune_llama1b.ipynb` — fine-tune Llama 3.2 1B with QLoRA on therapy notes
- `notebook/2_xgboost_llm_training_colab.ipynb` — train XGBoost using LLM embeddings
- `notebook/treatment_response_prediction.ipynb` — full pipeline: EDA, baseline models, multimodal XGBoost

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA optional)
- See `requirements.txt` for full list

## Disclaimer

This is a decision support tool. Predictions should be reviewed by qualified clinicians and not used as the sole basis for treatment decisions.
