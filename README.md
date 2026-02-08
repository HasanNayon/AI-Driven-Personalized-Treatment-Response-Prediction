# 🧠 Mental Health Treatment Response Prediction System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

An AI-driven decision support system for predicting youth mental health treatment responses using fine-tuned Llama 3.2 1B and XGBoost models.

## 🎯 Project Overview

This system combines **clinical data**, **therapy notes**, and **patient self-reports** to predict treatment outcomes for youth mental health interventions. It helps clinicians make data-driven decisions about treatment plans.

### Key Features

- 🤖 **Fine-tuned LLM**: Llama 3.2 1B model for extracting insights from clinical text
- 📊 **XGBoost Classifier**: High-accuracy prediction model (88-92% accuracy)
- 🔍 **Multimodal Analysis**: Integrates structured clinical data with unstructured text
- 🌐 **REST API**: Flask-based API for easy integration
- 📈 **Comprehensive Evaluation**: Detailed visualizations and performance metrics
- 💡 **Explainable AI**: Feature importance and prediction explanations

### Treatment Response Categories

- **Responder**: ≥50% symptom reduction
- **Partial Responder**: 25-49% symptom reduction  
- **Non-Responder**: <25% symptom reduction

## 📁 Project Structure

```
mental health/
├── src/                          # Source code
│   ├── config.py                 # Configuration management
│   ├── logger.py                 # Logging setup
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py          # Data preprocessing & feature engineering
│   ├── llm_extractor.py          # LLM feature extraction
│   ├── model_manager.py          # Model management & inference
│   ├── api.py                    # Flask REST API
│   ├── evaluator.py              # Model evaluation
│   └── visualizer.py             # Visualization tools
├── Dataset/                      # Dataset directory
│   ├── processed/                # Processed training data
│   │   ├── patient_profiles.csv
│   │   ├── therapy_notes.csv
│   │   ├── digital_therapy_chats.csv
│   │   └── patient_reddit_posts.csv
│   └── raw data/                 # Raw data archives
├── LLM Fine tuning model/        # Fine-tuned Llama 3.2 1B model
├── LLM fine Tuniing xgboost model/  # Trained XGBoost model
│   └── xgboost_llm_enhanced.pkl
├── notebook/                     # Jupyter notebooks
├── outputs/                      # Evaluation results & visualizations
├── logs/                         # Application logs
├── app.py                        # Main CLI application
├── examples.py                   # Usage examples
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository (if applicable)
cd "d:\mental health"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure:

```bash
copy .env.example .env
```

Edit `.env` with your settings (optional - defaults work out of the box).

### 3. Run Examples

```bash
# Run all examples
python examples.py

# Run specific example
python examples.py 1  # Single patient prediction
python examples.py 2  # Batch prediction
python examples.py 3  # Model evaluation
python examples.py 4  # Prediction explanation
python examples.py 5  # Generate visualizations
```

## 💻 Usage

### Command Line Interface

#### Predict for a New Patient

```bash
python app.py predict \
  --patient-id P_NEW_001 \
  --age 17 \
  --gender F \
  --baseline-phq9 18 \
  --baseline-gad7 15 \
  --baseline-severity moderate-severe \
  --treatment-type CBT+Digital \
  --treatment-duration-weeks 12 \
  --session-attendance-rate 0.85 \
  --digital-engagement-score 0.7 \
  --therapy-notes "Patient shows good engagement..."
```

#### Predict for Existing Patient

```bash
python app.py predict-db --patient-id P0001
```

#### Batch Predictions

```bash
python app.py batch --input-file patients.csv --output-file predictions.csv
```

#### Model Evaluation

```bash
python app.py evaluate --output-dir outputs/evaluation
```

#### Start API Server

```bash
python app.py api
```

API will be available at `http://localhost:5000`  
Documentation at `http://localhost:5000/apidocs`

### Python API

```python
from src.model_manager import ModelManager

# Initialize model
model_manager = ModelManager()

# Define patient information
patient_info = {
    'patient_id': 'P_TEST_001',
    'age': 16,
    'gender': 'F',
    'baseline_phq9': 18,
    'baseline_gad7': 16,
    'baseline_severity': 'moderate-severe',
    'treatment_type': 'CBT+Digital',
    'treatment_duration_weeks': 12,
    'session_attendance_rate': 0.85,
    'digital_engagement_score': 0.75,
    'outcome_phq9': 0,
    'outcome_gad7': 0
}

# Make prediction
result = model_manager.predict_single_patient(
    patient_info,
    therapy_notes="Patient shows engagement...",
    digital_chats="Feeling better today...",
    reddit_posts=""
)

print(f"Predicted Response: {result['predicted_response']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Recommendation: {result['recommendation']}")
```

## 🌐 REST API

### Endpoints

#### Health Check
```http
GET /health
```

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "patient_id": "P9999",
  "age": 17,
  "gender": "F",
  "baseline_phq9": 18,
  "baseline_gad7": 15,
  "baseline_severity": "moderate-severe",
  "treatment_type": "CBT+Digital",
  "treatment_duration_weeks": 12,
  "session_attendance_rate": 0.85,
  "digital_engagement_score": 0.7,
  "therapy_notes": "Optional therapy notes...",
  "digital_chats": "Optional chat transcripts...",
  "reddit_posts": "Optional self-reports..."
}
```

#### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "patients": [
    { /* patient 1 data */ },
    { /* patient 2 data */ }
  ]
}
```

#### Get Patient from Database
```http
GET /patient/P0001
```

#### Get Statistics
```http
GET /statistics
```

### API Documentation

Full interactive API documentation available at `/apidocs` when server is running.

## 📊 Model Performance

- **Accuracy**: 88-92%
- **Balanced Accuracy**: ~89%
- **F1 Score (weighted)**: ~90%
- **ROC AUC**: ~0.93

### Performance by Treatment Type

| Treatment Type | Accuracy | F1 Score |
|---------------|----------|----------|
| CBT | 90% | 0.89 |
| Medication | 87% | 0.86 |
| CBT+Medication | 92% | 0.91 |
| Digital Therapy | 85% | 0.84 |
| CBT+Digital | 91% | 0.90 |

## 🔧 Configuration

Edit `config.yaml` to customize:

- Model paths
- Dataset paths
- LLM parameters (max length, batch size, device)
- XGBoost parameters
- API settings
- Logging configuration

## 📈 Evaluation Tools

### Generate Visualizations

```python
from src.visualizer import Visualizer
from src.data_loader import DataLoader

visualizer = Visualizer(output_dir="outputs")
data_loader = DataLoader()
patient_df = data_loader.load_patient_profiles()

# Generate plots
visualizer.plot_treatment_distribution(patient_df)
visualizer.plot_clinical_scores(patient_df)
visualizer.create_interactive_dashboard(patient_df)
```

### Evaluate Model

```python
from src.evaluator import ModelEvaluator

evaluator = ModelEvaluator(output_dir="outputs")
report = evaluator.generate_evaluation_report(
    model, X_test, y_test, patient_df, feature_names
)
```

## 🛠️ Development

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- XGBoost 2.0+
- Flask 3.0+
- CUDA (optional, for GPU acceleration)

### Testing

```bash
# Run examples to test functionality
python examples.py

# Test API
python app.py api
# Then visit http://localhost:5000/apidocs
```

## 📝 Dataset

The system uses synthetic clinical data for youth mental health (ages 13-22):

- **Patient Profiles**: Demographics, baseline scores, treatment assignments, outcomes
- **Therapy Notes**: Longitudinal clinical session notes
- **Digital Chats**: Digital therapy conversation transcripts  
- **Reddit Posts**: Simulated self-reported mental health posts

See `Dataset/README.md` for detailed dataset documentation.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This system is designed as a **decision support tool** for clinical use. It should:

- ✅ Be used to **augment** clinical judgment, not replace it
- ✅ Be reviewed by qualified mental health professionals
- ✅ Be used in conjunction with comprehensive clinical assessment
- ❌ **NOT** be used as the sole basis for treatment decisions
- ❌ **NOT** be used without clinical oversight

## 📧 Contact

For questions or issues, please open an issue in the repository.

## 🙏 Acknowledgments

- Fine-tuned using Meta's Llama 3.2 1B model
- Built with Hugging Face Transformers
- XGBoost for classification
- Flask for API development

---

**Version**: 1.0.0  
**Last Updated**: February 2026