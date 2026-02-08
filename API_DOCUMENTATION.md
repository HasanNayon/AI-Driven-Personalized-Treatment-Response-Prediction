# API Documentation

## Mental Health Treatment Response Prediction API

Base URL: `http://localhost:5000`

---

## Endpoints

### 1. Health Check

Check if the API is running and models are loaded.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": {
    "xgboost": true,
    "llm": true
  },
  "version": "1.0.0"
}
```

---

### 2. Single Patient Prediction

Predict treatment response for a single patient.

**Endpoint**: `POST /predict`

**Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
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
  "therapy_notes": "Patient shows good engagement in sessions...",
  "digital_chats": "I've been feeling better lately...",
  "reddit_posts": "Dealing with anxiety about school..."
}
```

**Required Fields**:
- `patient_id`: string
- `age`: integer (13-22)
- `gender`: "M" or "F"
- `baseline_phq9`: integer (0-27)
- `baseline_gad7`: integer (0-21)
- `baseline_severity`: "mild-moderate", "moderate", "moderate-severe", or "severe"
- `treatment_type`: "CBT", "Medication", "Digital_Therapy", "CBT+Medication", or "CBT+Digital"
- `treatment_duration_weeks`: integer (6-18)
- `session_attendance_rate`: float (0.0-1.0)

**Optional Fields**:
- `digital_engagement_score`: float (0.0-1.0, default: 0.0)
- `therapy_notes`: string
- `digital_chats`: string
- `reddit_posts`: string

**Response**:
```json
{
  "patient_id": "P9999",
  "predicted_response": "responder",
  "confidence": 0.87,
  "probabilities": {
    "non-responder": 0.05,
    "partial": 0.08,
    "responder": 0.87
  },
  "top_features": [
    {
      "feature": "baseline_phq9",
      "value": 18.0,
      "importance": 0.234
    },
    ...
  ],
  "recommendation": "Patient likely to respond well to assigned treatment (confidence: 87%). Continue with current treatment plan and monitor progress regularly."
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid input (missing required fields or invalid values)
- `500`: Server error

---

### 3. Batch Prediction

Predict treatment response for multiple patients.

**Endpoint**: `POST /predict/batch`

**Request Body**:
```json
{
  "patients": [
    {
      "patient_id": "P9999",
      "age": 17,
      "gender": "F",
      "baseline_phq9": 18,
      "baseline_gad7": 15,
      "baseline_severity": "moderate-severe",
      "treatment_type": "CBT+Digital",
      "treatment_duration_weeks": 12,
      "session_attendance_rate": 0.85
    },
    {
      "patient_id": "P9998",
      ...
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "patient_id": "P9999",
      "predicted_response": "responder",
      "confidence": 0.87,
      "probabilities": {...},
      "recommendation": "..."
    },
    {
      "patient_id": "P9998",
      ...
    }
  ]
}
```

---

### 4. Get Patient Prediction

Get prediction for an existing patient from the database.

**Endpoint**: `GET /patient/{patient_id}`

**Path Parameters**:
- `patient_id`: Patient ID (e.g., "P0001")

**Response**:
```json
{
  "patient_id": "P0001",
  "predicted_response": "partial",
  "confidence": 0.65,
  "probabilities": {
    "non-responder": 0.15,
    "partial": 0.65,
    "responder": 0.20
  },
  "actual_response": "partial"
}
```

**Status Codes**:
- `200`: Success
- `404`: Patient not found
- `500`: Server error

---

### 5. Get Available Treatment Types

Get list of supported treatment types.

**Endpoint**: `GET /treatments`

**Response**:
```json
{
  "treatments": [
    "CBT",
    "Medication",
    "Digital_Therapy",
    "CBT+Medication",
    "CBT+Digital"
  ]
}
```

---

### 6. Get Severity Levels

Get list of baseline severity levels.

**Endpoint**: `GET /severity_levels`

**Response**:
```json
{
  "severity_levels": [
    "mild-moderate",
    "moderate",
    "moderate-severe",
    "severe"
  ]
}
```

---

### 7. Get Dataset Statistics

Get statistics about the training dataset.

**Endpoint**: `GET /statistics`

**Response**:
```json
{
  "total_patients": 3000,
  "treatment_distribution": {
    "CBT": 600,
    "Medication": 550,
    "Digital_Therapy": 500,
    "CBT+Medication": 700,
    "CBT+Digital": 650
  },
  "response_distribution": {
    "responder": 1950,
    "partial": 750,
    "non-responder": 300
  },
  "severity_distribution": {
    "mild-moderate": 800,
    "moderate": 1100,
    "moderate-severe": 800,
    "severe": 300
  },
  "gender_distribution": {
    "F": 1650,
    "M": 1350
  },
  "age_statistics": {
    "mean": 17.5,
    "min": 13,
    "max": 22,
    "std": 2.3
  }
}
```

---

## Example Usage

### Python

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={
        'patient_id': 'P_TEST_001',
        'age': 17,
        'gender': 'F',
        'baseline_phq9': 18,
        'baseline_gad7': 15,
        'baseline_severity': 'moderate-severe',
        'treatment_type': 'CBT+Digital',
        'treatment_duration_weeks': 12,
        'session_attendance_rate': 0.85,
        'digital_engagement_score': 0.7
    }
)

result = response.json()
print(f"Prediction: {result['predicted_response']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P_TEST_001",
    "age": 17,
    "gender": "F",
    "baseline_phq9": 18,
    "baseline_gad7": 15,
    "baseline_severity": "moderate-severe",
    "treatment_type": "CBT+Digital",
    "treatment_duration_weeks": 12,
    "session_attendance_rate": 0.85
  }'
```

### JavaScript (Fetch)

```javascript
const prediction = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    patient_id: 'P_TEST_001',
    age: 17,
    gender: 'F',
    baseline_phq9: 18,
    baseline_gad7: 15,
    baseline_severity: 'moderate-severe',
    treatment_type: 'CBT+Digital',
    treatment_duration_weeks: 12,
    session_attendance_rate: 0.85
  })
});

const result = await prediction.json();
console.log('Prediction:', result.predicted_response);
console.log('Confidence:', result.confidence);
```

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Description of the error"
}
```

Common error codes:
- `400`: Bad Request - Invalid input or missing required fields
- `404`: Not Found - Resource (e.g., patient) not found
- `500`: Internal Server Error - Server-side error

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider adding rate limiting middleware.

---

## Interactive Documentation

When the API server is running, visit `http://localhost:5000/apidocs` for interactive Swagger documentation where you can test endpoints directly in your browser.

---

## Support

For issues or questions about the API, please refer to the main README or open an issue in the repository.
