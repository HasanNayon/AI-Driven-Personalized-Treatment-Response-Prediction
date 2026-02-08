# Youth Mental Health Treatment Response Prediction Dataset

## Project Overview
This dataset supports the development of an AI-driven decision support system for personalized youth mental health care, specifically designed to predict individual patient responses to different therapeutic interventions.

## Dataset Structure

### 📁 processed/ (Primary Training Data)
**Core synthetic clinical data for model training - START HERE**

#### 1. **patient_profiles.csv** (100 patients)
Main patient dataset with demographics, treatment assignments, and outcomes.

**Columns:**
- `patient_id`: Unique identifier (P001-P100)
- `age`: Patient age (13-22 years, youth population)
- `gender`: M/F
- `baseline_phq9`: Depression severity at baseline (0-27, higher = more severe)
- `baseline_gad7`: Anxiety severity at baseline (0-21, higher = more severe)
- `baseline_severity`: Categorical severity (mild-moderate, moderate, moderate-severe, severe)
- `treatment_type`: Assigned intervention
  - `CBT`: Cognitive Behavioral Therapy only
  - `Medication`: Antidepressant medication only
  - `Digital_Therapy`: App-based digital intervention only
  - `CBT+Medication`: Combined therapy and medication
  - `CBT+Digital`: Combined in-person therapy with digital tools
- `treatment_duration_weeks`: Length of treatment (6-18 weeks)
- `session_attendance_rate`: Proportion of sessions attended (0-1)
- `digital_engagement_score`: For digital interventions, app usage score (0-1)
- `outcome_phq9`: Depression severity post-treatment
- `outcome_gad7`: Anxiety severity post-treatment
- `treatment_response`: **TARGET VARIABLE**
  - `responder`: ≥50% symptom reduction
  - `partial`: 25-49% symptom reduction
  - `non-responder`: <25% symptom reduction
- `improvement_percentage`: Actual improvement rate

**Key Statistics:**
- Responders: ~65%
- Partial responders: ~25%
- Non-responders: ~10%

---

#### 2. **therapy_notes.csv**
Longitudinal clinical notes from therapy sessions.

**Columns:**
- `patient_id`: Links to patient_profiles.csv
- `session_number`: Session count (1-9)
- `session_week`: Week of treatment
- `therapist_notes`: Unstructured clinical text (rich NLP features)
- `patient_mood`: Clinician-observed mood state
- `engagement_level`: Session engagement (low/medium/high)

**Use Cases:**
- Extract linguistic features from clinical text
- Track longitudinal mood trajectories
- Identify engagement patterns predicting outcomes

---

#### 3. **digital_therapy_chats.csv**
Chat transcripts from digital therapy interventions.

**Columns:**
- `patient_id`: Links to patient_profiles.csv
- `chat_timestamp`: Message datetime
- `message_type`: patient/bot
- `message_text`: Actual chat content
- `sentiment_score`: Computed sentiment (-1 to 1)
- `emotion_detected`: Primary emotion label

**Use Cases:**
- Analyze patient language patterns in digital therapy
- Compare engagement across digital vs. traditional therapy
- Extract emotional trajectories from patient messages

---

#### 4. **patient_reddit_posts.csv**
Simulated self-reported mental health posts (linked to synthetic patients).

**Columns:**
- `patient_id`: Links to patient_profiles.csv
- `reddit_post_id`: Unique post identifier
- `post_text`: User-generated mental health narrative
- `post_subreddit`: Anxiety/depression/mentalhealth
- `post_sentiment`: Overall sentiment score
- `emotional_intensity`: Intensity of emotional expression (0-1)
- `linguistic_features`: Comma-separated feature tags

**Use Cases:**
- Train LLM embeddings on mental health language
- Extract self-reported symptom patterns
- Compare self-report vs. clinical assessment language

---

### 📁 raw data/ (Original Social Media Data)
Real Reddit posts from mental health communities (2019-2022).

**Structure:** `raw data/YEAR/MONTH/condition/`
- `anx*.csv`: r/Anxiety posts (~24K/month)
- `dep*.csv`: r/depression posts (~72K/month)
- `lone*.csv`: r/lonely posts (~5K/month)
- `sw*.csv`: r/SuicideWatch posts (~30K/month)
- `mh*.csv`: r/mentalhealth posts (~23K/month)

**Total:** ~155,000 posts across 48 months

**Columns:**
- `author`, `created_utc`, `score`, `selftext`, `subreddit`, `title`, `timestamp`

**Use Cases:**
- Pre-train language models on large mental health corpus
- Build symptom classifiers
- Create embeddings for transfer learning

---

### 📁 Labelled Data/ (Curated Subsets)
Manually labeled Reddit posts with risk factor categories.

**Files:**
- `LD DA 1.csv`: Drug & Alcohol mentions (995 posts)
- `LD EL1.csv`: Early Life factors (1,609 posts)
- `LD PF1.csv`: Personality Factors (1,006 posts)
- `LD TS 1.csv`: Trauma & Stress (1,083 posts)

**Use Cases:**
- Train risk factor classifiers
- Identify treatment-relevant patient characteristics
- Feature engineering for prediction models

---

### 📁 data/ (Activity Monitoring Study)
Clinical depression study with actigraphy data.

#### **scores.csv**
- 23 depression patients + 32 healthy controls
- MADRS scores (baseline and follow-up)
- Demographics and clinical features

#### **condition/**, **control/**
- Minute-by-minute activity monitoring data
- 23 CSV files (depression patients)
- 32 CSV files (healthy controls)

**Use Cases:**
- Build activity-based depression classifiers
- Extract behavioral biomarkers
- Supplement text-based models with behavioral data

---

### 📁 survey.csv
Workplace mental health survey (1,261 tech workers, 2014).

**Use Cases:**
- Analyze stigma and treatment-seeking patterns
- Identify barriers to mental health care
- Contextual background on mental health attitudes

---

## Machine Learning Tasks

### Primary Task: Treatment Response Prediction
**Goal:** Predict treatment_response (responder/partial/non-responder) given baseline data

**Input Features:**
1. **Structured:** Demographics, baseline severity, treatment type
2. **Text:** Therapy notes, digital chats, Reddit posts
3. **Behavioral:** Engagement metrics, session attendance
4. **Optional:** Activity data from actigraphy study

**Target:** `treatment_response` in patient_profiles.csv

**Evaluation Metrics:**
- Multi-class classification accuracy
- Per-class F1 scores
- ROC-AUC for responder vs. non-responder
- Clinical utility metrics (specificity for non-responders)

---

### Secondary Tasks

1. **Symptom Severity Prediction**
   - Predict outcome_phq9 / outcome_gad7 scores
   - Regression task with baseline + text features

2. **Treatment Recommendation**
   - Given patient features, recommend optimal treatment type
   - Counterfactual reasoning across treatment arms

3. **Early Detection**
   - Predict final outcome from early sessions (weeks 1-4)
   - Enable early intervention adjustments

4. **Engagement Prediction**
   - Predict session_attendance_rate or digital_engagement_score
   - Identify patients at risk of dropout

5. **Language-Based Risk Assessment**
   - Classify posts by risk factor (using Labelled Data)
   - Extract linguistic markers of severity

---

## Recommended Modeling Approaches

### 1. Baseline Models
- Logistic Regression on structured features only
- Random Forest with basic demographic + severity features
- Establishes performance floor

### 2. LLM-Enhanced Models
- **BERT/RoBERTa:** Fine-tune on therapy notes + Reddit posts
- **Clinical BERT variants:** MentalBERT, PsychBERT
- Extract embeddings as features for prediction

### 3. Multimodal Fusion
- Combine structured data + text embeddings + behavioral signals
- Late fusion: Separate models → ensemble
- Early fusion: Concatenate features → single classifier

### 4. Sequence Models
- LSTM/GRU on longitudinal therapy notes
- Capture temporal dynamics of treatment response
- Model trajectory of improvement

### 5. Transformer-Based
- GPT-style models for text generation + prediction
- Attention mechanisms to identify key predictive phrases
- Explainable predictions via attention weights

---

## Data Integration Guide

### Linking Datasets

**Patient-Level Integration:**
```python
# Core patient data
patients = pd.read_csv('processed/patient_profiles.csv')

# Add therapy notes
notes = pd.read_csv('processed/therapy_notes.csv')
patient_notes = notes.groupby('patient_id').agg({
    'therapist_notes': lambda x: ' '.join(x),
    'engagement_level': 'mean'
})

# Add Reddit posts
posts = pd.read_csv('processed/patient_reddit_posts.csv')
patient_posts = posts.groupby('patient_id').agg({
    'post_text': lambda x: ' '.join(x),
    'post_sentiment': 'mean',
    'emotional_intensity': 'mean'
})

# Merge all
full_data = patients.merge(patient_notes, on='patient_id', how='left') \
                    .merge(patient_posts, on='patient_id', how='left')
```

**Text Feature Extraction:**
```python
# Combine all text sources per patient
full_data['all_text'] = (
    full_data['therapist_notes'].fillna('') + ' ' +
    full_data['post_text'].fillna('')
)

# Example: Extract embeddings using sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = model.encode(full_data['all_text'].tolist())
```

---

## Ethical Considerations

### Synthetic Data Notice
- Therapy notes, chats, and patient-Reddit links are **synthetically generated**
- Reddit posts are **real** but de-identified
- Activity data is from published research study
- Use synthetic data for development; validate on real clinical data before deployment

### Privacy
- No real patient identifiers in synthetic data
- Reddit data is publicly posted content (anonymized)
- Follow IRB protocols if using real clinical data

### Clinical Use
- This is a **research/development dataset** only
- Not for direct clinical decision-making without validation
- Clinician-in-the-loop essential for any deployed system
- Predictions should support, not replace, clinical judgment

### Bias Considerations
- Limited demographic diversity in synthetic data
- English-only text
- May not generalize across cultures or languages
- Systematic evaluation needed for fairness across subgroups

---

## Citation & Acknowledgments

**Original Reddit Data Sources:**
- r/Anxiety, r/depression, r/lonely, r/SuicideWatch, r/mentalhealth communities
- Publicly available posts from 2019-2022

**Actigraphy Study:**
- Published clinical depression research (condition/control design)
- MADRS assessment protocol

**Synthetic Components:**
- Patient profiles, therapy notes, and digital chats generated for research purposes
- Designed to match clinical presentation patterns from literature

---

## Getting Started

### Quick Start
```python
import pandas as pd

# Load main dataset
patients = pd.read_csv('Dataset/processed/patient_profiles.csv')

# Check target variable distribution
print(patients['treatment_response'].value_counts())

# Basic features for first model
features = ['age', 'baseline_phq9', 'baseline_gad7', 
            'session_attendance_rate', 'treatment_duration_weeks']
X = patients[features]
y = patients['treatment_response']
```

### Training Notebook
See `treatment_response_prediction.ipynb` for complete training pipeline including:
- Data loading and preprocessing
- Text feature extraction with LLMs
- Model training (baseline → advanced)
- Evaluation and interpretation
- Google Colab compatible

---

## Support & Contact
For questions about dataset usage or to report issues, please open an issue in the repository.

**Last Updated:** February 2026
**Dataset Version:** 1.0
