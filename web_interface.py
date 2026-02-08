"""
Simple Web Interface
HTML-based web interface for the Mental Health Treatment Response Prediction System
"""

from flask import Flask, render_template_string, request, jsonify
from src.model_manager import ModelManager
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
model_manager = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Treatment Response Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        header p {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 12px;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideDown 0.3s ease-out;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 20px;
        }
        
        .prediction-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .prediction-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .prediction-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        
        .prediction-value.responder {
            color: #10b981;
        }
        
        .prediction-value.partial {
            color: #f59e0b;
        }
        
        .prediction-value.non-responder {
            color: #ef4444;
        }
        
        .confidence-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease-out;
        }
        
        .probabilities {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        
        .prob-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .prob-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .prob-value {
            font-size: 18px;
            font-weight: bold;
        }
        
        .recommendation {
            background: #e0f2fe;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #0ea5e9;
        }
        
        .recommendation p {
            color: #1e40af;
            line-height: 1.6;
        }
        
        .error {
            background: #fee2e2;
            color: #991b1b;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Treatment Response Predictor</h1>
            <p>AI-Driven Decision Support for Youth Mental Health</p>
        </header>
        
        <div class="content">
            <form id="predictionForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="patient_id">Patient ID *</label>
                        <input type="text" id="patient_id" name="patient_id" required value="P_TEST_001">
                    </div>
                    
                    <div class="form-group">
                        <label for="age">Age (13-22) *</label>
                        <input type="number" id="age" name="age" min="13" max="22" required value="17">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="gender">Gender *</label>
                        <select id="gender" name="gender" required>
                            <option value="F" selected>Female</option>
                            <option value="M">Male</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="baseline_severity">Baseline Severity *</label>
                        <select id="baseline_severity" name="baseline_severity" required>
                            <option value="mild-moderate">Mild-Moderate</option>
                            <option value="moderate">Moderate</option>
                            <option value="moderate-severe" selected>Moderate-Severe</option>
                            <option value="severe">Severe</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="baseline_phq9">PHQ-9 Score (0-27) *</label>
                        <input type="number" id="baseline_phq9" name="baseline_phq9" min="0" max="27" required value="18">
                    </div>
                    
                    <div class="form-group">
                        <label for="baseline_gad7">GAD-7 Score (0-21) *</label>
                        <input type="number" id="baseline_gad7" name="baseline_gad7" min="0" max="21" required value="15">
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="treatment_type">Treatment Type *</label>
                    <select id="treatment_type" name="treatment_type" required>
                        <option value="CBT">CBT</option>
                        <option value="Medication">Medication</option>
                        <option value="Digital_Therapy">Digital Therapy</option>
                        <option value="CBT+Medication">CBT + Medication</option>
                        <option value="CBT+Digital" selected>CBT + Digital</option>
                    </select>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="treatment_duration_weeks">Duration (weeks) *</label>
                        <input type="number" id="treatment_duration_weeks" name="treatment_duration_weeks" min="6" max="18" required value="12">
                    </div>
                    
                    <div class="form-group">
                        <label for="session_attendance_rate">Attendance Rate (0.0-1.0) *</label>
                        <input type="number" id="session_attendance_rate" name="session_attendance_rate" min="0" max="1" step="0.01" required value="0.85">
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="digital_engagement_score">Digital Engagement Score (0.0-1.0)</label>
                    <input type="number" id="digital_engagement_score" name="digital_engagement_score" min="0" max="1" step="0.01" value="0.7">
                </div>
                
                <div class="form-group">
                    <label for="therapy_notes">Therapy Notes (Optional)</label>
                    <textarea id="therapy_notes" name="therapy_notes" placeholder="Enter therapy session notes..."></textarea>
                </div>
                
                <button type="submit">Predict Treatment Response</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px;">Analyzing patient data...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <h3>📊 Prediction Results</h3>
                
                <div class="prediction-box">
                    <div class="prediction-label">Predicted Treatment Response</div>
                    <div class="prediction-value" id="predictedResponse">-</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                    </div>
                    <div style="margin-top: 5px; font-size: 14px; color: #666;">
                        Confidence: <span id="confidence">-</span>
                    </div>
                </div>
                
                <div class="probabilities">
                    <div class="prob-item">
                        <div class="prob-label">Non-Responder</div>
                        <div class="prob-value" id="probNonResponder">-</div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-label">Partial</div>
                        <div class="prob-value" id="probPartial">-</div>
                    </div>
                    <div class="prob-item">
                        <div class="prob-label">Responder</div>
                        <div class="prob-value" id="probResponder">-</div>
                    </div>
                </div>
                
                <div class="recommendation">
                    <strong>Clinical Recommendation:</strong>
                    <p id="recommendation">-</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            // Convert numeric fields
            data.age = parseInt(data.age);
            data.baseline_phq9 = parseInt(data.baseline_phq9);
            data.baseline_gad7 = parseInt(data.baseline_gad7);
            data.treatment_duration_weeks = parseInt(data.treatment_duration_weeks);
            data.session_attendance_rate = parseFloat(data.session_attendance_rate);
            data.digital_engagement_score = parseFloat(data.digital_engagement_score || 0);
            
            // Show loading, hide results and errors
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            document.getElementById('error').classList.remove('show');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Display results
                    const responseEl = document.getElementById('predictedResponse');
                    responseEl.textContent = result.predicted_response.toUpperCase();
                    responseEl.className = 'prediction-value ' + result.predicted_response;
                    
                    document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + '%';
                    document.getElementById('confidenceFill').style.width = (result.confidence * 100) + '%';
                    
                    document.getElementById('probNonResponder').textContent = (result.probabilities['non-responder'] * 100).toFixed(1) + '%';
                    document.getElementById('probPartial').textContent = (result.probabilities.partial * 100).toFixed(1) + '%';
                    document.getElementById('probResponder').textContent = (result.probabilities.responder * 100).toFixed(1) + '%';
                    
                    document.getElementById('recommendation').textContent = result.recommendation;
                    
                    document.getElementById('result').classList.add('show');
                } else {
                    throw new Error(result.error || 'Prediction failed');
                }
            } catch (error) {
                document.getElementById('error').textContent = 'Error: ' + error.message;
                document.getElementById('error').classList.add('show');
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global model_manager
    
    try:
        data = request.get_json()
        
        # Initialize model manager if not already done
        if model_manager is None:
            logger.info("Initializing model manager...")
            model_manager = ModelManager()
        
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
            'outcome_phq9': 0,
            'outcome_gad7': 0
        }
        
        # Make prediction
        result = model_manager.predict_single_patient(
            patient_info,
            therapy_notes=data.get('therapy_notes', ''),
            digital_chats=data.get('digital_chats', ''),
            reddit_posts=data.get('reddit_posts', '')
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def run_web_interface():
    """Run the web interface"""
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 5000)
    
    print("\n" + "="*70)
    print("  Mental Health Treatment Response Prediction System")
    print("  Web Interface")
    print("="*70)
    print(f"\n🌐 Starting web server at http://{host}:{port}")
    print(f"📱 Open your browser and navigate to: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    run_web_interface()
