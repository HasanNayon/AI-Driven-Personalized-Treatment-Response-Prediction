# Setup Guide

## Mental Health Treatment Response Prediction System

Complete setup instructions for Windows, macOS, and Linux.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [GPU Setup (Optional)](#gpu-setup-optional)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **CPU**: 4+ cores recommended

### Recommended Requirements
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 6GB+ VRAM (for faster LLM inference)
- **CUDA**: 11.8 or higher (if using GPU)

---

## Installation

### Step 1: Clone or Navigate to Project

```bash
# Navigate to project directory
cd "d:/mental health"
```

### Step 2: Create Virtual Environment (Recommended)

#### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This will install:
- PyTorch (CPU version by default)
- Transformers
- XGBoost
- Flask and related packages
- Data science libraries (pandas, numpy, scikit-learn)
- Visualization libraries (matplotlib, seaborn, plotly)

**Installation time**: 5-15 minutes depending on internet speed

---

## Configuration

### Step 1: Create Environment File

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

### Step 2: Edit Configuration (Optional)

Open `.env` file and modify if needed:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Model Configuration
MODEL_DEVICE=cpu  # Change to 'cuda' if you have GPU

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000

# Logging
LOG_LEVEL=INFO
```

### Step 3: Review config.yaml

The `config.yaml` file contains all system configurations. Default settings work out of the box, but you can customize:

- Model paths
- Data paths
- LLM parameters (max_length, batch_size)
- XGBoost parameters
- Feature engineering options

---

## Verification

### Step 1: Check Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check pip packages
pip list
```

### Step 2: Run Test Examples

```bash
# Run example 1 (single patient prediction)
python examples.py 1
```

Expected output:
```
==============================================================
EXAMPLE 1: Single Patient Prediction
==============================================================

Patient ID: P_NEW_001
Predicted Response: responder
Confidence: XX.X%
...
```

### Step 3: Start API Server

```bash
python app.py api
```

Expected output:
```
INFO | Starting API server on 0.0.0.0:5000
INFO | API documentation available at http://0.0.0.0:5000/apidocs
 * Running on http://0.0.0.0:5000
```

Open browser and navigate to:
- API: http://localhost:5000
- Documentation: http://localhost:5000/apidocs

Press `Ctrl+C` to stop the server.

### Step 4: Test Command Line Interface

```bash
# Test prediction for existing patient
python app.py predict-db --patient-id P0001
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'XXX'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Model Files Not Found

**Problem**: `FileNotFoundError: Model file not found`

**Solution**:
1. Check that model files exist:
   - `LLM Fine tuning model/` directory
   - `LLM fine Tuniing xgboost model/xgboost_llm_enhanced.pkl`

2. Verify paths in `config.yaml`:
```yaml
models:
  llm_path: "LLM Fine tuning model"
  xgboost_path: "LLM fine Tuniing xgboost model/xgboost_llm_enhanced.pkl"
```

### Issue: CUDA/GPU Errors

**Problem**: `RuntimeError: CUDA out of memory` or GPU not detected

**Solution**:
1. Change device to CPU in `.env`:
```env
MODEL_DEVICE=cpu
```

2. Or in `config.yaml`:
```yaml
llm:
  device: "cpu"
```

3. Reduce batch size in `config.yaml`:
```yaml
llm:
  batch_size: 4  # Reduce from 8
```

### Issue: Port Already in Use

**Problem**: `Address already in use` when starting API

**Solution**:
1. Change port in `.env`:
```env
API_PORT=5001  # Use different port
```

2. Or kill process using the port:

**Windows**:
```powershell
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**macOS/Linux**:
```bash
lsof -ti:5000 | xargs kill -9
```

### Issue: Slow Predictions

**Problem**: Predictions are very slow

**Solutions**:
1. Use GPU if available (see GPU Setup section)
2. Reduce max_length in `config.yaml`:
```yaml
llm:
  max_length: 256  # Reduce from 512
```
3. Disable LLM features for faster predictions (clinical features only)

### Issue: Import Errors

**Problem**: `ImportError` or `AttributeError` when importing modules

**Solution**:
```bash
# Ensure you're in the project root directory
cd "d:/mental health"

# Check Python path
python -c "import sys; print(sys.path)"

# Run with explicit path
python -m app predict-db --patient-id P0001
```

---

## GPU Setup (Optional)

For significantly faster LLM inference, configure GPU support.

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Drivers** installed
3. **CUDA Toolkit** 11.8 or higher

### Installation

#### Step 1: Uninstall CPU PyTorch

```bash
pip uninstall torch torchvision torchaudio
```

#### Step 2: Install GPU PyTorch

**For CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
```

#### Step 4: Configure for GPU

Edit `.env`:
```env
MODEL_DEVICE=cuda
```

Or `config.yaml`:
```yaml
llm:
  device: "cuda"
```

#### Step 5: Test GPU Inference

```bash
python examples.py 1
```

Check logs for:
```
INFO | Initializing LLM on device: cuda
```

---

## Next Steps

After successful setup:

1. **Explore Examples**: Run all examples
   ```bash
   python examples.py
   ```

2. **Read Documentation**: 
   - Main README: `README.md`
   - API Docs: `API_DOCUMENTATION.md`
   - Dataset Info: `Dataset/README.md`

3. **Start Using**:
   - CLI: `python app.py predict --help`
   - API: `python app.py api`
   - Python: See examples in `examples.py`

4. **Customize**:
   - Modify `config.yaml` for your needs
   - Explore visualization outputs in `outputs/`
   - Review logs in `logs/`

---

## Additional Resources

### Documentation
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Support

For issues or questions:
1. Check this setup guide
2. Review troubleshooting section
3. Check logs in `logs/app.log`
4. Open an issue in the repository

---

## Uninstallation

To remove the system:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment directory
# Windows
rmdir /s venv

# macOS/Linux
rm -rf venv
```

To keep the code but remove large model files:
```bash
# Remove model checkpoints (keep only necessary files)
rmdir /s "checkpoint-1500"
```

---

**Last Updated**: February 2026  
**Version**: 1.0.0
