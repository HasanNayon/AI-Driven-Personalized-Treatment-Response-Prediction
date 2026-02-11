import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "-"*50)
    print(f"  {text}")
    print("-"*50 + "\n")

def run_command(cmd, description, check=True):
    print(f"  {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        if result.returncode == 0:
            print(f"  {description} - OK\n")
            return True
        else:
            print(f"  {description} - FAILED")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}\n")
            return False
    except Exception as e:
        print(f"  {description} - FAILED")
        print(f"Error: {str(e)[:200]}\n")
        return False

def check_python_version():
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("  Python version OK (3.8+)\n")
        return True
    else:
        print("  Python 3.8+ required\n")
        return False

def check_files():
    print_header("Checking Required Files")
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'app.py',
        'examples.py',
        'src/config.py',
        'src/model_manager.py',
        'Dataset/processed/patient_profiles.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  [ok] {file_path}")
        else:
            print(f"  [missing] {file_path}")
            all_exist = False
    
    print()
    return all_exist

def install_dependencies():
    print_header("Installing Dependencies")
    
    print("This may take 5-15 minutes depending on your internet connection...\n")
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies",
        check=False
    )

def create_directories():
    print_header("Creating Directories")
    
    directories = ['logs', 'outputs', 'outputs/examples', 'outputs/visualizations']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        print(f"  [ok] {directory}")
    
    print()
    return True

def copy_env_file():
    print_header("Configuring Environment")
    
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_file.exists():
        print("  .env file already exists")
        return True
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("  Created .env from .env.example\n")
        return True
    else:
        print("  .env.example not found")
        return False

def test_imports():
    print_header("Testing Package Imports")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('xgboost', 'XGBoost'),
        ('flask', 'Flask'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn')
    ]
    
    all_imported = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  [ok] {name}")
        except ImportError:
            print(f"  [fail] {name}")
            all_imported = False
    
    print()
    return all_imported

def run_simple_test():
    print_header("Running Functionality Test")
    
    print("Testing data loading and model initialization...\n")
    
    test_code = """
from src.data_loader import DataLoader
from src.config import config

print("→ Loading configuration...")
print(f"  Model path: {config.get('models.llm_path')}")

print("→ Initializing data loader...")
loader = DataLoader()

print("→ Loading patient profiles...")
df = loader.load_patient_profiles()
print(f"  Loaded {len(df)} patients")

print("\\n✓ Basic functionality test passed!")
"""
    
    return run_command(
        f"{sys.executable} -c \"{test_code}\"",
        "Testing basic functionality",
        check=False
    )

def show_next_steps():
    print_header("Setup Complete")
    
    print("Next steps:\n")
    print("  python examples.py              # run examples")
    print("  python app.py predict-db --patient-id P0001  # predict from db")
    print("  python app.py api               # start API server")
    print("  python web_interface.py          # start web UI")
    print()

def main():
    print("\n" + "-"*50)
    print("  Setup Verification")
    print("-"*50 + "\n")
    
    # Check Python version
    if not check_python_version():
        print("Please upgrade Python to 3.8 or higher and try again.")
        return False
    
    # Check required files
    if not check_files():
        print("Some required files are missing. Please check your installation.")
        return False
    
    # Create directories
    create_directories()
    
    # Copy environment file
    copy_env_file()
    
    # Ask user about installation
    print("\nDo you want to install/update dependencies? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        if not install_dependencies():
            print("\n  Dependency installation had issues.")
            print("  Try manually: pip install -r requirements.txt\n")
    
    # Test imports
    if not test_imports():
        print("\n  Some packages failed to import.")
        print("  Try: pip install -r requirements.txt\n")
        return False
    
    # Run functionality test
    if not run_simple_test():
        print("\n  Functionality test had issues. Check logs.\n")
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
