#!/usr/bin/env python3
"""
Quick setup script for YouTube Summarizer
Installs dependencies and runs basic tests
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """Main setup process"""
    print("🚀 YouTube Summarizer Setup")
    print("=" * 40)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️ Dependency installation failed. Please check your pip installation.")
        return False
    
    # Run tests
    if not run_command("python tests/test_app.py", "Running tests"):
        print("⚠️ Some tests failed. The app might still work, but check the output above.")
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    print("\nTo start the application, run:")
    print("  streamlit run src/app.py")
    print("\nFor faster processing, try:")
    print("  streamlit run src/app_fast.py")
    print("  streamlit run src/app_ultra_fast.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
