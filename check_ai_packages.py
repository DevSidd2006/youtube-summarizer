#!/usr/bin/env python3
"""Check AI transcription package installation"""

import sys

def check_package(package_name):
    try:
        __import__(package_name)
        print(f"✅ {package_name} - INSTALLED")
        return True
    except ImportError:
        print(f"❌ {package_name} - MISSING")
        return False

print("🔍 Checking AI Transcription Dependencies...")
print("=" * 50)

packages = [
    "transformers",
    "torch", 
    "pytube",
    "pydub",
    "speech_recognition",
    "faster_whisper"
]

installed = 0
for package in packages:
    if check_package(package):
        installed += 1

print("=" * 50)
print(f"📊 Status: {installed}/{len(packages)} packages installed")

if installed == len(packages):
    print("🎉 All AI transcription packages are ready!")
    print("💡 Restart the Streamlit app to enable AI transcription")
else:
    print("⚠️ Some packages are missing. Install with:")
    print("pip install transformers torch pytube pydub SpeechRecognition faster-whisper")
