#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Preparation Script
Automatically configures the repository for successful cloud deployment
"""

import shutil
import os

def prepare_for_cloud_deployment():
    """Prepare repository for Streamlit Cloud deployment"""
    
    print("🚀 Preparing YouTube Summarizer for Streamlit Cloud Deployment")
    print("=" * 60)
    
    # Step 1: Backup current requirements.txt
    if os.path.exists("requirements.txt"):
        shutil.copy("requirements.txt", "requirements_local_backup.txt")
        print("✅ Backed up local requirements.txt to requirements_local_backup.txt")
    
    # Step 2: Copy cloud requirements as main requirements.txt
    if os.path.exists("requirements_cloud.txt"):
        shutil.copy("requirements_cloud.txt", "requirements.txt")
        print("✅ Replaced requirements.txt with cloud-optimized version")
    else:
        print("❌ requirements_cloud.txt not found!")
        return False
    
    # Step 3: Verify streamlit_app.py has cloud detection
    try:
        with open("streamlit_app.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "IS_STREAMLIT_CLOUD" in content and "STREAMLIT_SHARING" in content:
                print("✅ streamlit_app.py has cloud detection enabled")
            else:
                print("⚠️ streamlit_app.py may not have proper cloud detection")
    except Exception as e:
        print(f"❌ Error checking streamlit_app.py: {e}")
        return False
    
    # Step 4: Create .streamlit/config.toml for cloud optimization
    os.makedirs(".streamlit", exist_ok=True)
    
    config_content = """[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
    
    with open(".streamlit/config.toml", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("✅ Created .streamlit/config.toml for cloud optimization")
    
    print("\n" + "=" * 60)
    print("🎯 DEPLOYMENT READY!")
    print("=" * 60)
    
    print("📋 Next Steps:")
    print("1. Commit changes: git add . && git commit -m 'Prepare for cloud deployment'")
    print("2. Push to GitHub: git push origin main")
    print("3. Go to https://share.streamlit.io")
    print("4. Deploy using:")
    print("   - Main file: streamlit_app.py")
    print("   - Requirements file: requirements.txt (now cloud-optimized)")
    print("   - Python version: 3.9 or 3.10")
    
    print("\n✨ Your app will:")
    print("   ✅ Deploy successfully without PyTorch errors")
    print("   ✅ Handle YouTube videos with subtitles")
    print("   ✅ Translate Hindi content")
    print("   ✅ Provide basic summarization")
    print("   ⚠️ Skip AI transcription (cloud limitation)")
    
    return True

def restore_local_development():
    """Restore full local development environment"""
    
    print("🏠 Restoring Local Development Environment")
    print("=" * 50)
    
    if os.path.exists("requirements_local_backup.txt"):
        shutil.copy("requirements_local_backup.txt", "requirements.txt")
        print("✅ Restored local requirements.txt from backup")
        
        print("\n📋 To restore full AI functionality:")
        print("pip install transformers torch pytube pydub SpeechRecognition faster-whisper")
        
        return True
    else:
        print("❌ No local backup found!")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_local_development()
    else:
        prepare_for_cloud_deployment()
        
        print(f"\n💡 To restore local development later, run:")
        print(f"   python {__file__} restore")
