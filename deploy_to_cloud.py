#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Helper
Prepares the repository for cloud deployment by switching requirements files
"""

import os
import shutil
from pathlib import Path

def prepare_for_cloud_deployment():
    """Prepare repository for Streamlit Cloud deployment"""
    print("🚀 Preparing for Streamlit Cloud deployment...")
    
    # Backup local requirements
    if os.path.exists("requirements.txt"):
        print("📁 Backing up local requirements.txt → requirements_local.txt")
        shutil.copy("requirements.txt", "requirements_local.txt")
    
    # Use cloud requirements as main requirements
    if os.path.exists("requirements_cloud.txt"):
        print("☁️ Switching to cloud requirements: requirements_cloud.txt → requirements.txt")
        shutil.copy("requirements_cloud.txt", "requirements.txt")
        print("✅ Cloud requirements activated!")
    else:
        print("❌ Error: requirements_cloud.txt not found!")
        return False
    
    print("\n📋 Current requirements.txt content:")
    with open("requirements.txt", "r") as f:
        content = f.read()
        print(content[:500] + "..." if len(content) > 500 else content)
    
    print("\n🎯 Ready for deployment!")
    print("📝 Next steps:")
    print("   1. Commit and push to GitHub")
    print("   2. Deploy on Streamlit Cloud")
    print("   3. Streamlit will automatically use requirements.txt")
    
    return True

def restore_local_development():
    """Restore local development requirements"""
    print("🏠 Restoring local development setup...")
    
    if os.path.exists("requirements_local.txt"):
        print("📁 Restoring local requirements: requirements_local.txt → requirements.txt")
        shutil.copy("requirements_local.txt", "requirements.txt")
        print("✅ Local development requirements restored!")
    else:
        print("⚠️ Warning: requirements_local.txt not found!")
    
    print("🔧 Local development ready!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_local_development()
    else:
        prepare_for_cloud_deployment()
