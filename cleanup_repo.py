#!/usr/bin/env python3
"""
Repository Cleanup Script
Removes unnecessary development/test files for a clean GitHub repository
"""

import os
import shutil
from pathlib import Path

def cleanup_repository():
    """Remove unnecessary files and folders for clean GitHub repo"""
    
    # Current directory
    project_root = Path(__file__).parent
    
    # Files to remove (development artifacts)
    files_to_remove = [
        # Test files
        'test_*.py',
        '*_test.py',
        'verify_*.py',
        'benchmark_*.py',
        'fix_*.py',
        'launch_*.py',
        
        # Development docs
        '*_SUMMARY.md',
        '*_COMPLETED*.md',
        '*_FINAL*.md',
        'FIX_*.md',
        'CLEANUP_*.md',
        'DEPLOYMENT.md',
        'GITHUB_STRUCTURE.md',
        'STREAMLIT_READY.md',
        'README_FIXED.md',
        'FINAL_COMPLETION_REPORT.md',
        
        # Alternative versions
        'streamlit_app1.py',
    ]
    
    # Folders to remove
    folders_to_remove = [
        '__pycache__',
        'src',  # Contains alternative app versions
        'tests',  # Development test folder
    ]
    
    print("🧹 Cleaning up repository for GitHub...")
    print("=" * 50)
    
    removed_count = 0
    
    # Remove files
    for pattern in files_to_remove:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                print(f"🗑️  Removing file: {file_path.name}")
                file_path.unlink()
                removed_count += 1
    
    # Remove folders
    for folder_name in folders_to_remove:
        folder_path = project_root / folder_name
        if folder_path.exists() and folder_path.is_dir():
            print(f"📁 Removing folder: {folder_name}/")
            shutil.rmtree(folder_path)
            removed_count += 1
    
    # Replace README with clean version
    clean_readme = project_root / "README_CLEAN.md"
    main_readme = project_root / "README.md"
    
    if clean_readme.exists():
        print(f"📝 Replacing README.md with clean version")
        shutil.move(str(clean_readme), str(main_readme))
        removed_count += 1
    
    print("=" * 50)
    print(f"✅ Cleanup complete! Removed {removed_count} items")
    print("\n🎯 Repository is now clean and GitHub-ready!")
    print("\n📁 Essential files remaining:")
    
    # Show remaining essential files
    essential_patterns = [
        "*.py",
        "*.md", 
        "*.txt",
        "*.bat",
        ".gitignore"
    ]
    
    for pattern in essential_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                print(f"   ✓ {file_path.name}")

if __name__ == "__main__":
    cleanup_repository()
