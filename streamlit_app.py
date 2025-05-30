#!/usr/bin/env python3
"""
Streamlit Community Cloud Entry Point for YouTube Summarizer
This file serves as the main entry point for Streamlit Cloud deployment.
"""

import sys
import os

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import and run the main application
try:
    # Import the main app module
    # The app.py file contains Streamlit code that runs at module level
    # When imported, it will automatically execute the Streamlit app
    import app
    
    # Note: The app.py contains code like:
    # st.set_page_config(...)
    # st.title(...)
    # etc.
    # These run automatically when the module is imported
    
except ImportError as e:
    import streamlit as st
    st.error(f"❌ Failed to import main application: {e}")
    st.error("Please ensure all dependencies are installed correctly.")
    st.info("💡 Try running: `pip install -r requirements.txt`")
    st.markdown("---")
    st.subheader("🔧 Troubleshooting")
    st.markdown("""
    **Common solutions:**
    1. Check that all packages in `requirements.txt` are installed
    2. Ensure Python version is 3.8+ (Python 3.13 supported)
    3. Verify that the `src` directory exists with `app.py`
    4. Try: `pip install --upgrade -r requirements.txt`
    """)
except Exception as e:
    import streamlit as st
    st.error(f"❌ Application startup error: {e}")
    st.error("Please check the application logs for more details.")
    st.markdown("---")
    st.subheader("🔧 Debug Information")
    st.code(f"Error type: {type(e).__name__}")
    st.code(f"Error details: {str(e)}")
    st.code(f"Python path: {sys.path}")
    st.code(f"Current directory: {os.getcwd()}")
    st.code(f"Source directory: {src_dir}")
    st.code(f"Source directory exists: {os.path.exists(src_dir)}")
    if os.path.exists(src_dir):
        st.code(f"Files in src: {os.listdir(src_dir)}")
