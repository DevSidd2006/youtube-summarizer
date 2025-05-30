# 🚀 Streamlit Deployment Guide

## Quick Deploy to Streamlit Community Cloud

### Prerequisites
- GitHub repository (public or private)
- Streamlit Community Cloud account

### Deployment Steps

1. **Push to GitHub**: Ensure all your code is committed and pushed
2. **Connect Repository**: Go to [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Select your repository and main branch
4. **Entry Point**: Use `src/app.py` as your main file
5. **Deploy**: Click "Deploy"

### Important Files for Deployment

#### ✅ Required Files (Kept in Repository)
- `src/app.py` - Main application entry point
- `src/app_fast.py` - Fast processing variant  
- `src/app_ultra_fast.py` - Ultra-fast variant
- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies (ffmpeg, git)
- `.streamlit/config.toml` - Streamlit configuration
- `README.md` - Project documentation

#### ❌ Hidden Files (Ignored by Git)
- Development tools (`.vscode/`, `.idea/`)
- Virtual environments (`venv/`, `env/`)
- Cache files (`__pycache__/`, `*.pyc`)
- Model cache (`models/`, `*.bin`, `*.safetensors`)
- Temporary files (`*.tmp`, `temp/`)
- Debug scripts (`debug_*.py`, `experiment_*.py`)
- Large media files (`*.mp4`, `*.mp3`, `*.wav`)
- Secrets and API keys (`.env`, `secrets.py`)

### Deployment URL Structure
Your app will be available at:
```
https://[your-app-name]-[random-string].streamlit.app/
```

### Performance Modes Available
- **Standard**: `src/app.py` (Full features)
- **Fast**: `src/app_fast.py` (3-5x faster)
- **Ultra-Fast**: `src/app_ultra_fast.py` (Maximum speed)

### Troubleshooting

#### Common Issues:
1. **Build fails**: Check `requirements.txt` for version conflicts
2. **Missing dependencies**: Ensure `packages.txt` includes system packages
3. **Import errors**: Verify all required packages are in `requirements.txt`
4. **Memory issues**: Use Fast or Ultra-Fast modes for better performance

#### Resource Limits:
- **Memory**: 1GB RAM limit on free tier
- **CPU**: Shared resources
- **Storage**: Limited to repository size
- **Network**: Good for most use cases

### Local Testing Before Deployment

```bash
# Test the exact deployment setup
pip install -r requirements.txt
streamlit run src/app.py

# Test different performance modes
streamlit run src/app_fast.py
streamlit run src/app_ultra_fast.py
```

### Repository Structure for Deployment
```
youtube-summarizer/
├── src/
│   ├── app.py              # 🎯 Main entry point
│   ├── app_fast.py         # ⚡ Fast variant
│   └── app_ultra_fast.py   # 🚀 Ultra-fast variant
├── requirements.txt        # 📦 Python dependencies
├── packages.txt           # 🔧 System dependencies
├── .streamlit/
│   └── config.toml        # ⚙️ Streamlit config
└── README.md              # 📖 Documentation
```

All development files, caches, and unnecessary artifacts are automatically hidden via `.gitignore`.
