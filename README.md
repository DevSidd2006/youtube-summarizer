# YouTube Video Summarizer

A powerful Streamlit application that summarizes YouTube videos using AI and provides translations in Hindi with Python 3.13 compatibility.

## 🚀 Features

- **AI-Powered Summaries**: Extract key insights from YouTube videos using advanced transformer models
- **Hindi Translation**: Automatic translation to Hindi with robust error handling
- **Python 3.13 Compatible**: Includes compatibility fixes for the latest Python version
- **Multiple Performance Modes**: Choose from standard, fast, and ultra-fast processing variants
- **User-Friendly Interface**: Clean Streamlit web interface with progress tracking
- **Robust Error Handling**: Comprehensive error management and fallback mechanisms

## 📋 Requirements

- Python 3.8+ (including Python 3.13)
- Internet connection for video download and translation
- Sufficient RAM for AI model processing (recommended: 4GB+)

## 🛠️ Installation

### Quick Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DevSidd2006/Link-Bypass.git
   cd Link-Bypass
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Start the application**:
   ```bash
   streamlit run src/app.py
   ```

### Manual Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DevSidd2006/Link-Bypass.git
   cd Link-Bypass
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests** (optional):
   ```bash
   python tests/test_app.py
   ```

4. **Run the application**:
   ```bash
   streamlit run src/app.py
   ```

## 🎯 Usage

1. **Launch the app**: Run `streamlit run src/app.py` in your terminal
2. **Enter YouTube URL**: Paste any YouTube video URL in the input field
3. **Choose processing mode**: Select from available performance options
4. **Get summary**: The app will download, process, and summarize the video
5. **Hindi translation**: Automatic translation will be provided with the summary

## 📁 Project Structure

```
youtube-summarizer/
├── .git/                   # Git repository
├── .gitignore             # Git ignore rules
├── README.md              # Project overview
├── setup.py               # Quick setup script
├── requirements.txt       # Dependencies
├── src/
│   ├── app.py             # Main application
│   ├── app_fast.py        # Fast processing variant
│   └── app_ultra_fast.py  # Ultra-fast variant
├── tests/
│   └── test_app.py        # Comprehensive test suite
└── docs/
    ├── README.md          # Detailed documentation
    └── ENHANCEMENTS.md    # Feature specifications
```

## 🔧 Technical Features

### Python 3.13 Compatibility
- **CGI Module Fix**: Includes compatibility layer for Python 3.13 where `cgi` module was removed
- **Session State Management**: Optimized translator initialization to prevent repeated warnings
- **Enhanced Error Handling**: Comprehensive error categorization with user-friendly messages

### Performance Variants
- **Standard Mode** (`app.py`): Full-featured processing with all optimizations
- **Fast Mode** (`app_fast.py`): 3-5x faster processing with DistilBART model
- **Ultra-Fast Mode** (`app_ultra_fast.py`): Maximum speed with minimal memory usage

### Translation Features
- **Primary**: Google Translate with timeout handling
- **Fallback**: Deep Translator for enhanced reliability
- **Error Recovery**: Automatic fallback switching and detailed error guidance

## 🧪 Testing

Run the comprehensive test suite:

```bash
python tests/test_app.py
```

The test suite verifies:
- Python 3.13 CGI compatibility
- Translation functionality
- Package imports and dependencies

## 🔍 Troubleshooting

### Common Issues

1. **"Translator initialization test failed" warnings**:
   - Fixed in this version with session state optimization
   - The translator is tested only once per session

2. **Python 3.13 compatibility errors**:
   - Includes automatic CGI module compatibility layer
   - No manual intervention required

3. **Translation failures**:
   - Automatic fallback to alternative translation service
   - Network connectivity and rate limit handling included

4. **Performance issues**:
   - Use `app_fast.py` or `app_ultra_fast.py` for better performance
   - Check system RAM availability (4GB+ recommended)

## 📚 Documentation

For detailed documentation, see:
- [docs/README.md](docs/README.md) - Complete user guide
- [docs/ENHANCEMENTS.md](docs/ENHANCEMENTS.md) - Feature specifications and optimization details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python tests/test_app.py`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Transformers**: Hugging Face transformers library for AI processing
- **Streamlit**: Modern web app framework
- **yt-dlp**: YouTube video download capabilities
- **GoogleTrans**: Translation services
