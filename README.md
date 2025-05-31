# 🎬 YouTube Summarizer

A powerful Streamlit application that automatically summarizes YouTube videos with AI-powered transcription and intelligent multilingual support.

## ✨ Features

- **🎯 Smart Video Processing**: Automatically extracts and processes YouTube video content
- **🌍 Multilingual Support**: Intelligent handling of English and Hindi content
- **🤖 AI-Powered Transcription**: Advanced speech recognition for accurate transcripts
- **📊 Intelligent Summarization**: Generates concise, meaningful summaries
- **🚀 Optimized Performance**: Smart language detection prevents unnecessary processing
- **🎨 Modern Interface**: Clean, intuitive Streamlit web application

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (including Python 3.13)
- Internet connection for YouTube access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/youtube-summarizer.git
   cd youtube-summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   
   **Option A: Use the launcher (Windows)**
   ```bash
   start_app.bat
   ```
   
   **Option B: Direct command**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Paste any YouTube URL and start summarizing!

## 🔧 How It Works

1. **📥 Video Input**: Paste any YouTube URL
2. **🔍 Content Analysis**: Extracts video metadata and available transcripts
3. **🧠 Language Detection**: Automatically detects content language (English/Hindi)
4. **📝 Transcription**: Uses AI transcription for videos without existing subtitles
5. **⚡ Smart Processing**: Applies optimized processing based on detected language
6. **📊 Summarization**: Generates structured, concise summaries

## 📁 Project Structure

```
youtube-summarizer/
├── streamlit_app.py      # Main application
├── requirements.txt      # Python dependencies
├── start_app.bat        # Windows launcher
├── README.md            # Project documentation
├── .gitignore           # Git ignore rules
└── docs/                # Additional documentation
```

## 🌟 Key Features

### Intelligent Language Processing
- **English Content**: Direct processing without unnecessary translation overhead
- **Hindi Content**: Full translation and processing pipeline when needed
- **Smart Detection**: Automatic language identification prevents redundant operations

### Performance Optimizations
- **Early Language Detection**: Skips unnecessary processing steps
- **Efficient Transcript Extraction**: Multiple fallback methods for reliability
- **Optimized AI Usage**: Smart model selection based on content type

## 🛠️ Configuration

The application works out-of-the-box with sensible defaults. For customization:

- **Port Configuration**: Modify port in `start_app.bat` or use `--server.port` flag
- **Processing Settings**: Configure AI and translation settings through the web interface

## 🔍 Troubleshooting

### Common Solutions

**Installation Issues**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Port Already in Use**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Python Compatibility**
- Supports Python 3.8+ including Python 3.13
- Use virtual environment for clean dependency management

## 📚 Documentation

For detailed information:
- [docs/README.md](docs/README.md) - Complete user guide
- [docs/ENHANCEMENTS.md](docs/ENHANCEMENTS.md) - Technical specifications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** - Modern web app framework
- **Hugging Face Transformers** - AI processing capabilities
- **yt-dlp** - YouTube video processing
- **OpenAI Whisper** - Speech recognition technology

---

**⭐ If you find this project helpful, please give it a star on GitHub!**

**Made with ❤️ for the open source community**
