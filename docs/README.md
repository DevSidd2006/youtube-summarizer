# 🚀 Advanced YouTube Video Analyzer & Summarizer

A powerful AI-powered tool that extracts transcripts from YouTube videos and provides comprehensive analysis including summaries, sentiment analysis, key insights, and automatic translation capabilities with special support for Hindi content.

## 🆕 Latest Updates (v2.1)

### 🛡️ Enhanced Error Handling
- **Robust Video Access**: Better handling of restricted, age-gated, and private videos
- **HTTP Error Recovery**: Detailed explanations for 400/403 errors with solution guidance
- **XML Parsing Fixes**: Improved handling of malformed YouTube API responses
- **Multiple Fallback Methods**: Alternative transcript extraction when primary methods fail

### 🇮🇳 Improved Hindi Processing
- **Enhanced Translation**: Fixed Google Translate API coroutine object issues
- **Better Error Recovery**: Multiple translation attempts with different language specifications
- **Encoding Improvements**: Robust UTF-8 handling for Hindi text processing
- **User-Friendly Progress**: Clear status updates during Hindi translation process

### 🤖 AI Transcription Enhancements
- **Smarter Fallbacks**: Automatic AI transcription when subtitles are restricted
- **Better Audio Handling**: Enhanced audio download with detailed error explanations
- **Progress Tracking**: Real-time status updates during AI transcription process

## ✨ Features

### 🎯 Core Functionality
- **Smart Transcript Extraction**: Multiple fallback methods including official subtitles, auto-generated captions, and AI transcription
- **Comprehensive Summarization**: Overview, detailed timeline, and chapter-based summaries
- **Advanced Analytics**: Sentiment analysis, key phrase extraction, and content categorization
- **Interactive Visualizations**: Word clouds, sentiment timelines, and topic analysis charts

### 🌍 Translation & Language Support
- **Auto-Translation**: Automatically detect and translate non-English videos to English
- **Hindi Auto-Detection**: Special support for Hindi content with enhanced translation
- **Multi-Language Support**: Support for 12+ languages including Spanish, French, German, Japanese, Korean, Chinese, Arabic, and more
- **Language Quality Assessment**: Automatic transcript quality evaluation and enhancement

### 🤖 AI-Powered Features
- **AI Transcription Fallback**: Uses Whisper AI when subtitles are unavailable
- **Quality Enhancement**: Automatic correction of common transcription errors
- **Smart Chunking**: Intelligent text segmentation for better processing
- **Parallel Processing**: Optimized performance for long-form content

### 📊 Analysis Capabilities
- **Sentiment Timeline**: Track emotional changes throughout the video
- **Chapter Generation**: Automatic chapter detection and summarization
- **Key Insights**: Extract main themes, conclusions, and important points
- **Content Categories**: Classify content into relevant topics
- **Export Options**: Download summaries, transcripts, and chapters

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd youtube-summarizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run yt_summarizer_enhanced.py
   ```

### Dependencies
The project uses the following key packages:
- `streamlit` - Web interface
- `transformers` - AI models for summarization
- `youtube-transcript-api` - Transcript extraction
- `googletrans` - Translation support
- `faster-whisper` - AI transcription (optional)
- `nltk` - Natural language processing
- `plotly` - Interactive visualizations
- `wordcloud` - Word cloud generation

## 🚦 Quick Start

1. **Launch the application**:
   ```bash
   streamlit run yt_summarizer_enhanced.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Enter a YouTube URL** in the input field

4. **Configure settings** in the sidebar:
   - Enable/disable features (sentiment analysis, word clouds, etc.)
   - Set translation preferences
   - Choose summary detail level
   - Configure advanced options

5. **Click analyze** and wait for processing to complete

6. **Explore results**:
   - Read comprehensive summaries
   - View interactive charts and visualizations
   - Download transcripts and analysis results

## 📋 Usage Examples

### Basic Usage
```python
# Simply paste any YouTube URL
https://www.youtube.com/watch?v=VIDEO_ID
```

### Supported Video Types
- ✅ Videos with official subtitles
- ✅ Videos with auto-generated captions
- ✅ Videos in multiple languages (with translation)
- ✅ Long-form content (lectures, podcasts, tutorials)
- ✅ Short-form content (clips, summaries)
- ✅ Hindi videos (automatic detection and translation)

### Configuration Options

#### Translation Settings
- **Auto-translate foreign videos**: Automatically translate non-English content
- **Auto-detect & translate Hindi**: Special Hindi detection and translation
- **Preferred languages**: Set priority order for subtitle languages
- **Show original language info**: Display detected language details

#### Analysis Options
- **Sentiment Analysis**: Track emotional tone throughout the video
- **Word Cloud**: Generate visual word frequency representations
- **Auto-Generate Chapters**: Create timeline-based chapter summaries
- **Key Phrase Extraction**: Identify important terms and concepts
- **Content Categorization**: Classify video topics

#### Performance Options
- **Force AI Transcription**: Use AI even when subtitles are available
- **Enhance Transcript Quality**: Apply automatic corrections
- **Fast Processing Mode**: Optimized for speed (in fast_processing variant)
- **Quick Analysis**: Ultra-fast basic summary only (in ultra_fast variant)

## 🎛️ Available Versions

The project includes several optimized versions for different use cases:

### 1. Enhanced Version (`yt_summarizer_enhanced.py`)
- **Full-featured version** with all capabilities
- Translation support and Hindi auto-detection
- AI transcription fallback
- Comprehensive analysis and visualizations
- **Best for**: Complete analysis and translation needs

### 2. Fast Processing Version (`yt_summarizer_fast_processing.py`)
- **Performance-optimized** for faster processing
- Parallel processing and caching
- Reduced analysis time for long videos
- **Best for**: Quick analysis of long-form content

### 3. Ultra Fast Version (`yt_summarizer_ultra_fast.py`)
- **Minimal processing** for maximum speed
- Basic summaries and key phrases only
- Streamlined interface
- **Best for**: Rapid overview of video content

### 4. Basic Version (`yt_summarizer.py`)
- **Standard functionality** without AI transcription
- Core summarization and analysis features
- Lightweight and stable
- **Best for**: Simple summarization needs

## 🚀 Performance Optimization

### For Large Videos (>1 hour)
1. Enable **Fast Processing Mode**
2. Use **Quick Analysis** for initial overview
3. Disable heavy visualizations (word clouds, complex charts)
4. Choose **Concise** summary detail level

### For Multiple Videos
1. Use the **Ultra Fast** version for batch processing
2. Enable caching for repeated analysis
3. Process videos sequentially to avoid memory issues

### For Hindi Content
1. Enable **Auto-detect & translate Hindi** for seamless processing
2. The system automatically optimizes Hindi translation with smaller batches
3. Hindi translation takes slightly longer but provides better accuracy

## 🌍 Language Support

### Fully Supported Languages
- **English** (en) - Native support
- **Hindi** (hi) - Special auto-detection and translation
- **Spanish** (es) - Auto-translation
- **French** (fr) - Auto-translation
- **German** (de) - Auto-translation
- **Italian** (it) - Auto-translation
- **Portuguese** (pt) - Auto-translation
- **Russian** (ru) - Auto-translation
- **Japanese** (ja) - Auto-translation
- **Korean** (ko) - Auto-translation
- **Chinese** (zh) - Auto-translation
- **Arabic** (ar) - Auto-translation

### Translation Features
- **Automatic Language Detection**: Identifies video language automatically
- **Batch Translation**: Efficient processing of long transcripts
- **Quality Assessment**: Evaluates translation accuracy
- **Fallback Support**: Uses original language if translation fails

## 📊 Output Formats

### Summary Types
1. **Executive Summary**: High-level overview and key takeaways
2. **Detailed Timeline**: Time-stamped summaries with chapters
3. **Key Insights**: Main themes, conclusions, and important points
4. **Comprehensive Analysis**: Complete breakdown with all features

### Export Options
- **Text Files**: Summary, transcript, and chapters as `.txt`
- **Structured Data**: Timeline information with timestamps
- **Analysis Results**: Charts and visualizations (when applicable)

### Visualization Types
- **Sentiment Timeline**: Emotional progression chart
- **Word Cloud**: Visual word frequency representation
- **Topic Distribution**: Content category breakdown
- **Chapter Timeline**: Visual chapter navigation

## 🔧 Troubleshooting

### Common Issues

#### "No transcript available"
**Solutions**:
1. Enable **Force AI Transcription** in settings
2. Ensure video has public visibility
3. Try a different video URL format
4. Install AI transcription packages: `pip install faster-whisper pytube pydub`

#### Video access restricted / XML parsing errors
**Solutions**:
1. **Try AI Transcription**: Enable in sidebar settings
2. **Check Video Access**: Ensure video is public and not age-restricted
3. **Wait and Retry**: May be temporary YouTube restrictions
4. **Try Different Video**: Some videos have permanent restrictions

#### HTTP 400/403 errors
**Solutions**:
1. **Rate Limiting**: Wait a few minutes before retrying
2. **Geographic Restrictions**: Video may not be available in your region
3. **Authentication Required**: Video may require login to access
4. **Use VPN**: Try accessing from different location

#### Hindi translation not working
**Solutions**:
1. Check internet connection
2. Verify googletrans installation: `pip install googletrans==4.0.0rc1`
3. Try disabling auto-translation and use original language
4. Restart the application

#### Translation not working
**Solutions**:
1. Check internet connection
2. Verify googletrans installation: `pip install googletrans==4.0.0rc1`
3. Try disabling auto-translation and use original language
4. Restart the application

#### Slow processing
**Solutions**:
1. Use **Fast Processing** or **Ultra Fast** versions
2. Disable heavy features (word clouds, sentiment analysis)
3. Choose **Concise** summary level
4. Close other applications to free up memory

#### AI transcription fails
**Solutions**:
1. Ensure all AI packages are installed: `pip install faster-whisper SpeechRecognition pydub pytube`
2. Check video length (very long videos may timeout)
3. Verify audio quality is sufficient
4. Try with a different video

### Error Messages

| Error | Solution |
|-------|----------|
| "Invalid YouTube URL" | Ensure URL format: `https://www.youtube.com/watch?v=VIDEO_ID` |
| "Video Access Restricted" | Enable AI Transcription or try different video |
| "XML parsing failed" | YouTube API restrictions - try AI transcription |
| "HTTP 400 Error" | Rate limiting or video restrictions - wait and retry |
| "HTTP 403 Error" | Access forbidden - video may be private/geo-restricted |
| "Translation failed" | Check internet connection and googletrans installation |
| "Hindi translation failed" | Verify connection, restart app, or disable auto-translation |
| "AI transcription not available" | Install AI packages or use videos with subtitles |
| "Audio download failed" | Video has download restrictions - try different video |
| "Processing timeout" | Use faster version or shorter videos |

## 🔄 Updates and Maintenance

### Keeping Dependencies Updated
```bash
pip install --upgrade -r requirements.txt
```

### Model Updates
The application automatically downloads and caches AI models:
- **Summarization models**: Updated automatically
- **Sentiment analysis models**: Cached after first use
- **Whisper models**: Downloaded as needed

### Performance Monitoring
The application tracks:
- Processing time for optimization
- Memory usage during analysis
- Translation accuracy and speed
- Cache effectiveness

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Make your changes
5. Test with various video types
6. Submit a pull request

### Areas for Contribution
- **Performance optimization**: Faster processing algorithms
- **Language support**: Additional language pairs
- **UI improvements**: Better user interface design
- **Feature additions**: New analysis capabilities
- **Bug fixes**: Resolve issues and edge cases

### Testing
Use the included test script to verify functionality:
```bash
python test_translation.py
```

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **YouTube Transcript API**: For transcript extraction
- **Hugging Face Transformers**: For AI summarization models
- **Google Translate**: For translation capabilities
- **OpenAI Whisper**: For AI transcription
- **Streamlit**: For the web interface
- **NLTK**: For natural language processing

## 📞 Support

For support, issues, or feature requests:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Include error messages and system information

---

**Made with ❤️ for the YouTube community**

*Transform any YouTube video into actionable insights with AI-powered analysis and seamless translation support.*
