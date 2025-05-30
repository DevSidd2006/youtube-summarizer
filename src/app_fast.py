# Enhanced YouTube Summarizer with Performance Optimizations
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import re
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import asyncio
import multiprocessing

# Enhanced imports for AI transcription (optional)
import os
import warnings
warnings.filterwarnings("ignore")

# Performance optimization imports
from functools import lru_cache
import pickle

try:
    import speech_recognition as sr
    from pytube import YouTube
    from pydub import AudioSegment
    AI_TRANSCRIPTION_AVAILABLE = True
    WHISPER_AVAILABLE = False
    
    # Try to import faster-whisper (more stable alternative)
    try:
        from faster_whisper import WhisperModel
        WHISPER_AVAILABLE = True
        WHISPER_TYPE = "faster"
    except ImportError:
        WHISPER_AVAILABLE = False
        WHISPER_TYPE = None
        
except ImportError as e:
    AI_TRANSCRIPTION_AVAILABLE = False
    WHISPER_AVAILABLE = False
    WHISPER_TYPE = None

# Download required NLTK data only once
@st.cache_data
def download_nltk_data():
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        nltk.download('punkt', quiet=True)

download_nltk_data()

# ----------------------
# PERFORMANCE OPTIMIZATIONS
# ----------------------

# Use smaller, faster models for better performance
@st.cache_resource
def load_fast_summarizer():
    """Load a faster, smaller summarization model"""
    try:
        # Use a smaller, faster model
        return pipeline("summarization", 
                       model="sshleifer/distilbart-cnn-6-6",  # Much faster than bart-large
                       device=0 if st.session_state.get('use_gpu', False) else -1)
    except:
        # Fallback to CPU
        return pipeline("summarization", 
                       model="sshleifer/distilbart-cnn-6-6",
                       device=-1)

@st.cache_resource  
def load_fast_sentiment_analyzer():
    """Load a faster sentiment analysis model"""
    return pipeline("sentiment-analysis", 
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                   device=-1)  # Use CPU for faster startup

@st.cache_resource
def load_whisper_model():
    """Load and cache Whisper model for audio transcription"""
    if not AI_TRANSCRIPTION_AVAILABLE or not WHISPER_AVAILABLE:
        return None
    try:
        if WHISPER_TYPE == "faster":
            # Use faster-whisper with optimized settings
            return WhisperModel("base", device="cpu", compute_type="int8", num_workers=2)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# ----------------------
# FAST PROCESSING FUNCTIONS
# ----------------------

@lru_cache(maxsize=100)
def cached_sentence_tokenize(text_hash):
    """Cache sentence tokenization for repeated texts"""
    return sent_tokenize(text_hash)

def fast_chunk_text(text, max_words=400):
    """Optimized text chunking with caching"""
    words = text.split()
    chunks = []
    
    # Process in batches for better memory efficiency
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        if len(chunk.strip()) > 50:  # Only include meaningful chunks
            chunks.append(chunk)
    
    return chunks

def parallel_summarize_chunks(chunks, max_workers=3):
    """Parallelize chunk summarization for speed"""
    summarizer = load_fast_summarizer()
    summaries = []
    
    def process_chunk(chunk):
        try:
            if len(chunk.strip()) < 50:
                return ""
                
            words = len(chunk.split())
            max_len = min(130, max(50, words // 3))  # Shorter summaries for speed
            min_len = min(30, max_len // 2)
            
            result = summarizer(chunk, 
                              max_length=max_len, 
                              min_length=min_len, 
                              do_sample=False)
            
            return result[0]['summary_text'] if result else ""
        except Exception as e:
            print(f"Error in chunk processing: {e}")
            return ""
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(future_to_chunk):
            try:
                summary = future.result(timeout=30)  # 30 second timeout per chunk
                if summary:
                    summaries.append(summary)
            except Exception as e:
                print(f"Chunk processing failed: {e}")
                continue
    
    return summaries

def fast_generate_summary(text, is_long_form=False):
    """Optimized summary generation with parallel processing"""
    try:
        if not text or len(text.strip()) < 100:
            return "Insufficient content for summarization."
        
        word_count = len(text.split())
        
        # Optimize chunk size based on content length
        if word_count > 8000:  # Long content
            max_chunk = 300  # Smaller chunks for parallel processing
            max_workers = 4
        elif word_count > 3000:  # Medium content
            max_chunk = 400
            max_workers = 3
        else:  # Short content
            max_chunk = 500
            max_workers = 2
        
        # Fast chunking
        chunks = fast_chunk_text(text, max_chunk)
        
        if not chunks:
            return "No content available for summarization."
        
        # For very long content, use hierarchical approach
        if len(chunks) > 15:
            return fast_hierarchical_summary(chunks, max_workers)
        else:
            summaries = parallel_summarize_chunks(chunks, max_workers)
            return " ".join(summaries) if summaries else "Unable to generate summary."
            
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def fast_hierarchical_summary(chunks, max_workers=4):
    """Fast hierarchical summarization for long content"""
    try:
        # Process chunks in groups for speed
        group_size = 8  # Larger groups for efficiency
        intermediate_summaries = []
        
        # Process groups in parallel
        chunk_groups = [chunks[i:i+group_size] for i in range(0, len(chunks), group_size)]
        
        def process_group(group):
            group_text = " ".join(group)[:3000]  # Limit text length
            try:
                summarizer = load_fast_summarizer()
                result = summarizer(group_text, 
                                  max_length=150, 
                                  min_length=60, 
                                  do_sample=False)
                return result[0]['summary_text'] if result else ""
            except:
                return ""
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            intermediate_summaries = list(filter(None, executor.map(process_group, chunk_groups)))
        
        if not intermediate_summaries:
            return "Unable to generate summary."
        
        # Final summary
        final_text = " ".join(intermediate_summaries)
        if len(final_text.split()) > 800:
            summarizer = load_fast_summarizer()
            final_result = summarizer(final_text[:2000], 
                                    max_length=200, 
                                    min_length=80, 
                                    do_sample=False)
            return final_result[0]['summary_text'] if final_result else final_text
        
        return final_text
        
    except Exception as e:
        return f"Error in hierarchical summarization: {str(e)}"

def fast_sentiment_analysis(transcript_data):
    """Optimized sentiment analysis with sampling"""
    sentiment_analyzer = load_fast_sentiment_analyzer()
    
    if isinstance(transcript_data, str):
        return [{'timestamp': '00:00', 'sentiment': 'NEUTRAL', 'score': 0.0}]
    
    # Sample every 2-3 minutes instead of every minute for speed
    segments = create_timeline_segments(transcript_data, 150)  # 2.5 minute segments
    
    sentiment_timeline = []
    
    def analyze_segment(segment):
        text = segment['text'][:300]  # Limit text length for speed
        if len(text.strip()) > 20:
            try:
                result = sentiment_analyzer(text)[0]
                return {
                    'timestamp': segment['timestamp'],
                    'sentiment': result['label'],
                    'score': result['score'],
                    'text_sample': text[:50] + "..."
                }
            except:
                return {
                    'timestamp': segment['timestamp'],
                    'sentiment': 'NEUTRAL',
                    'score': 0.0,
                    'text_sample': text[:50] + "..."
                }
        return None
    
    # Parallel sentiment analysis
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(analyze_segment, segments)
        sentiment_timeline = [r for r in results if r is not None]
    
    return sentiment_timeline

def fast_key_phrases(text):
    """Optimized key phrase extraction"""
    # Simple but fast approach
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    
    # Common stop words
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 
                  'were', 'said', 'what', 'make', 'like', 'time', 'very', 'when', 
                  'come', 'here', 'just', 'know', 'take', 'people', 'into', 'year', 
                  'your', 'good', 'some', 'could', 'them', 'think', 'would', 'more',
                  'about', 'after', 'back', 'first', 'well', 'work', 'way', 'even',
                  'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most',
                  'us', 'going', 'right', 'may', 'still', 'much', 'through', 'down'}
    
    return [word for word, freq in word_freq.most_common(15) 
            if word not in stop_words and freq >= 2]

def fast_create_wordcloud(text):
    """Optimized word cloud generation"""
    try:
        # Limit text length for faster processing
        text_sample = text[:5000] if len(text) > 5000 else text
        
        wordcloud = WordCloud(width=600, height=300,  # Smaller size for speed
                             background_color='white',
                             colormap='viridis',
                             max_words=50,  # Limit words for speed
                             relative_scaling=0.5).generate(text_sample)
        
        fig, ax = plt.subplots(figsize=(8, 4))  # Smaller figure
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None

# ----------------------
# STREAMLINED MAIN FUNCTIONS
# ----------------------

def fast_comprehensive_analysis(transcript_data):
    """Fast comprehensive analysis with parallel processing"""
    
    # Get text from transcript
    if isinstance(transcript_data, str):
        full_text = transcript_data
    else:
        full_text = " ".join([t['text'] for t in transcript_data])
    
    # Use ThreadPoolExecutor for parallel processing of different analyses
    results = {}
    
    def generate_overview():
        return fast_generate_summary(full_text)
    
    def generate_detailed():
        return fast_detailed_summary(transcript_data)
    
    def extract_phrases():
        return fast_key_phrases(full_text)
    
    def analyze_categories():
        return fast_content_categories(full_text)
    
    # Run analyses in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_overview = executor.submit(generate_overview)
        future_detailed = executor.submit(generate_detailed)
        future_phrases = executor.submit(extract_phrases)
        future_categories = executor.submit(analyze_categories)
        
        # Collect results with timeout
        try:
            results['overview'] = future_overview.result(timeout=60)
            results['detailed'] = future_detailed.result(timeout=90)
            results['phrases'] = future_phrases.result(timeout=10)
            results['categories'] = future_categories.result(timeout=10)
        except Exception as e:
            print(f"Parallel processing error: {e}")
            # Fallback to sequential processing
            results['overview'] = fast_generate_summary(full_text)
            results['detailed'] = "Detailed summary temporarily unavailable."
            results['phrases'] = fast_key_phrases(full_text)
            results['categories'] = fast_content_categories(full_text)
    
    return results

def fast_detailed_summary(transcript_data, segment_duration=240):
    """Fast detailed summary with optimized processing"""
    try:
        segments = create_timeline_segments(transcript_data, segment_duration)
        detailed_summary = "## 📋 **DETAILED VIDEO SUMMARY WITH TIMELINE**\n\n"
        
        # Process only key segments for speed (every 2nd segment for long content)
        if len(segments) > 10:
            segments = segments[::2]  # Process every other segment
        
        def process_segment_fast(segment_info):
            i, segment = segment_info
            timestamp = segment["timestamp"]
            text = segment["text"].strip()
            
            if len(text) < 80:  # Skip short segments
                return None
            
            try:
                # Use fast summarization
                words = len(text.split())
                max_len = min(120, max(60, words // 4))  # Shorter summaries
                min_len = min(40, max_len // 2)
                
                summarizer = load_fast_summarizer()
                out = summarizer(text[:1500], max_length=max_len, min_length=min_len, do_sample=False)
                segment_summary = out[0]['summary_text'] if out and len(out) > 0 else "Summary unavailable."
                
                return f"### 🕐 **[{timestamp}]** - Segment {i+1}\n{segment_summary}\n\n"
                
            except Exception as e:
                return None
        
        # Parallel processing with smaller thread pool for speed
        with ThreadPoolExecutor(max_workers=2) as executor:
            segment_results = list(executor.map(process_segment_fast, enumerate(segments)))
        
        # Combine results
        for result in segment_results:
            if result:
                detailed_summary += result
        
        return detailed_summary
        
    except Exception as e:
        return f"Error generating detailed summary: {str(e)}"

def fast_content_categories(text):
    """Fast content categorization"""
    categories = {
        'Technology': ['tech', 'ai', 'software', 'digital', 'computer'],
        'Business': ['business', 'marketing', 'money', 'company'],
        'Education': ['learn', 'education', 'tutorial', 'course'],
        'Entertainment': ['movie', 'music', 'game', 'fun'],
        'Health': ['health', 'fitness', 'medical'],
        'Science': ['science', 'research', 'study']
    }
    
    text_lower = text.lower()
    category_scores = {}
    
    for category, keywords in categories.items():
        score = sum(text_lower.count(keyword) for keyword in keywords)
        if score > 0:
            category_scores[category] = score
    
    return sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]

# ----------------------
# ENHANCED STREAMLIT APP WITH SPEED OPTIMIZATIONS
# ----------------------

# Initialize session state for performance tracking
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []

st.set_page_config(
    page_title="⚡ Ultra-Fast YouTube Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Ultra-Fast YouTube Video Analyzer")
st.markdown("*Lightning-fast AI analysis with optimized processing*")

# Performance settings in sidebar
with st.sidebar:
    st.header("⚙️ Performance Settings")
    
    # Speed optimization options
    st.subheader("🚀 Speed Optimizations")
    enable_fast_mode = st.checkbox("⚡ Fast Mode", value=True, 
                                  help="Use optimized models and parallel processing")
    
    use_gpu = st.checkbox("🖥️ Use GPU (if available)", value=False,
                         help="Enable GPU acceleration for faster processing")
    st.session_state.use_gpu = use_gpu
    
    max_workers = st.slider("🔄 Parallel Workers", 1, 6, 3,
                           help="Number of parallel workers for processing")
    
    # Analysis options
    st.subheader("📊 Analysis Options")
    enable_detailed_timeline = st.checkbox("Detailed Timeline", value=True)
    enable_sentiment = st.checkbox("Sentiment Analysis", value=True)
    enable_wordcloud = st.checkbox("Word Cloud", value=True)
    enable_quick_analysis = st.checkbox("🚀 Quick Analysis Only", value=False,
                                       help="Skip detailed analysis for maximum speed")

# Main input
video_url = st.text_input("🔗 Enter YouTube Video URL:", 
                         placeholder="https://www.youtube.com/watch?v=...")

if video_url:
    # Start timing
    total_start_time = time.time()
    
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL.")
        st.stop()

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Get metadata (fast)
    status_text.text("🔍 Fetching video info...")
    progress_bar.progress(10)
    
    metadata = get_video_metadata(video_id)
    
    # Step 2: Get transcript (optimized)
    status_text.text("📝 Extracting transcript...")
    progress_bar.progress(25)
    
    transcript_start = time.time()
    transcript = get_transcript(video_id)
    transcript_time = time.time() - transcript_start
    
    if isinstance(transcript, str) and "Error" in transcript:
        st.error(f"❌ {transcript}")
        st.stop()

    progress_bar.progress(40)
    
    # Get text and basic stats
    full_text = get_transcript_text(transcript)
    word_count = len(full_text.split())
    
    # Display quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Words", f"{word_count:,}")
    with col2:
        st.metric("⏰ Extract Time", f"{transcript_time:.1f}s")
    with col3:
        complexity = "High" if word_count > 8000 else "Medium" if word_count > 3000 else "Low"
        st.metric("🧠 Complexity", complexity)
    with col4:
        st.metric("🚀 Fast Mode", "ON" if enable_fast_mode else "OFF")
    
    # Step 3: Fast comprehensive analysis
    status_text.text("🚀 Running fast analysis...")
    progress_bar.progress(60)
    
    if enable_quick_analysis:
        # Ultra-fast mode - basic summary only
        analysis_start = time.time()
        overview_summary = fast_generate_summary(full_text)
        key_phrases = fast_key_phrases(full_text)
        analysis_time = time.time() - analysis_start
        
        st.markdown("---")
        st.markdown(f"# 🎯 **QUICK SUMMARY**\n\n{overview_summary}")
        
        if key_phrases:
            st.markdown("## 🔑 **Key Phrases**")
            phrase_html = ""
            for phrase in key_phrases:
                phrase_html += f'<span style="background-color: #e1f5fe; padding: 4px 8px; margin: 2px; border-radius: 12px; display: inline-block;">{phrase}</span> '
            st.markdown(phrase_html, unsafe_allow_html=True)
        
        total_time = time.time() - total_start_time
        st.success(f"⚡ **Ultra-fast analysis complete!** Total time: {total_time:.1f}s")
        
    else:
        # Full fast analysis
        analysis_start = time.time()
        results = fast_comprehensive_analysis(transcript)
        analysis_time = time.time() - analysis_start
        
        progress_bar.progress(80)
        status_text.text("📊 Generating visualizations...")
        
        # Display comprehensive results
        st.markdown("---")
        
        # Main summary
        st.markdown(f"""
        # 🎯 **EXECUTIVE SUMMARY**
        
        {results['overview']}
        
        ## 📊 **CONTENT ANALYSIS**
        - **Primary Topics:** {', '.join([f"{cat} ({score} mentions)" for cat, score in results['categories']]) if results['categories'] else 'General Content'}
        - **Key Phrases:** {', '.join(results['phrases'][:8]) if results['phrases'] else 'No key phrases detected'}
        
        ---
        
        {results['detailed'] if enable_detailed_timeline else ""}
        """)
        
        # Fast visualizations
        if enable_wordcloud or enable_sentiment:
            col1, col2 = st.columns(2)
            
            with col1:
                if enable_wordcloud:
                    st.subheader("☁️ Word Cloud")
                    wordcloud_fig = fast_create_wordcloud(full_text)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
            
            with col2:
                if enable_sentiment:
                    st.subheader("🎭 Sentiment Analysis")
                    sentiment_data = fast_sentiment_analysis(transcript)
                    
                    if sentiment_data and len(sentiment_data) > 1:
                        sentiment_df = pd.DataFrame(sentiment_data)
                        fig = px.bar(sentiment_df, x='timestamp', y='score', 
                                   color='sentiment', title="Sentiment Timeline")
                        st.plotly_chart(fig, use_container_width=True)
    
    # Final progress
    progress_bar.progress(100)
    total_time = time.time() - total_start_time
    
    # Performance summary
    st.markdown("---")
    performance_col1, performance_col2, performance_col3 = st.columns(3)
    
    with performance_col1:
        st.metric("⚡ Total Time", f"{total_time:.1f}s")
    with performance_col2:
        processing_speed = word_count / total_time if total_time > 0 else 0
        st.metric("🚀 Processing Speed", f"{processing_speed:.0f} words/sec")
    with performance_col3:
        st.metric("🧮 Analysis Time", f"{analysis_time:.1f}s")
    
    # Store performance data
    st.session_state.processing_times.append({
        'words': word_count,
        'total_time': total_time,
        'speed': processing_speed
    })
    
    status_text.text(f"✅ Analysis complete! {processing_speed:.0f} words/sec")
    
    # Remove progress bar
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

# Performance history
if st.session_state.processing_times:
    with st.expander("📈 Performance History"):
        perf_df = pd.DataFrame(st.session_state.processing_times)
        
        col1, col2 = st.columns(2)
        with col1:
            avg_speed = perf_df['speed'].mean()
            st.metric("Average Speed", f"{avg_speed:.0f} words/sec")
        with col2:
            best_speed = perf_df['speed'].max()
            st.metric("Best Speed", f"{best_speed:.0f} words/sec")
        
        fig = px.line(perf_df, y='speed', title="Processing Speed Over Time")
        st.plotly_chart(fig, use_container_width=True)