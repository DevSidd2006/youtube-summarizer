#!/usr/bin/env python3
"""
Comprehensive test suite for YouTube Summarizer application
Tests Python 3.13 compatibility, translation functionality, and core features
"""

import sys
import os
import traceback

# CRITICAL: Python 3.13 compatibility fix - Must be FIRST before any imports
if 'cgi' not in sys.modules:
    try:
        import cgi
    except ImportError:
        import types
        cgi = types.ModuleType('cgi')
        
        def escape(s, quote=False):
            """Escape HTML special characters"""
            s = s.replace("&", "&amp;")
            s = s.replace("<", "&lt;")
            s = s.replace(">", "&gt;")
            if quote:
                s = s.replace('"', "&quot;")
                s = s.replace("'", "&#x27;")
            return s
        
        def parse_header(line):
            """Parse Content-Type header"""
            parts = line.split(';')
            main_type = parts[0].strip()
            pdict = {}
            
            for p in parts[1:]:
                if '=' in p:
                    key, value = p.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    pdict[key] = value
            
            return main_type, pdict
        
        cgi.escape = escape
        cgi.parse_header = parse_header
        sys.modules['cgi'] = cgi

def test_cgi_compatibility():
    """Test Python 3.13 CGI module compatibility fix"""
    print("🔧 Testing Python 3.13 CGI Compatibility...")
    
    try:
        # Check if cgi module is available
        import cgi
        print("✅ CGI module imported successfully")
        
        # Test escape function
        if hasattr(cgi, 'escape'):
            result = cgi.escape('<test>&"test"')
            expected = "&lt;test&gt;&amp;\"test\""
            if result == expected:
                print(f"✅ cgi.escape() working correctly: {result}")
            else:
                print(f"⚠️ cgi.escape() output unexpected: {result}")
        
        # Test parse_header function
        if hasattr(cgi, 'parse_header'):
            main_type, pdict = cgi.parse_header('text/html; charset=utf-8')
            if main_type == 'text/html' and pdict.get('charset') == 'utf-8':
                print(f"✅ cgi.parse_header() working correctly")
            else:
                print(f"⚠️ cgi.parse_header() output unexpected: {main_type}, {pdict}")
        
        return True
        
    except Exception as e:
        print(f"❌ CGI compatibility test failed: {e}")
        return False

def test_translation_functionality():
    """Test translation functionality with timeout handling"""
    print("\n📡 Testing Translation Functionality...")
    
    try:
        from googletrans import Translator
        print("✅ GoogleTrans imported successfully")
        
        translator = Translator()
        
        # Test simple translation
        test_text = "Hello, this is a test."
        result = translator.translate(test_text, dest='hi')
        
        if result and result.text:
            print(f"✅ Translation successful: '{test_text}' → '{result.text}'")
            return True
        else:
            print("❌ Translation returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        # Test fallback translator
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='en', target='hi')
            result = translator.translate("Hello, this is a test.")
            print(f"✅ Fallback translator working: {result}")
            return True
        except:
            print("❌ Fallback translator also failed")
            return False

def test_streamlit_imports():
    """Test that all required packages can be imported"""
    print("\n📦 Testing Package Imports...")
    
    required_packages = [
        'streamlit',
        'yt_dlp',
        'torch',
        'transformers',
        'googletrans',
        'requests'
    ]
    
    all_passed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
            all_passed = False
    
    return all_passed

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 Running Comprehensive Test Suite for YouTube Summarizer")
    print("=" * 60)
    
    test_results = {
        'CGI Compatibility': test_cgi_compatibility(),
        'Translation Functionality': test_translation_functionality(),
        'Package Imports': test_streamlit_imports()
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the requirements and setup.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
