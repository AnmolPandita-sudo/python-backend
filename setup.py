#!/usr/bin/env python3
"""
Setup script for Advanced Sentiment Analysis API
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    """Download spaCy English model"""
    print("Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to download spaCy model. You can download it manually with:")
        print("python -m spacy download en_core_web_sm")

def download_nltk_data():
    """Download NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('stopwords')
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to download NLTK data: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up Advanced Sentiment Analysis API...")
    
    try:
        install_requirements()
        download_spacy_model()
        download_nltk_data()
        
        print("\n✅ Setup completed successfully!")
        print("\nTo start the API server, run:")
        print("python sentiment_analyzer.py")
        print("\nThe API will be available at: http://localhost:5000")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()