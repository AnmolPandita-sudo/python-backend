#!/usr/bin/env python3
"""
Quick start script for sentiment analysis API
Minimal dependencies version for testing
"""

import sys
import subprocess

def install_minimal_deps():
    """Install only essential dependencies"""
    essential_packages = [
        'flask==2.3.3',
        'flask-cors==4.0.0',
        'vaderSentiment==3.3.2',
        'textblob==0.17.1'
    ]
    
    print("Installing minimal dependencies...")
    for package in essential_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def test_imports():
    """Test if essential imports work"""
    try:
        from flask import Flask
        from flask_cors import CORS
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from textblob import TextBlob
        print("✅ All essential imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_minimal_server():
    """Create a minimal working server"""
    server_code = '''
from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import json

app = Flask(__name__)
CORS(app)

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'available_methods': {
            'vader': True,
            'textblob': True,
            'spacy': False,
            'finbert': False,
            'bert': False
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # VADER Analysis
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_sentiment = 'positive' if vader_scores['compound'] >= 0.05 else 'negative' if vader_scores['compound'] <= -0.05 else 'neutral'
        
        # TextBlob Analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_sentiment = 'positive' if textblob_polarity > 0.1 else 'negative' if textblob_polarity < -0.1 else 'neutral'
        
        # Simple consensus
        consensus_score = (vader_scores['compound'] + textblob_polarity) / 2
        consensus_sentiment = 'positive' if consensus_score > 0.1 else 'negative' if consensus_score < -0.1 else 'neutral'
        
        result = {
            'text': text,
            'timestamp': '2024-01-01T00:00:00',
            'sentiment_results': [
                {
                    'method': 'VADER',
                    'sentiment': vader_sentiment,
                    'score': vader_scores['compound'],
                    'confidence': abs(vader_scores['compound']),
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu'],
                    'compound': vader_scores['compound']
                },
                {
                    'method': 'TextBlob',
                    'sentiment': textblob_sentiment,
                    'score': textblob_polarity,
                    'confidence': abs(textblob_polarity),
                    'positive': max(0, textblob_polarity),
                    'negative': max(0, -textblob_polarity),
                    'neutral': 1 - abs(textblob_polarity)
                }
            ],
            'consensus_sentiment': {
                'label': consensus_sentiment,
                'score': consensus_score,
                'confidence': abs(consensus_score),
                'method_count': 2,
                'agreement': abs(vader_scores['compound'] - textblob_polarity)
            },
            'entities': [],
            'keywords': {
                'bullish': [],
                'bearish': [],
                'neutral': []
            },
            'financial_metrics': {
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'avg_sentence_length': len(text.split()) / max(len(text.split('.')), 1),
                'readability_score': 7.5,
                'financial_keyword_density': 0.0
            },
            'market_indicators': {
                'detected_stocks': [],
                'detected_people': [],
                'detected_locations': [],
                'market_impact': {
                    'score': 5,
                    'level': 'Medium',
                    'factors': []
                }
            },
            'recommendations': [
                'Basic sentiment analysis completed',
                'Consider using full version for advanced features'
            ]
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Minimal Sentiment Analysis API...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/health")
    print("⚡ Using VADER + TextBlob for sentiment analysis")
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
    
    with open('minimal_server.py', 'w') as f:
        f.write(server_code)
    
    print("✅ Created minimal_server.py")

def main():
    print("🔧 Quick Start Setup for Sentiment Analysis API")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    
    # Install minimal dependencies
    if not install_minimal_deps():
        print("❌ Failed to install dependencies")
        return
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return
    
    # Create minimal server
    create_minimal_server()
    
    print("\n✅ Setup completed successfully!")
    print("\n🚀 To start the server:")
    print("python minimal_server.py")
    print("\n🌐 Then open your browser to:")
    print("http://localhost:5000/health")

if __name__ == "__main__":
    main()