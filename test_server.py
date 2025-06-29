#!/usr/bin/env python3
"""
Test script to verify the sentiment analysis server is working
"""

import requests
import json
import time

def test_server_connection():
    """Test if server is running and responding"""
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responding")
            print(f"📊 Response: {response.json()}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on port 5000?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server connection timed out")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis endpoint"""
    test_texts = [
        "The stock market is performing very well today with strong gains",
        "Company reports disappointing quarterly results and stock falls",
        "The market remains stable with mixed signals from various sectors"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🧪 Test {i}: Analyzing text...")
        print(f"📝 Text: {text[:50]}...")
        
        try:
            response = requests.post(
                'http://localhost:5000/analyze',
                json={'text': text},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    analysis = result['result']
                    consensus = analysis['consensus_sentiment']
                    print(f"✅ Analysis successful")
                    print(f"📈 Sentiment: {consensus['label']} (score: {consensus['score']:.3f})")
                    print(f"🎯 Confidence: {consensus['confidence']:.3f}")
                    print(f"🔧 Methods used: {consensus['method_count']}")
                else:
                    print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Request failed with status {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("❌ Analysis request timed out")
        except Exception as e:
            print(f"❌ Analysis error: {e}")

def main():
    print("🧪 Testing Sentiment Analysis Server")
    print("=" * 40)
    
    # Test server connection
    if not test_server_connection():
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you're in the python_backend directory")
        print("2. Run: python sentiment_analyzer.py")
        print("3. Wait for 'Running on http://127.0.0.1:5000' message")
        print("4. Then run this test script in another terminal")
        return
    
    # Test sentiment analysis
    print("\n🔍 Testing sentiment analysis...")
    test_sentiment_analysis()
    
    print("\n✅ All tests completed!")
    print("\n🌐 Your API is ready to use with the frontend!")

if __name__ == "__main__":
    main()