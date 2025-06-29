#!/usr/bin/env python3
"""
Advanced Sentiment Analysis API using multiple NLP libraries
Supports VADER, TextBlob, spaCy, and Transformers (BERT-based models)
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Core libraries
import numpy as np
import pandas as pd

# NLP Libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Install with: pip install textblob")

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Install with: pip install spacy")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    
    # Initialize FinBERT for financial sentiment analysis
    try:
        finbert_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
        FINBERT_AVAILABLE = True
    except Exception as e:
        print(f"FinBERT not available: {e}")
        FINBERT_AVAILABLE = False
        
    # Initialize general BERT model as fallback
    try:
        bert_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        BERT_AVAILABLE = True
    except Exception as e:
        print(f"BERT not available: {e}")
        BERT_AVAILABLE = False
        
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    FINBERT_AVAILABLE = False
    BERT_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

# Flask for API
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

@dataclass
class SentimentResult:
    """Structured sentiment analysis result"""
    method: str
    sentiment: str
    score: float
    confidence: float
    positive: float
    negative: float
    neutral: float
    compound: Optional[float] = None

@dataclass
class EntityResult:
    """Named entity recognition result"""
    text: str
    label: str
    start: int
    end: int
    confidence: float

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    text: str
    timestamp: str
    sentiment_results: List[SentimentResult]
    consensus_sentiment: Dict[str, Any]
    entities: List[EntityResult]
    keywords: Dict[str, List[str]]
    financial_metrics: Dict[str, Any]
    market_indicators: Dict[str, Any]
    recommendations: List[str]

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analyzer using multiple NLP libraries"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self.stock_symbols = self._load_stock_symbols()
        self.financial_keywords = self._load_financial_keywords()
        
    def _load_stock_symbols(self) -> Dict[str, List[str]]:
        """Load comprehensive stock symbol mappings"""
        return {
            'RELIANCE': ['reliance', 'ril', 'mukesh ambani', 'jio', 'petrochemicals', 'refinery'],
            'TCS': ['tcs', 'tata consultancy', 'it services', 'software', 'digital transformation'],
            'HDFCBANK': ['hdfc bank', 'hdfc', 'banking', 'private bank', 'aditya puri'],
            'INFY': ['infosys', 'it services', 'software', 'narayana murthy', 'salil parekh'],
            'ICICIBANK': ['icici bank', 'icici', 'banking', 'private bank', 'sandeep bakhshi'],
            'BHARTIARTL': ['bharti airtel', 'airtel', 'telecom', 'sunil mittal', '5g'],
            'ITC': ['itc', 'cigarettes', 'tobacco', 'fmcg', 'sanjiv puri'],
            'SBIN': ['sbi', 'state bank', 'public sector bank', 'dinesh khara'],
            'LT': ['larsen toubro', 'l&t', 'engineering', 'construction', 'infrastructure'],
            'WIPRO': ['wipro', 'it services', 'software', 'rishad premji'],
            'HCLTECH': ['hcl tech', 'hcl technologies', 'it services', 'software'],
            'ASIANPAINT': ['asian paints', 'paint', 'decorative', 'amit syngle'],
            'MARUTI': ['maruti suzuki', 'maruti', 'automobile', 'cars', 'hisashi takeuchi'],
            'BAJFINANCE': ['bajaj finance', 'bajaj', 'nbfc', 'rajeev jain'],
            'KOTAKBANK': ['kotak bank', 'kotak mahindra', 'uday kotak', 'private bank'],
            'AXISBANK': ['axis bank', 'axis', 'private bank', 'amitabh chaudhry'],
            'TITAN': ['titan', 'jewellery', 'watches', 'tanishq', 'ck venkataraman'],
            'NESTLEIND': ['nestle india', 'nestle', 'fmcg', 'food', 'suresh narayanan'],
            'ULTRACEMCO': ['ultratech cement', 'ultratech', 'cement', 'kumar mangalam birla'],
            'ADANIPORTS': ['adani ports', 'adani', 'gautam adani', 'ports', 'logistics'],
            'TATASTEEL': ['tata steel', 'steel', 'tata group', 'tv narendran'],
            'HINDALCO': ['hindalco', 'aluminium', 'copper', 'aditya birla', 'satish pai'],
            'COALINDIA': ['coal india', 'coal', 'mining', 'pramod agrawal'],
            'ONGC': ['ongc', 'oil', 'gas', 'petroleum', 'alka mittal'],
            'NTPC': ['ntpc', 'power', 'electricity', 'gurdeep singh'],
            'POWERGRID': ['power grid', 'transmission', 'electricity', 'kandikuppa sreekant'],
            'BPCL': ['bharat petroleum', 'bpcl', 'oil', 'refinery', 'arun kumar singh'],
            'IOC': ['indian oil', 'ioc', 'petroleum', 'refinery', 'shrikant madhav vaidya'],
            'GAIL': ['gail', 'gas', 'natural gas', 'manoj jain'],
            'SUNPHARMA': ['sun pharma', 'pharmaceutical', 'dilip shanghvi'],
            'DRREDDY': ['dr reddy', 'pharmaceutical', 'gv prasad'],
            'CIPLA': ['cipla', 'pharmaceutical', 'umang vohra'],
            'DIVISLAB': ['divi labs', 'pharmaceutical', 'murali divi'],
            'BIOCON': ['biocon', 'biotechnology', 'kiran mazumdar shaw'],
            'TECHM': ['tech mahindra', 'it services', 'cp gurnani'],
        }
    
    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive financial keywords"""
        return {
            'bullish': [
                'growth', 'profit', 'gain', 'rise', 'surge', 'rally', 'bullish', 'optimistic',
                'beat', 'exceed', 'outperform', 'boost', 'jump', 'soar', 'climb', 'advance',
                'improve', 'recovery', 'strong', 'robust', 'solid', 'healthy', 'positive',
                'upgrade', 'buy', 'recommend', 'target', 'upside', 'momentum', 'breakthrough',
                'expansion', 'acquisition', 'merger', 'partnership', 'deal', 'contract',
                'revenue', 'earnings', 'dividend', 'bonus', 'split', 'buyback', 'all-time high',
                'record', 'milestone', 'achievement', 'success', 'winner', 'leader'
            ],
            'bearish': [
                'fall', 'drop', 'decline', 'crash', 'plunge', 'bearish', 'pessimistic',
                'miss', 'underperform', 'concern', 'worry', 'risk', 'threat', 'slide',
                'tumble', 'slump', 'weak', 'poor', 'disappointing', 'loss', 'deficit',
                'downgrade', 'sell', 'avoid', 'caution', 'warning', 'alert', 'pressure',
                'challenge', 'headwind', 'uncertainty', 'volatility', 'correction',
                'recession', 'slowdown', 'crisis', 'scandal', 'fraud', 'investigation',
                'bankruptcy', 'default', 'debt', 'liability', 'lawsuit', 'penalty'
            ],
            'neutral': [
                'stable', 'maintain', 'hold', 'unchanged', 'flat', 'sideways', 'range',
                'consolidation', 'mixed', 'neutral', 'wait', 'watch', 'monitor', 'review',
                'analysis', 'report', 'update', 'announcement', 'statement', 'meeting',
                'conference', 'discussion', 'evaluation', 'assessment', 'study'
            ],
            'market_impact': {
                'high': [
                    'rbi', 'reserve bank', 'monetary policy', 'repo rate', 'interest rate',
                    'budget', 'union budget', 'finance minister', 'economic survey',
                    'gdp', 'inflation', 'cpi', 'wpi', 'fiscal deficit', 'current account',
                    'foreign investment', 'fii', 'dii', 'sebi', 'regulatory', 'policy',
                    'election', 'government', 'prime minister', 'cabinet', 'parliament'
                ],
                'medium': [
                    'quarterly results', 'earnings', 'guidance', 'outlook', 'forecast',
                    'sector', 'industry', 'segment', 'market share', 'competition',
                    'expansion', 'capex', 'investment', 'acquisition', 'merger',
                    'ipo', 'listing', 'delisting', 'bonus', 'dividend', 'split'
                ],
                'low': [
                    'management', 'appointment', 'resignation', 'board', 'director',
                    'conference', 'meeting', 'presentation', 'interview', 'statement',
                    'clarification', 'update', 'announcement', 'press release'
                ]
            }
        }
    
    def analyze_with_vader(self, text: str) -> SentimentResult:
        """Analyze sentiment using VADER"""
        if not self.vader_analyzer:
            return None
            
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine sentiment label
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return SentimentResult(
            method='VADER',
            sentiment=sentiment,
            score=scores['compound'],
            confidence=abs(scores['compound']),
            positive=scores['pos'],
            negative=scores['neg'],
            neutral=scores['neu'],
            compound=scores['compound']
        )
    
    def analyze_with_textblob(self, text: str) -> SentimentResult:
        """Analyze sentiment using TextBlob"""
        if not TEXTBLOB_AVAILABLE:
            return None
            
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return SentimentResult(
            method='TextBlob',
            sentiment=sentiment,
            score=polarity,
            confidence=abs(polarity),
            positive=max(0, polarity),
            negative=max(0, -polarity),
            neutral=1 - abs(polarity)
        )
    
    def analyze_with_finbert(self, text: str) -> SentimentResult:
        """Analyze sentiment using FinBERT (Financial BERT)"""
        if not FINBERT_AVAILABLE:
            return None
            
        try:
            # Truncate text if too long for BERT
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
                
            result = finbert_analyzer(text)[0]
            
            # Map FinBERT labels to standard format
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral'
            }
            
            sentiment = label_mapping.get(result['label'].lower(), 'neutral')
            confidence = result['score']
            
            # Convert to score (-1 to 1)
            if sentiment == 'positive':
                score = confidence
            elif sentiment == 'negative':
                score = -confidence
            else:
                score = 0
                
            return SentimentResult(
                method='FinBERT',
                sentiment=sentiment,
                score=score,
                confidence=confidence,
                positive=confidence if sentiment == 'positive' else 0,
                negative=confidence if sentiment == 'negative' else 0,
                neutral=confidence if sentiment == 'neutral' else 0
            )
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return None
    
    def analyze_with_bert(self, text: str) -> SentimentResult:
        """Analyze sentiment using general BERT model"""
        if not BERT_AVAILABLE:
            return None
            
        try:
            # Truncate text if too long
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
                
            result = bert_analyzer(text)[0]
            
            # Map BERT labels (usually 1-5 stars) to sentiment
            label = result['label']
            confidence = result['score']
            
            if 'STAR' in label:
                stars = int(label.split()[0])
                if stars >= 4:
                    sentiment = 'positive'
                    score = (stars - 3) / 2  # 4-5 stars -> 0.5-1.0
                elif stars <= 2:
                    sentiment = 'negative'
                    score = -(3 - stars) / 2  # 1-2 stars -> -1.0--0.5
                else:
                    sentiment = 'neutral'
                    score = 0
            else:
                # Fallback for other label formats
                if 'positive' in label.lower():
                    sentiment = 'positive'
                    score = confidence
                elif 'negative' in label.lower():
                    sentiment = 'negative'
                    score = -confidence
                else:
                    sentiment = 'neutral'
                    score = 0
                    
            return SentimentResult(
                method='BERT',
                sentiment=sentiment,
                score=score,
                confidence=confidence,
                positive=confidence if sentiment == 'positive' else 0,
                negative=confidence if sentiment == 'negative' else 0,
                neutral=confidence if sentiment == 'neutral' else 0
            )
        except Exception as e:
            print(f"BERT analysis error: {e}")
            return None
    
    def extract_entities_spacy(self, text: str) -> List[EntityResult]:
        """Extract named entities using spaCy"""
        if not SPACY_AVAILABLE:
            return []
            
        try:
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append(EntityResult(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0  # spaCy doesn't provide confidence scores by default
                ))
                
            return entities
        except Exception as e:
            print(f"spaCy entity extraction error: {e}")
            return []
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities (stocks, companies, etc.)"""
        text_lower = text.lower()
        
        detected_stocks = []
        detected_companies = []
        detected_people = []
        detected_locations = []
        
        # Extract stock symbols
        for symbol, keywords in self.stock_symbols.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if symbol not in detected_stocks:
                        detected_stocks.append(symbol)
                        
        # Extract people names (simple pattern matching)
        people_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b'
        people_matches = re.findall(people_pattern, text)
        detected_people = [name for name in people_matches if len(name) > 5]
        
        # Extract locations
        location_pattern = r'\b(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad|India|USA|US|China|Europe|Asia)\b'
        location_matches = re.findall(location_pattern, text, re.IGNORECASE)
        detected_locations = list(set(location_matches))
        
        return {
            'stocks': detected_stocks,
            'companies': detected_companies,
            'people': detected_people[:10],  # Limit to top 10
            'locations': detected_locations
        }
    
    def calculate_market_impact(self, text: str) -> Dict[str, Any]:
        """Calculate potential market impact"""
        text_lower = text.lower()
        impact_score = 0
        impact_factors = []
        
        # High impact keywords
        for keyword in self.financial_keywords['market_impact']['high']:
            if keyword in text_lower:
                impact_score += 3
                impact_factors.append(f"High impact: {keyword}")
                
        # Medium impact keywords
        for keyword in self.financial_keywords['market_impact']['medium']:
            if keyword in text_lower:
                impact_score += 2
                impact_factors.append(f"Medium impact: {keyword}")
                
        # Low impact keywords
        for keyword in self.financial_keywords['market_impact']['low']:
            if keyword in text_lower:
                impact_score += 1
                impact_factors.append(f"Low impact: {keyword}")
        
        # Normalize impact score (0-10)
        normalized_impact = min(10, impact_score)
        
        # Determine impact level
        if normalized_impact >= 7:
            impact_level = 'High'
        elif normalized_impact >= 4:
            impact_level = 'Medium'
        else:
            impact_level = 'Low'
            
        return {
            'score': normalized_impact,
            'level': impact_level,
            'factors': impact_factors[:5]  # Top 5 factors
        }
    
    def generate_recommendations(self, sentiment_results: List[SentimentResult], 
                               entities: Dict[str, List[str]], 
                               market_impact: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        # Calculate consensus sentiment
        if sentiment_results:
            avg_score = np.mean([r.score for r in sentiment_results if r])
            avg_confidence = np.mean([r.confidence for r in sentiment_results if r])
            
            # Stock-specific recommendations
            if entities['stocks']:
                if avg_score > 0.2 and avg_confidence > 0.7:
                    recommendations.append(f"Consider monitoring {entities['stocks'][0]} for potential buying opportunities")
                elif avg_score < -0.2 and avg_confidence > 0.7:
                    recommendations.append(f"Exercise caution with {entities['stocks'][0]} positions")
                    
            # Market impact recommendations
            if market_impact['score'] > 6:
                recommendations.append("Monitor broader market indices for potential impact")
                
            # Confidence-based recommendations
            if avg_confidence < 0.6:
                recommendations.append("Seek additional sources to confirm sentiment analysis")
                
            # General recommendations
            recommendations.append("Cross-reference with technical analysis before making trading decisions")
            
            if market_impact['level'] == 'High':
                recommendations.append("Consider position sizing adjustments due to high market impact potential")
                
        return recommendations
    
    def analyze_comprehensive(self, text: str) -> AnalysisResult:
        """Perform comprehensive analysis using all available methods"""
        
        # Sentiment analysis using multiple methods
        sentiment_results = []
        
        vader_result = self.analyze_with_vader(text)
        if vader_result:
            sentiment_results.append(vader_result)
            
        textblob_result = self.analyze_with_textblob(text)
        if textblob_result:
            sentiment_results.append(textblob_result)
            
        finbert_result = self.analyze_with_finbert(text)
        if finbert_result:
            sentiment_results.append(finbert_result)
            
        bert_result = self.analyze_with_bert(text)
        if bert_result:
            sentiment_results.append(bert_result)
        
        # Calculate consensus sentiment
        if sentiment_results:
            scores = [r.score for r in sentiment_results]
            confidences = [r.confidence for r in sentiment_results]
            
            consensus_score = np.mean(scores)
            consensus_confidence = np.mean(confidences)
            
            if consensus_score > 0.1:
                consensus_label = 'positive'
            elif consensus_score < -0.1:
                consensus_label = 'negative'
            else:
                consensus_label = 'neutral'
                
            consensus_sentiment = {
                'label': consensus_label,
                'score': float(consensus_score),
                'confidence': float(consensus_confidence),
                'method_count': len(sentiment_results),
                'agreement': float(np.std(scores))  # Lower std = higher agreement
            }
        else:
            consensus_sentiment = {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'method_count': 0,
                'agreement': 0.0
            }
        
        # Entity extraction
        spacy_entities = self.extract_entities_spacy(text)
        financial_entities = self.extract_financial_entities(text)
        
        # Keyword extraction
        text_lower = text.lower()
        found_keywords = {
            'bullish': [kw for kw in self.financial_keywords['bullish'] if kw in text_lower],
            'bearish': [kw for kw in self.financial_keywords['bearish'] if kw in text_lower],
            'neutral': [kw for kw in self.financial_keywords['neutral'] if kw in text_lower]
        }
        
        # Market impact analysis
        market_impact = self.calculate_market_impact(text)
        
        # Financial metrics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        financial_metrics = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': max(1, min(10, 10 - (avg_sentence_length - 15) * 0.2)),
            'financial_keyword_density': len(found_keywords['bullish'] + found_keywords['bearish']) / max(word_count, 1) * 100
        }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            sentiment_results, financial_entities, market_impact
        )
        
        return AnalysisResult(
            text=text,
            timestamp=datetime.now().isoformat(),
            sentiment_results=sentiment_results,
            consensus_sentiment=consensus_sentiment,
            entities=spacy_entities,
            keywords=found_keywords,
            financial_metrics=financial_metrics,
            market_indicators={
                'detected_stocks': financial_entities['stocks'],
                'detected_people': financial_entities['people'],
                'detected_locations': financial_entities['locations'],
                'market_impact': market_impact
            },
            recommendations=recommendations
        )

# Flask API
if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)
    
    # Initialize analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'available_methods': {
                'vader': VADER_AVAILABLE,
                'textblob': TEXTBLOB_AVAILABLE,
                'spacy': SPACY_AVAILABLE,
                'finbert': FINBERT_AVAILABLE,
                'bert': BERT_AVAILABLE
            }
        })
    
    @app.route('/analyze', methods=['POST'])
    def analyze_sentiment():
        """Main sentiment analysis endpoint"""
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({'error': 'No text provided'}), 400
                
            text = data['text'].strip()
            
            if not text:
                return jsonify({'error': 'Empty text provided'}), 400
                
            if len(text) > 10000:  # Limit text length
                return jsonify({'error': 'Text too long (max 10000 characters)'}), 400
            
            # Perform analysis
            result = analyzer.analyze_comprehensive(text)
            
            # Convert to dict for JSON serialization
            result_dict = asdict(result)
            
            return jsonify({
                'success': True,
                'result': result_dict
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/methods', methods=['GET'])
    def get_available_methods():
        """Get available analysis methods"""
        return jsonify({
            'methods': {
                'VADER': {
                    'available': VADER_AVAILABLE,
                    'description': 'Valence Aware Dictionary and sEntiment Reasoner - Rule-based sentiment analysis'
                },
                'TextBlob': {
                    'available': TEXTBLOB_AVAILABLE,
                    'description': 'Simple API for diving into common NLP tasks'
                },
                'spaCy': {
                    'available': SPACY_AVAILABLE,
                    'description': 'Industrial-strength Natural Language Processing'
                },
                'FinBERT': {
                    'available': FINBERT_AVAILABLE,
                    'description': 'BERT model fine-tuned for financial sentiment analysis'
                },
                'BERT': {
                    'available': BERT_AVAILABLE,
                    'description': 'General BERT model for sentiment analysis'
                }
            }
        })
    
    if __name__ == '__main__':
        print("Starting Advanced Sentiment Analysis API...")
        print(f"Available methods: VADER={VADER_AVAILABLE}, TextBlob={TEXTBLOB_AVAILABLE}, spaCy={SPACY_AVAILABLE}, FinBERT={FINBERT_AVAILABLE}, BERT={BERT_AVAILABLE}")
        app.run(host='0.0.0.0', port=5000, debug=True)

else:
    print("Flask not available. Cannot start API server.")
    print("Install with: pip install flask flask-cors")