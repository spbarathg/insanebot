"""
Real AI Coordinator - Production-Ready AI System

This module replaces the mock AI components with real machine learning models:
- Online learning for continuous adaptation
- Local LLM for market reasoning
- Real sentiment analysis
- Pattern recognition with actual ML
- Ensemble decision making

Integrates seamlessly with the existing Titan Shield and agent hierarchy.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import pickle
import os
from pathlib import Path

# Online learning framework
try:
    from river import linear_model, preprocessing, compose, metrics, anomaly
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("River not available - install with: pip install river")

# Local LLM
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available - install with: pip install ollama")

# Sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - install with: pip install transformers torch")

logger = logging.getLogger(__name__)

@dataclass
class AIDecision:
    """AI decision with confidence and reasoning"""
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    factors: Dict[str, float]
    timestamp: float
    model_version: str

@dataclass
class LearningUpdate:
    """Learning update from trade outcome"""
    features: Dict[str, float]
    outcome: Dict[str, Any]
    profit_loss: float
    success: bool
    timestamp: float

class OnlineLearningEngine:
    """Continuously learning trading model using River framework"""
    
    def __init__(self):
        self.available = RIVER_AVAILABLE
        if not self.available:
            logger.warning("OnlineLearningEngine disabled - River not available")
            return
            
        # Create online learning pipeline
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(l2=0.1)
        )
        
        # Anomaly detection for unusual market conditions
        self.anomaly_detector = anomaly.HalfSpaceTrees(
            n_trees=10,
            height=8,
            window_size=250
        )
        
        # Performance tracking
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        
        # Learning history
        self.learning_history = deque(maxlen=1000)
        self.feature_importance = {}
        self.model_version = "1.0.0"
        
        logger.info("OnlineLearningEngine initialized")
    
    def extract_features(self, market_data: Dict) -> Dict[str, float]:
        """Extract numerical features from market data"""
        try:
            features = {}
            
            # Price features
            current_price = market_data.get('price', 0)
            features['price_usd'] = float(current_price)
            features['price_change_1h'] = float(market_data.get('price_change_1h', 0))
            features['price_change_24h'] = float(market_data.get('price_change_24h', 0))
            
            # Volume features
            volume_24h = market_data.get('volume_24h', 0)
            features['volume_24h'] = float(volume_24h)
            features['volume_change'] = float(market_data.get('volume_change', 0))
            
            # Market cap and liquidity
            market_cap = market_data.get('market_cap', 0)
            liquidity = market_data.get('liquidity', 0)
            features['market_cap'] = float(market_cap)
            features['liquidity'] = float(liquidity)
            features['liquidity_ratio'] = float(liquidity / market_cap if market_cap > 0 else 0)
            
            # Holder metrics
            holders = market_data.get('holders', 0)
            features['holder_count'] = float(holders)
            features['holder_concentration'] = float(market_data.get('top_holder_percentage', 0))
            
            # Technical indicators
            features['rsi'] = float(market_data.get('rsi', 50))
            features['volatility'] = float(market_data.get('volatility', 0))
            
            # Time-based features
            current_time = time.time()
            features['hour_of_day'] = float((current_time % 86400) / 3600)
            features['day_of_week'] = float((current_time // 86400) % 7)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def predict(self, market_data: Dict) -> Tuple[float, Dict]:
        """Make prediction with confidence"""
        if not self.available:
            return 0.5, {}
            
        try:
            features = self.extract_features(market_data)
            if not features:
                return 0.5, {}
            
            # Get prediction probability
            prediction = self.model.predict_proba_one(features)
            
            # Check for anomalies
            anomaly_score = self.anomaly_detector.score_one(features)
            is_anomaly = anomaly_score > 0.5
            
            # Adjust confidence based on anomaly detection
            confidence = prediction.get(True, 0.5)
            if is_anomaly:
                confidence *= 0.7  # Reduce confidence for anomalous conditions
            
            return confidence, {
                'prediction_raw': prediction,
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'feature_count': len(features)
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.5, {}
    
    def learn_from_outcome(self, market_data: Dict, outcome: Dict) -> bool:
        """Learn from trade outcome"""
        if not self.available:
            return False
            
        try:
            features = self.extract_features(market_data)
            if not features:
                return False
            
            # Determine success based on outcome
            success = outcome.get('success', False)
            profit_loss = outcome.get('profit_loss_percent', 0)
            
            # Convert to binary classification
            target = success and profit_loss > 1.0  # Profitable trade
            
            # Update models
            prediction = self.model.predict_proba_one(features)
            self.model.learn_one(features, target)
            self.anomaly_detector.learn_one(features)
            
            # Update metrics
            self.accuracy.update(target, prediction.get(True, 0.5) > 0.5)
            self.precision.update(target, prediction.get(True, 0.5) > 0.5)
            self.recall.update(target, prediction.get(True, 0.5) > 0.5)
            
            # Store learning update
            learning_update = LearningUpdate(
                features=features,
                outcome=outcome,
                profit_loss=profit_loss,
                success=target,
                timestamp=time.time()
            )
            self.learning_history.append(learning_update)
            
            logger.debug(f"Model learned from outcome: success={target}, accuracy={self.accuracy.get():.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error in learning: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current model performance"""
        if not self.available:
            return {}
            
        return {
            'accuracy': self.accuracy.get(),
            'precision': self.precision.get(),
            'recall': self.recall.get(),
            'learning_samples': len(self.learning_history),
            'model_version': self.model_version
        }

class LocalLLMEngine:
    """Local LLM for market reasoning and analysis"""
    
    def __init__(self):
        self.available = OLLAMA_AVAILABLE
        self.client = None
        self.model_name = "llama2:7b"  # Default model
        
        if self.available:
            try:
                self.client = ollama.Client()
                # Test if model is available
                self._test_model()
                logger.info(f"LocalLLMEngine initialized with {self.model_name}")
            except Exception as e:
                logger.warning(f"LocalLLM failed to initialize: {e}")
                self.available = False
        else:
            logger.warning("LocalLLMEngine disabled - Ollama not available")
    
    def _test_model(self):
        """Test if the model is available and working"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt="Test",
                stream=False
            )
            return True
        except Exception as e:
            logger.warning(f"Model {self.model_name} not available, trying alternatives...")
            
            # Try alternative models
            alternatives = ["llama2:7b", "mistral:7b", "orca-mini:3b"]
            for model in alternatives:
                try:
                    response = self.client.generate(
                        model=model,
                        prompt="Test",
                        stream=False
                    )
                    self.model_name = model
                    logger.info(f"Using alternative model: {model}")
                    return True
                except:
                    continue
            
            raise Exception("No compatible models found")
    
    async def analyze_market_context(self, market_data: Dict, sentiment_score: float) -> Dict[str, Any]:
        """Analyze market context with LLM reasoning"""
        if not self.available:
            return {
                'reasoning': 'LLM not available - using fallback logic',
                'confidence': 0.5,
                'action': 'hold'
            }
        
        try:
            # Prepare context for LLM
            context = self._prepare_market_context(market_data, sentiment_score)
            
            prompt = f"""
You are an expert cryptocurrency trader analyzing a memecoin opportunity. Based on the data below, provide a trading decision.

Market Data:
{context}

Provide your analysis in this exact format:
ACTION: [buy/sell/hold]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]

Focus on risk management and only recommend 'buy' if there's strong evidence of opportunity.
"""

            # Get LLM response
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._get_llm_response, prompt
            )
            
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                'reasoning': f'LLM error: {str(e)}',
                'confidence': 0.3,
                'action': 'hold'
            }
    
    def _prepare_market_context(self, market_data: Dict, sentiment_score: float) -> str:
        """Prepare market context for LLM"""
        return f"""
Price: ${market_data.get('price', 0):.6f}
24h Change: {market_data.get('price_change_24h', 0):.2f}%
Volume 24h: ${market_data.get('volume_24h', 0):,.0f}
Market Cap: ${market_data.get('market_cap', 0):,.0f}
Liquidity: ${market_data.get('liquidity', 0):,.0f}
Holders: {market_data.get('holders', 0):,}
Sentiment Score: {sentiment_score:.3f}
Top Holder %: {market_data.get('top_holder_percentage', 0):.1f}%
"""
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM (synchronous)"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'max_tokens': 200
                }
            )
            return response.get('response', '')
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "ERROR: Could not generate response"
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            lines = response.strip().split('\n')
            result = {
                'action': 'hold',
                'confidence': 0.5,
                'reasoning': 'Could not parse LLM response'
            }
            
            for line in lines:
                line = line.strip()
                if line.startswith('ACTION:'):
                    action = line.split(':', 1)[1].strip().lower()
                    if action in ['buy', 'sell', 'hold']:
                        result['action'] = action
                
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        result['confidence'] = max(0.0, min(1.0, confidence))
                    except:
                        pass
                
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                'action': 'hold',
                'confidence': 0.3,
                'reasoning': f'Parse error: {str(e)}'
            }

class SentimentEngine:
    """Real sentiment analysis using FinBERT or similar models"""
    
    def __init__(self):
        self.available = TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
        
        if self.available:
            try:
                # Try to load FinBERT for financial sentiment
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                logger.info("SentimentEngine initialized with FinBERT")
            except Exception as e:
                logger.warning(f"FinBERT not available, using distilbert: {e}")
                try:
                    # Fallback to general sentiment model
                    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    logger.info("SentimentEngine initialized with DistilBERT")
                except Exception as e2:
                    logger.error(f"No sentiment models available: {e2}")
                    self.available = False
        else:
            logger.warning("SentimentEngine disabled - Transformers not available")
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, Dict]:
        """Analyze sentiment of text (news, social media, etc.)"""
        if not self.available or not text:
            return 0.0, {}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to sentiment score (-1 to 1)
            if predictions.shape[1] == 3:  # FinBERT: positive, negative, neutral
                positive, negative, neutral = predictions[0].tolist()
                sentiment_score = positive - negative
            else:  # Binary: negative, positive
                negative, positive = predictions[0].tolist()
                sentiment_score = positive - negative
            
            return sentiment_score, {
                'positive_prob': positive,
                'negative_prob': negative,
                'confidence': max(positive, negative)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0, {}
    
    def analyze_market_sentiment(self, market_data: Dict) -> float:
        """Analyze overall market sentiment from various signals"""
        try:
            sentiment_signals = []
            
            # Price momentum sentiment
            price_change_24h = market_data.get('price_change_24h', 0)
            price_sentiment = np.tanh(price_change_24h / 10)  # Normalize to [-1, 1]
            sentiment_signals.append(price_sentiment * 0.3)
            
            # Volume sentiment
            volume_change = market_data.get('volume_change', 0)
            volume_sentiment = np.tanh(volume_change / 50)
            sentiment_signals.append(volume_sentiment * 0.2)
            
            # Holder growth sentiment
            holders = market_data.get('holders', 0)
            if holders > 1000:
                holder_sentiment = 0.3
            elif holders > 500:
                holder_sentiment = 0.1
            elif holders < 100:
                holder_sentiment = -0.2
            else:
                holder_sentiment = 0.0
            sentiment_signals.append(holder_sentiment)
            
            # Liquidity sentiment
            market_cap = market_data.get('market_cap', 0)
            liquidity = market_data.get('liquidity', 0)
            if market_cap > 0:
                liquidity_ratio = liquidity / market_cap
                liquidity_sentiment = np.tanh((liquidity_ratio - 0.05) * 20)
                sentiment_signals.append(liquidity_sentiment * 0.2)
            
            # Text sentiment if available
            description = market_data.get('description', '')
            if description:
                text_sentiment, _ = self.analyze_text_sentiment(description)
                sentiment_signals.append(text_sentiment * 0.3)
            
            # Combine all signals
            overall_sentiment = sum(sentiment_signals)
            return max(-1.0, min(1.0, overall_sentiment))
            
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {e}")
            return 0.0

class RealAICoordinator:
    """Production-ready AI coordinator with real machine learning"""
    
    def __init__(self):
        self.learning_engine = OnlineLearningEngine()
        self.llm_engine = LocalLLMEngine()
        self.sentiment_engine = SentimentEngine()
        
        # Decision history for analysis
        self.decision_history = deque(maxlen=1000)
        self.model_performance = {}
        
        # Model weights for ensemble
        self.ensemble_weights = {
            'online_learning': 0.4,
            'llm_reasoning': 0.3,
            'sentiment_analysis': 0.2,
            'risk_adjustment': 0.1
        }
        
        logger.info("RealAICoordinator initialized with all engines")
    
    async def analyze_trading_opportunity(self, token_address: str, market_data: Dict) -> AIDecision:
        """Comprehensive AI analysis of trading opportunity"""
        try:
            start_time = time.time()
            
            # Get predictions from all engines
            ml_confidence, ml_details = self.learning_engine.predict(market_data)
            sentiment_score = self.sentiment_engine.analyze_market_sentiment(market_data)
            llm_analysis = await self.llm_engine.analyze_market_context(market_data, sentiment_score)
            
            # Risk adjustment based on market conditions
            risk_score = self._calculate_risk_score(market_data)
            
            # Ensemble decision making
            decision = self._make_ensemble_decision(
                ml_confidence=ml_confidence,
                sentiment_score=sentiment_score,
                llm_analysis=llm_analysis,
                risk_score=risk_score,
                market_data=market_data
            )
            
            # Create comprehensive AI decision
            ai_decision = AIDecision(
                action=decision['action'],
                confidence=decision['confidence'],
                reasoning=decision['reasoning'],
                factors={
                    'ml_confidence': ml_confidence,
                    'sentiment_score': sentiment_score,
                    'llm_confidence': llm_analysis['confidence'],
                    'risk_score': risk_score,
                    'ensemble_score': decision['ensemble_score']
                },
                timestamp=time.time(),
                model_version="real_ai_v1.0"
            )
            
            # Store decision for analysis
            self.decision_history.append(ai_decision)
            
            processing_time = time.time() - start_time
            logger.debug(f"AI analysis completed in {processing_time:.3f}s: {decision['action']} ({decision['confidence']:.3f})")
            
            return ai_decision
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            # Fallback conservative decision
            return AIDecision(
                action="hold",
                confidence=0.1,
                reasoning=f"AI analysis failed: {str(e)}",
                factors={},
                timestamp=time.time(),
                model_version="fallback"
            )
    
    def _calculate_risk_score(self, market_data: Dict) -> float:
        """Calculate risk score from market data"""
        try:
            risk_factors = []
            
            # Market cap risk (very small caps are risky)
            market_cap = market_data.get('market_cap', 0)
            if market_cap < 100000:  # <$100k
                risk_factors.append(0.8)
            elif market_cap < 1000000:  # <$1M
                risk_factors.append(0.5)
            else:
                risk_factors.append(0.2)
            
            # Liquidity risk
            liquidity = market_data.get('liquidity', 0)
            if liquidity < 10000:  # <$10k liquidity
                risk_factors.append(0.9)
            elif liquidity < 50000:  # <$50k liquidity
                risk_factors.append(0.6)
            else:
                risk_factors.append(0.3)
            
            # Holder concentration risk
            top_holder_pct = market_data.get('top_holder_percentage', 0)
            if top_holder_pct > 50:
                risk_factors.append(0.8)
            elif top_holder_pct > 20:
                risk_factors.append(0.5)
            else:
                risk_factors.append(0.2)
            
            # Volatility risk
            volatility = market_data.get('volatility', 0)
            volatility_risk = min(1.0, volatility / 100)  # Normalize
            risk_factors.append(volatility_risk)
            
            return sum(risk_factors) / len(risk_factors)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.8  # High risk as fallback
    
    def _make_ensemble_decision(self, ml_confidence: float, sentiment_score: float, 
                              llm_analysis: Dict, risk_score: float, market_data: Dict) -> Dict:
        """Make final decision using ensemble of all models"""
        try:
            # Convert inputs to buy signals (0-1)
            ml_signal = ml_confidence
            sentiment_signal = (sentiment_score + 1) / 2  # Convert from [-1,1] to [0,1]
            
            # LLM signal
            llm_action = llm_analysis['action']
            if llm_action == 'buy':
                llm_signal = llm_analysis['confidence']
            elif llm_action == 'sell':
                llm_signal = -llm_analysis['confidence']
            else:  # hold
                llm_signal = 0.5
            
            # Risk adjustment (higher risk reduces confidence)
            risk_adjustment = 1.0 - risk_score
            
            # Weighted ensemble
            ensemble_score = (
                ml_signal * self.ensemble_weights['online_learning'] +
                sentiment_signal * self.ensemble_weights['sentiment_analysis'] +
                llm_signal * self.ensemble_weights['llm_reasoning'] +
                risk_adjustment * self.ensemble_weights['risk_adjustment']
            )
            
            # Make final decision
            if ensemble_score > 0.7:
                action = "buy"
                confidence = ensemble_score
            elif ensemble_score < 0.3:
                action = "sell"
                confidence = 1.0 - ensemble_score
            else:
                action = "hold"
                confidence = 0.5
            
            # Adjust confidence based on risk
            final_confidence = confidence * risk_adjustment
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                ml_confidence, sentiment_score, llm_analysis, risk_score, ensemble_score
            )
            
            return {
                'action': action,
                'confidence': final_confidence,
                'ensemble_score': ensemble_score,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble decision: {e}")
            return {
                'action': 'hold',
                'confidence': 0.3,
                'ensemble_score': 0.5,
                'reasoning': f'Decision error: {str(e)}'
            }
    
    def _generate_reasoning(self, ml_confidence: float, sentiment_score: float, 
                          llm_analysis: Dict, risk_score: float, ensemble_score: float) -> str:
        """Generate human-readable reasoning for the decision"""
        try:
            reasoning_parts = []
            
            # ML component
            if ml_confidence > 0.6:
                reasoning_parts.append(f"ML model shows high confidence ({ml_confidence:.2f})")
            elif ml_confidence < 0.4:
                reasoning_parts.append(f"ML model suggests caution ({ml_confidence:.2f})")
            
            # Sentiment component
            if sentiment_score > 0.3:
                reasoning_parts.append(f"Positive market sentiment ({sentiment_score:.2f})")
            elif sentiment_score < -0.3:
                reasoning_parts.append(f"Negative market sentiment ({sentiment_score:.2f})")
            
            # LLM component
            if llm_analysis['action'] != 'hold':
                reasoning_parts.append(f"LLM recommends {llm_analysis['action']}: {llm_analysis['reasoning'][:50]}...")
            
            # Risk component
            if risk_score > 0.7:
                reasoning_parts.append("High risk environment detected")
            elif risk_score < 0.3:
                reasoning_parts.append("Low risk environment")
            
            # Final ensemble
            reasoning_parts.append(f"Ensemble score: {ensemble_score:.2f}")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            return f"Reasoning generation failed: {str(e)}"
    
    async def learn_from_trade_outcome(self, trade_data: Dict, outcome: Dict) -> bool:
        """Learn from trade outcome across all models"""
        try:
            # Update online learning model
            learning_success = self.learning_engine.learn_from_outcome(
                trade_data.get('market_data', {}),
                outcome
            )
            
            # Update performance tracking
            if learning_success:
                self._update_performance_metrics(outcome)
            
            logger.debug(f"Learning update completed: success={learning_success}")
            return learning_success
            
        except Exception as e:
            logger.error(f"Error in learning update: {e}")
            return False
    
    def _update_performance_metrics(self, outcome: Dict):
        """Update performance tracking metrics"""
        try:
            if 'model_performance' not in self.model_performance:
                self.model_performance = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'total_profit': 0.0,
                    'accuracy': 0.0
                }
            
            self.model_performance['total_trades'] += 1
            
            if outcome.get('success', False):
                self.model_performance['successful_trades'] += 1
            
            profit_loss = outcome.get('profit_loss_percent', 0)
            self.model_performance['total_profit'] += profit_loss
            
            # Update accuracy
            self.model_performance['accuracy'] = (
                self.model_performance['successful_trades'] / 
                self.model_performance['total_trades']
            )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get comprehensive AI system status"""
        return {
            'engines': {
                'online_learning': self.learning_engine.available,
                'local_llm': self.llm_engine.available,
                'sentiment_analysis': self.sentiment_engine.available
            },
            'performance': self.model_performance,
            'ml_metrics': self.learning_engine.get_performance_metrics(),
            'decision_history_count': len(self.decision_history),
            'ensemble_weights': self.ensemble_weights
        }
    
    async def initialize(self) -> bool:
        """Initialize the AI system"""
        try:
            logger.info("Initializing Real AI Coordinator...")
            
            # Test all engines
            test_data = {
                'price': 0.001,
                'market_cap': 100000,
                'volume_24h': 50000,
                'liquidity': 25000,
                'holders': 500,
                'price_change_24h': 5.0
            }
            
            # Test decision making
            decision = await self.analyze_trading_opportunity("test", test_data)
            logger.info(f"AI system test successful: {decision.action} ({decision.confidence:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"AI system initialization failed: {e}")
            return False

# Convenience function for creating the real AI coordinator
def create_real_ai_coordinator() -> RealAICoordinator:
    """Create and return a real AI coordinator instance"""
    return RealAICoordinator() 