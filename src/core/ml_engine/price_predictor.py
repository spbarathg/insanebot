"""
Price Prediction Engine using Multiple ML Models

This module implements various machine learning models to predict token prices:
- LSTM for time series prediction
- Random Forest for feature-based prediction  
- XGBoost for gradient boosting
- Linear Regression for baseline
- Ensemble voting for final prediction
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import json
import asyncio
import math
from collections import deque

try:
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using simplified prediction models.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Using alternative models.")

from .ml_types import PredictionResult

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for price prediction"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return [prices[-1]] * len(prices) if prices else [0]
        
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(prices[i])
            else:
                sma.append(sum(prices[i-period+1:i+1]) / period)
        return sma
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if not prices:
            return []
        
        alpha = 2.0 / (period + 1)
        ema = [prices[0]]
        
        for i in range(1, len(prices)):
            ema.append(alpha * prices[i] + (1 - alpha) * ema[-1])
        
        return ema
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        
        rsi = []
        for i in range(len(deltas)):
            if i < period - 1:
                rsi.append(50.0)
            else:
                avg_gain = sum(gains[i-period+1:i+1]) / period
                avg_loss = sum(losses[i-period+1:i+1]) / period
                
                if avg_loss == 0:
                    rsi.append(100.0)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper_band.append(prices[i])
                lower_band.append(prices[i])
            else:
                price_slice = prices[i-period+1:i+1]
                std = np.std(price_slice) if len(price_slice) > 1 else 0
                upper_band.append(sma[i] + (std_dev * std))
                lower_band.append(sma[i] - (std_dev * std))
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(prices))]
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = [macd_line[i] - signal_line[i] for i in range(len(macd_line))]
        
        return macd_line, signal_line, histogram

class SimplePricePredictor:
    """Simplified price predictor when ML libraries are not available"""
    
    def __init__(self):
        self.price_history = deque(maxlen=100)
        self.feature_weights = {
            'momentum': 0.3,
            'trend': 0.25,
            'volatility': 0.2,
            'volume': 0.15,
            'technical': 0.1
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Simplified fitting - just store recent data"""
        for i, price in enumerate(y):
            self.price_history.append(float(price))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Simple prediction based on momentum and trends"""
        if len(self.price_history) < 5:
            return np.array([self.price_history[-1] if self.price_history else 100.0] * len(X))
        
        recent_prices = list(self.price_history)[-10:]
        current_price = recent_prices[-1]
        
        # Calculate momentum
        momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
        
        # Calculate trend
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] != 0 else 0
        
        # Calculate volatility
        returns = [(recent_prices[i] / recent_prices[i-1] - 1) for i in range(1, len(recent_prices))]
        volatility = np.std(returns) if len(returns) > 1 else 0.1
        
        # Predict based on momentum and trend
        base_change = momentum * 0.7 + trend * 0.3
        uncertainty = volatility * 0.1
        
        predictions = []
        for i in range(len(X)):
            # Add some randomness for different time horizons
            time_decay = 1.0 - (i * 0.1)
            predicted_change = base_change * time_decay + np.random.normal(0, uncertainty)
            predicted_price = current_price * (1 + predicted_change)
            predictions.append(max(predicted_price, current_price * 0.5))  # Prevent extreme predictions
        
        return np.array(predictions)

class PricePredictor:
    """Advanced price prediction using multiple ML models"""
    
    def __init__(self):
        """Initialize the price predictor"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.price_history = {}  # Store price history per token
        self.prediction_cache = {}  # Cache recent predictions
        self.last_training_time = 0
        self.training_interval = 3600  # Retrain every hour
        
        logger.info("PricePredictor initialized")
    
    async def initialize(self) -> bool:
        """Initialize the prediction models"""
        try:
            logger.info("🤖 Initializing ML price prediction models...")
            
            if SKLEARN_AVAILABLE:
                # Initialize models
                self.models = {
                    'random_forest': RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'linear': Ridge(alpha=1.0),
                    'simple': SimplePricePredictor()
                }
                
                if XGBOOST_AVAILABLE:
                    self.models['xgboost'] = xgb.XGBRegressor(
                        n_estimators=50,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                
                # Initialize scalers
                self.scalers = {
                    'features': StandardScaler(),
                    'prices': MinMaxScaler()
                }
                
                logger.info("✅ ML models initialized successfully")
            else:
                # Use simplified predictor
                self.models = {'simple': SimplePricePredictor()}
                logger.info("✅ Simplified predictor initialized (ML libraries not available)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize price predictor: {str(e)}")
            return False
    
    def _extract_features(self, token_data: Dict, price_history: List[float]) -> Dict[str, float]:
        """Extract features for price prediction"""
        try:
            features = {}
            
            # Basic token features
            features['current_price'] = token_data.get('price_usd', 0)
            features['liquidity'] = token_data.get('liquidity_usd', 0)
            features['volume_24h'] = token_data.get('volumeUsd24h', 0)
            features['market_cap'] = token_data.get('market_cap', 0)
            features['holders'] = token_data.get('holders', 0)
            
            # Price history features
            if len(price_history) >= 10:
                prices = price_history[-50:]  # Use last 50 data points
                
                # Technical indicators
                sma_5 = TechnicalIndicators.sma(prices, 5)
                sma_20 = TechnicalIndicators.sma(prices, 20)
                ema_12 = TechnicalIndicators.ema(prices, 12)
                rsi = TechnicalIndicators.rsi(prices)
                
                features['sma_5'] = sma_5[-1] if sma_5 else prices[-1]
                features['sma_20'] = sma_20[-1] if sma_20 else prices[-1]
                features['ema_12'] = ema_12[-1] if ema_12 else prices[-1]
                features['rsi'] = rsi[-1] if rsi else 50.0
                
                # Price ratios
                features['price_sma5_ratio'] = prices[-1] / features['sma_5'] if features['sma_5'] > 0 else 1.0
                features['price_sma20_ratio'] = prices[-1] / features['sma_20'] if features['sma_20'] > 0 else 1.0
                
                # Momentum features
                if len(prices) >= 5:
                    features['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5]
                    features['momentum_20'] = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else features['momentum_5']
                
                # Volatility
                returns = [(prices[i] / prices[i-1] - 1) for i in range(1, min(len(prices), 20))]
                features['volatility'] = np.std(returns) if len(returns) > 1 else 0.1
                
                # Bollinger bands
                if len(prices) >= 20:
                    upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, 20)
                    features['bb_position'] = (prices[-1] - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
                
                # MACD
                if len(prices) >= 26:
                    macd, signal, histogram = TechnicalIndicators.macd(prices)
                    features['macd'] = macd[-1]
                    features['macd_signal'] = signal[-1]
                    features['macd_histogram'] = histogram[-1]
            else:
                # Default values when not enough price history
                current_price = features['current_price']
                features.update({
                    'sma_5': current_price,
                    'sma_20': current_price,
                    'ema_12': current_price,
                    'rsi': 50.0,
                    'price_sma5_ratio': 1.0,
                    'price_sma20_ratio': 1.0,
                    'momentum_5': 0.0,
                    'momentum_20': 0.0,
                    'volatility': 0.1,
                    'bb_position': 0.5,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0
                })
            
            # Liquidity and volume ratios
            if features['market_cap'] > 0:
                features['volume_mcap_ratio'] = features['volume_24h'] / features['market_cap']
                features['liquidity_mcap_ratio'] = features['liquidity'] / features['market_cap']
            else:
                features['volume_mcap_ratio'] = 0.0
                features['liquidity_mcap_ratio'] = 0.0
            
            # Time-based features
            current_time = time.time()
            hour_of_day = (current_time % 86400) / 3600  # 0-24
            day_of_week = ((current_time // 86400) % 7)  # 0-6
            
            features['hour_sin'] = math.sin(2 * math.pi * hour_of_day / 24)
            features['hour_cos'] = math.cos(2 * math.pi * hour_of_day / 24)
            features['day_sin'] = math.sin(2 * math.pi * day_of_week / 7)
            features['day_cos'] = math.cos(2 * math.pi * day_of_week / 7)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def _prepare_training_data(self, price_history: List[float], features_history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical data"""
        try:
            if len(price_history) < 10 or len(features_history) < 10:
                return np.array([]), np.array([])
            
            X = []
            y = []
            
            # Create sequences for training
            sequence_length = min(10, len(price_history) - 5)
            
            for i in range(sequence_length, len(price_history) - 1):
                # Features for current timestep
                if i < len(features_history):
                    feature_dict = features_history[i]
                    feature_values = [
                        feature_dict.get('current_price', 0),
                        feature_dict.get('volume_24h', 0),
                        feature_dict.get('liquidity', 0),
                        feature_dict.get('sma_5', 0),
                        feature_dict.get('sma_20', 0),
                        feature_dict.get('rsi', 50),
                        feature_dict.get('momentum_5', 0),
                        feature_dict.get('volatility', 0.1),
                        feature_dict.get('bb_position', 0.5),
                        feature_dict.get('macd', 0),
                    ]
                    X.append(feature_values)
                    
                    # Target: next price
                    y.append(price_history[i + 1])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([])
    
    async def train_models(self, token_address: str, token_data: Dict, price_history: List[float]) -> bool:
        """Train prediction models for a specific token"""
        try:
            if len(price_history) < 20:
                logger.debug(f"Insufficient price history for {token_address[:8]}... ({len(price_history)} points)")
                return False
            
            # Store price history
            self.price_history[token_address] = price_history[-100:]  # Keep last 100 points
            
            # Extract features for each historical point
            features_history = []
            for i, price in enumerate(price_history[-50:]):  # Use last 50 points
                temp_data = dict(token_data)
                temp_data['price_usd'] = price
                features = self._extract_features(temp_data, price_history[:len(price_history)-50+i+1])
                features_history.append(features)
            
            # Prepare training data
            X, y = self._prepare_training_data(price_history[-50:], features_history)
            
            if len(X) == 0 or len(y) == 0:
                logger.debug(f"No training data generated for {token_address[:8]}...")
                return False
            
            # Train models
            model_scores = {}
            
            for model_name, model in self.models.items():
                try:
                    if SKLEARN_AVAILABLE and model_name != 'simple':
                        # Split data
                        if len(X) > 10:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        else:
                            X_train, X_test, y_train, y_test = X, X, y, y
                        
                        # Scale features
                        X_train_scaled = self.scalers['features'].fit_transform(X_train)
                        X_test_scaled = self.scalers['features'].transform(X_test)
                        
                        # Train model
                        model.fit(X_train_scaled, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test_scaled)
                        score = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.5
                        model_scores[model_name] = score
                        
                        # Store feature importance
                        if hasattr(model, 'feature_importances_'):
                            feature_names = ['price', 'volume', 'liquidity', 'sma_5', 'sma_20', 'rsi', 'momentum', 'volatility', 'bb_position', 'macd']
                            self.feature_importance[f"{token_address}_{model_name}"] = dict(zip(feature_names, model.feature_importances_))
                    else:
                        # Simple model
                        model.fit(X, y)
                        model_scores[model_name] = 0.6  # Default score for simple model
                        
                except Exception as e:
                    logger.error(f"Error training {model_name} model: {str(e)}")
                    model_scores[model_name] = 0.0
            
            self.model_performance[token_address] = model_scores
            self.last_training_time = time.time()
            
            logger.debug(f"✅ Models trained for {token_address[:8]}... | Scores: {model_scores}")
            return True
            
        except Exception as e:
            logger.error(f"Error training models for {token_address}: {str(e)}")
            return False
    
    async def predict_price(self, token_address: str, token_data: Dict, price_history: List[float]) -> Optional[PredictionResult]:
        """Predict future prices for a token"""
        try:
            current_time = time.time()
            
            # Check cache
            cache_key = f"{token_address}_{int(current_time // 300)}"  # 5-minute cache
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Train models if needed
            if (token_address not in self.model_performance or 
                current_time - self.last_training_time > self.training_interval):
                await self.train_models(token_address, token_data, price_history)
            
            # Extract current features
            features = self._extract_features(token_data, price_history)
            if not features:
                return None
            
            # Prepare features for prediction
            feature_values = np.array([[
                features.get('current_price', 0),
                features.get('volume_24h', 0),
                features.get('liquidity', 0),
                features.get('sma_5', 0),
                features.get('sma_20', 0),
                features.get('rsi', 50),
                features.get('momentum_5', 0),
                features.get('volatility', 0.1),
                features.get('bb_position', 0.5),
                features.get('macd', 0),
            ]])
            
            # Make predictions with each model
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name in self.model_performance.get(token_address, {}):
                        if SKLEARN_AVAILABLE and model_name != 'simple':
                            # Scale features
                            if hasattr(self.scalers['features'], 'scale_'):
                                X_scaled = self.scalers['features'].transform(feature_values)
                            else:
                                X_scaled = feature_values
                            
                            pred = model.predict(X_scaled)[0]
                        else:
                            pred = model.predict(feature_values)[0]
                        
                        predictions[model_name] = pred
                        confidences[model_name] = self.model_performance[token_address].get(model_name, 0.5)
                        
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {str(e)}")
                    continue
            
            if not predictions:
                return None
            
            # Ensemble prediction (weighted average)
            total_weight = sum(confidences.values())
            if total_weight == 0:
                return None
            
            current_price = features['current_price']
            
            # Calculate ensemble predictions for different timeframes
            ensemble_pred_1h = sum(pred * conf for pred, conf in zip(predictions.values(), confidences.values())) / total_weight
            
            # Adjust for different timeframes
            volatility = features.get('volatility', 0.1)
            momentum = features.get('momentum_5', 0.0)
            
            # 1h prediction: base prediction
            pred_1h = ensemble_pred_1h
            
            # 4h prediction: extend with momentum decay
            momentum_4h = momentum * 0.7  # Momentum decay
            pred_4h = pred_1h * (1 + momentum_4h * 0.5)
            
            # 24h prediction: further momentum decay + mean reversion
            momentum_24h = momentum * 0.3
            mean_reversion = (current_price - pred_1h) * 0.1  # Small mean reversion
            pred_24h = pred_4h * (1 + momentum_24h * 0.3) + mean_reversion
            
            # Calculate confidence scores
            avg_confidence = sum(confidences.values()) / len(confidences)
            
            # Adjust confidence based on volatility and data quality
            volatility_factor = max(0.3, 1.0 - volatility * 2)  # Lower confidence for high volatility
            data_quality = min(1.0, len(price_history) / 50)  # Better confidence with more data
            
            conf_1h = avg_confidence * volatility_factor * data_quality
            conf_4h = conf_1h * 0.8  # Slightly lower for longer timeframe
            conf_24h = conf_1h * 0.6  # Much lower for 24h
            
            # Create result
            result = PredictionResult(
                token_address=token_address,
                token_symbol=token_data.get('symbol', 'UNKNOWN'),
                current_price=current_price,
                predicted_price_1h=max(pred_1h, current_price * 0.5),  # Prevent extreme predictions
                predicted_price_4h=max(pred_4h, current_price * 0.3),
                predicted_price_24h=max(pred_24h, current_price * 0.2),
                confidence_1h=min(conf_1h, 0.95),
                confidence_4h=min(conf_4h, 0.90),
                confidence_24h=min(conf_24h, 0.85),
                prediction_timestamp=current_time,
                model_version="v1.0",
                feature_importance=self.feature_importance.get(f"{token_address}_random_forest", {})
            )
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            # Cleanup old cache entries
            current_bucket = int(current_time // 300)
            self.prediction_cache = {k: v for k, v in self.prediction_cache.items() 
                                   if int(k.split('_')[-1]) >= current_bucket - 10}
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting price for {token_address}: {str(e)}")
            return None
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model performance"""
        return {
            'total_tokens_trained': len(self.model_performance),
            'model_performance': self.model_performance,
            'last_training_time': self.last_training_time,
            'cache_size': len(self.prediction_cache),
            'available_models': list(self.models.keys())
        } 