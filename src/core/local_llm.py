import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import logging
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import json
from ..utils.config import settings
from loguru import logger

class LocalLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = {}
        self.trade_history = []
        self.learning_data = []
        self.performance_metrics = {
            'accuracy': [],
            'profit_loss': [],
            'confidence_scores': [],
            'training_loss': []
        }
        self.min_training_samples = 100
        self.retraining_interval = 1000  # Retrain every 1000 trades

    async def initialize(self):
        """Initialize the local LLM model"""
        try:
            logger.info("Initializing Local LLM...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load existing training data if available
            self._load_training_data()
            
            logger.info("Local LLM initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Local LLM: {str(e)}")
            return False

    def _load_training_data(self):
        """Load existing training data"""
        try:
            with open(settings.TRAINING_DATA_FILE, 'r') as f:
                data = json.load(f)
                self.learning_data = data.get('learning_data', [])
                self.performance_metrics = data.get('performance_metrics', self.performance_metrics)
        except FileNotFoundError:
            logger.info("No existing training data found")
        except Exception as e:
            logger.error(f"Error loading training data: {e}")

    def _save_training_data(self):
        """Save training data"""
        try:
            with open(settings.TRAINING_DATA_FILE, 'w') as f:
                json.dump({
                    'learning_data': self.learning_data,
                    'performance_metrics': self.performance_metrics
                }, f)
        except Exception as e:
            logger.error(f"Error saving training data: {e}")

    def _create_trading_prompt(self, data: Dict) -> str:
        """Create a prompt for trading analysis"""
        return f"""Analyze the following trading opportunity:
Token: {data.get('token_address')}
Current Price: {data.get('price')}
24h Volume: {data.get('volume_24h')}
Liquidity: {data.get('liquidity')}
Market Cap: {data.get('market_cap')}
Sentiment: {data.get('sentiment')}
Twitter Sentiment: {data.get('twitter_sentiment')}
Recent Trades: {data.get('recent_trades')}

Based on the above data and historical patterns, provide a trading decision with confidence score.
Decision should include:
1. Whether to trade (buy/sell/hold)
2. Position size
3. Entry price
4. Stop loss
5. Take profit
6. Confidence score (0-1)
7. Reasoning

Decision:"""

    async def analyze_market(self, data: Dict) -> Dict:
        """Analyze market data and make trading decision"""
        try:
            # Check cache first
            cache_key = f"{data.get('token_address')}_{data.get('price')}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            prompt = self._create_trading_prompt(data)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            decision = self._parse_trading_decision(outputs)
            if decision:
                self.cache[cache_key] = decision
                # Track confidence score
                self.performance_metrics['confidence_scores'].append(decision['confidence'])
            return decision

        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return None

    def _parse_trading_decision(self, outputs) -> Dict:
        """Parse the model's output into a structured decision"""
        try:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract decision components
            decision = {
                'action': self._extract_action(response),
                'position_size': self._extract_position_size(response),
                'entry_price': self._extract_price(response, 'entry'),
                'stop_loss': self._extract_price(response, 'stop loss'),
                'take_profit': self._extract_price(response, 'take profit'),
                'confidence': self._extract_confidence(response),
                'reasoning': self._extract_reasoning(response)
            }
            
            return decision

        except Exception as e:
            logger.error(f"Error parsing trading decision: {str(e)}")
            return None

    def _extract_action(self, response: str) -> str:
        """Extract trading action from response"""
        if 'buy' in response.lower():
            return 'buy'
        elif 'sell' in response.lower():
            return 'sell'
        return 'hold'

    def _extract_position_size(self, response: str) -> float:
        """Extract position size from response"""
        try:
            # Look for position size in response
            # Default to 0.1 if not found
            return 0.1
        except:
            return 0.1

    def _extract_price(self, response: str, price_type: str) -> float:
        """Extract price from response"""
        try:
            # Look for price in response
            # Default to 0 if not found
            return 0.0
        except:
            return 0.0

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        try:
            # Look for confidence score in response
            # Default to 0.5 if not found
            return 0.5
        except:
            return 0.5

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response"""
        try:
            # Extract reasoning section
            return "No reasoning provided"
        except:
            return "No reasoning provided"

    def learn_from_trade(self, trade_result: Dict):
        """Learn from trade results"""
        try:
            self.trade_history.append(trade_result)
            self.learning_data.append({
                'market_conditions': trade_result['market_state'],
                'decision': trade_result['decision'],
                'outcome': trade_result['profit'],
                'timestamp': trade_result['timestamp']
            })

            # Update performance metrics
            self._update_performance_metrics(trade_result)

            # Save training data
            self._save_training_data()

            # Retrain model if enough new data
            if len(self.learning_data) >= self.min_training_samples and \
               len(self.learning_data) % self.retraining_interval == 0:
                asyncio.create_task(self._retrain_model())

        except Exception as e:
            logger.error(f"Error in learning from trade: {str(e)}")

    def _update_performance_metrics(self, trade_result: Dict):
        """Update performance metrics"""
        try:
            # Calculate accuracy (correct decision)
            accuracy = 1.0 if trade_result['profit'] > 0 else 0.0
            self.performance_metrics['accuracy'].append(accuracy)
            
            # Track profit/loss
            self.performance_metrics['profit_loss'].append(trade_result['profit'])
            
            # Keep only last 1000 metrics
            for key in self.performance_metrics:
                if len(self.performance_metrics[key]) > 1000:
                    self.performance_metrics[key] = self.performance_metrics[key][-1000:]
                    
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def _retrain_model(self):
        """Retrain model with new data"""
        try:
            logger.info("Starting model retraining...")
            
            # Prepare training data
            training_data = self._prepare_training_data()
            if not training_data:
                logger.warning("No training data available")
                return
                
            # Create dataset
            dataset = Dataset.from_dict(training_data)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=settings.MODEL_SAVE_PATH,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=100,
                save_total_limit=2,
                learning_rate=2e-5,
                weight_decay=0.01,
                logging_dir=settings.LOG_DIR,
                logging_steps=10,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            # Train model
            trainer.train()
            
            # Save model
            trainer.save_model()
            
            # Update performance metrics
            self.performance_metrics['training_loss'].append(trainer.state.log_history[-1]['loss'])
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")

    def _prepare_training_data(self) -> Dict:
        """Prepare data for model training"""
        try:
            if not self.learning_data:
                return {}
                
            # Convert learning data to training format
            texts = []
            labels = []
            
            for data in self.learning_data:
                # Create input text
                text = self._create_training_prompt(data)
                texts.append(text)
                
                # Create label (correct decision)
                label = 1 if data['outcome'] > 0 else 0
                labels.append(label)
            
            return {
                'text': texts,
                'label': labels
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return {}

    def _create_training_prompt(self, data: Dict) -> str:
        """Create prompt for training data"""
        return f"""Market Conditions:
Price: {data['market_conditions']['price']}
Volume: {data['market_conditions']['volume']}
Liquidity: {data['market_conditions']['liquidity']}
Sentiment: {data['market_conditions']['sentiment']}

Decision: {data['decision']}
Outcome: {'Profitable' if data['outcome'] > 0 else 'Loss'}

Analysis:"""

    async def close(self):
        """Clean up resources"""
        try:
            # Save final training data
            self._save_training_data()
            
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error closing Local LLM: {str(e)}")

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            return {
                'accuracy': np.mean(self.performance_metrics['accuracy']),
                'avg_profit': np.mean(self.performance_metrics['profit_loss']),
                'avg_confidence': np.mean(self.performance_metrics['confidence_scores']),
                'training_loss': self.performance_metrics['training_loss'][-1] if self.performance_metrics['training_loss'] else None
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {} 