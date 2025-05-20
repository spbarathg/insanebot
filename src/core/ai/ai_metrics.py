"""
AI Model Metrics Collection for Prometheus monitoring.
"""
from prometheus_client import Counter, Gauge, Histogram, Summary
import time
import logging
import threading

logger = logging.getLogger(__name__)

# Define AI metrics
AI_MODEL_PREDICTIONS = Counter('ai_model_predictions_total', 'Total number of predictions made by AI model')
AI_MODEL_PREDICTION_TIME = Histogram('ai_model_prediction_seconds', 'Time taken for model predictions')
AI_MODEL_ACCURACY = Gauge('ai_model_prediction_accuracy', 'AI model prediction accuracy')
AI_MODEL_TRAINING_TIME = Summary('ai_model_training_seconds', 'Time spent training the model')
AI_MODEL_CONFIDENCE = Histogram('ai_model_confidence', 'Confidence score of predictions', buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
AI_MODEL_ACCURACY_BY_TYPE = Gauge('ai_model_accuracy_by_type', 'AI model accuracy by prediction type', ['prediction_type'])
AI_MODEL_TRAINING_SAMPLES = Gauge('ai_model_training_samples', 'Number of samples in training data')

class AIMetricsCollector:
    """Collects and exposes AI model metrics."""
    
    def __init__(self):
        self._last_accuracy = 0.0
        self._sample_count = 0
        
    def record_prediction(self, prediction_type: str):
        """Record a model prediction."""
        AI_MODEL_PREDICTIONS.inc()
        
    def record_prediction_time(self, seconds: float):
        """Record time taken for a prediction."""
        AI_MODEL_PREDICTION_TIME.observe(seconds)
        
    def update_accuracy(self, accuracy: float):
        """Update the overall model accuracy."""
        self._last_accuracy = accuracy
        AI_MODEL_ACCURACY.set(accuracy)
        
    def update_accuracy_by_type(self, prediction_type: str, accuracy: float):
        """Update accuracy for a specific prediction type."""
        AI_MODEL_ACCURACY_BY_TYPE.labels(prediction_type=prediction_type).set(accuracy)
        
    def record_confidence(self, confidence: float):
        """Record confidence score of a prediction."""
        AI_MODEL_CONFIDENCE.observe(confidence)
        
    def record_training_time(self, seconds: float):
        """Record time spent training the model."""
        AI_MODEL_TRAINING_TIME.observe(seconds)
        
    def update_training_samples(self, count: int):
        """Update the number of training samples."""
        self._sample_count = count
        AI_MODEL_TRAINING_SAMPLES.set(count)
        
    def get_current_accuracy(self) -> float:
        """Get the current model accuracy."""
        return self._last_accuracy
        
    def get_sample_count(self) -> int:
        """Get the current number of training samples."""
        return self._sample_count
        
    def track_prediction(self, prediction_type: str = "default"):
        """Decorator to track prediction metrics."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                self.record_prediction(prediction_type)
                
                try:
                    result = func(*args, **kwargs)
                    
                    # If result has confidence, record it
                    if isinstance(result, dict) and 'confidence' in result:
                        self.record_confidence(result['confidence'])
                        
                    return result
                finally:
                    elapsed = time.time() - start
                    self.record_prediction_time(elapsed)
            return wrapper
        return decorator
        
    def track_training(self):
        """Decorator to track model training metrics."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # If result has accuracy info, record it
                    if isinstance(result, dict):
                        if 'accuracy' in result:
                            self.update_accuracy(result['accuracy'])
                        if 'sample_count' in result:
                            self.update_training_samples(result['sample_count'])
                    
                    return result
                finally:
                    elapsed = time.time() - start
                    self.record_training_time(elapsed)
            return wrapper
        return decorator

# Create global instance
ai_metrics = AIMetricsCollector() 