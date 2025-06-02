"""
Message Queue System - Async task processing and event-driven architecture

Provides async task queues, background job scheduling, event publishing/subscribing,
and message routing for scalable system operations.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import redis.asyncio as redis
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Task definition for queue processing"""
    task_id: str
    queue_name: str
    function_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 60.0
    timeout: float = 300.0
    scheduled_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueueStats:
    """Queue statistics"""
    queue_name: str
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_processed: int
    avg_processing_time: float
    last_updated: float

class MessageQueue:
    """
    Comprehensive message queue system
    
    Features:
    - Redis-backed or in-memory queues
    - Task prioritization and scheduling
    - Automatic retries with exponential backoff
    - Dead letter queues for failed tasks
    - Real-time monitoring and metrics
    - Event publishing/subscribing
    - Background job scheduling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.backend_type = self.config.get('backend', 'memory')  # 'redis' or 'memory'
        
        # Queue storage
        self.queues: Dict[str, deque] = defaultdict(deque)
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        # Workers and monitoring
        self.workers: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.queue_stats: Dict[str, QueueStats] = {}
        
        # Event system
        self.event_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Redis connection (if using Redis backend)
        self.redis_client: Optional[redis.Redis] = None
        
        logger.info(f"MessageQueue initialized with {self.backend_type} backend")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default message queue configuration"""
        return {
            'backend': 'memory',  # 'redis' or 'memory'
            'redis_url': 'redis://localhost:6379',
            'default_queue': 'default',
            'max_workers_per_queue': 3,
            'task_timeout': 300.0,
            'max_retries': 3,
            'retry_delay': 60.0,
            'dead_letter_queue': 'dead_letter',
            'cleanup_interval': 3600.0,
            'stats_update_interval': 60.0,
            'enable_monitoring': True
        }
    
    async def initialize(self) -> bool:
        """Initialize the message queue system"""
        try:
            if self.backend_type == 'redis':
                await self._initialize_redis()
            
            # Start monitoring
            if self.config.get('enable_monitoring', True):
                asyncio.create_task(self._monitoring_loop())
                asyncio.create_task(self._cleanup_loop())
            
            self.is_running = True
            logger.info("MessageQueue system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MessageQueue: {e}")
            return False
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=False
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def register_task_handler(self, function_name: str, handler: Callable):
        """Register a task handler function"""
        self.task_handlers[function_name] = handler
        logger.info(f"Registered task handler: {function_name}")
    
    def add_middleware(self, middleware: Callable):
        """Add middleware for task processing"""
        self.middleware.append(middleware)
        logger.info("Added middleware to task processing pipeline")
    
    async def enqueue_task(self, queue_name: str, function_name: str, 
                          *args, **kwargs) -> str:
        """Enqueue a task for processing"""
        try:
            task_id = str(uuid.uuid4())
            
            # Extract task options from kwargs
            priority = kwargs.pop('_priority', TaskPriority.NORMAL)
            max_retries = kwargs.pop('_max_retries', self.config.get('max_retries', 3))
            timeout = kwargs.pop('_timeout', self.config.get('task_timeout', 300.0))
            scheduled_at = kwargs.pop('_scheduled_at', None)
            metadata = kwargs.pop('_metadata', {})
            
            task = Task(
                task_id=task_id,
                queue_name=queue_name,
                function_name=function_name,
                args=list(args),
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout,
                scheduled_at=scheduled_at,
                metadata=metadata
            )
            
            if self.backend_type == 'redis':
                await self._enqueue_task_redis(task)
            else:
                await self._enqueue_task_memory(task)
            
            logger.debug(f"Enqueued task {task_id} to queue {queue_name}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise
    
    async def _enqueue_task_redis(self, task: Task):
        """Enqueue task using Redis backend"""
        try:
            # Serialize task using JSON instead of pickle for security
            task_dict = {
                'task_id': task.task_id,
                'queue_name': task.queue_name,
                'function_name': task.function_name,
                'args': task.args,
                'kwargs': task.kwargs,
                'priority': task.priority.value,
                'max_retries': task.max_retries,
                'retry_delay': task.retry_delay,
                'timeout': task.timeout,
                'scheduled_at': task.scheduled_at,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'status': task.status.value,
                'result': task.result,
                'error': task.error,
                'retry_count': task.retry_count,
                'metadata': task.metadata
            }
            task_data = json.dumps(task_dict).encode('utf-8')
            
            # Add to appropriate queue based on priority and scheduling
            if task.scheduled_at and task.scheduled_at > time.time():
                # Delayed task - add to scheduled set
                await self.redis_client.zadd(
                    f"scheduled:{task.queue_name}",
                    {task_data: task.scheduled_at}
                )
            else:
                # Immediate task - add to priority queue
                priority_score = task.priority.value * 1000000 + time.time()
                await self.redis_client.zadd(
                    f"queue:{task.queue_name}",
                    {task_data: priority_score}
                )
            
        except Exception as e:
            logger.error(f"Failed to enqueue task to Redis: {e}")
            raise
    
    async def _enqueue_task_memory(self, task: Task):
        """Enqueue task using memory backend"""
        try:
            if task.scheduled_at and task.scheduled_at > time.time():
                # Handle scheduled tasks (simplified implementation)
                asyncio.create_task(self._schedule_task_memory(task))
            else:
                # Add to queue based on priority
                if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
                    self.queues[task.queue_name].appendleft(task)
                else:
                    self.queues[task.queue_name].append(task)
            
        except Exception as e:
            logger.error(f"Failed to enqueue task to memory: {e}")
            raise
    
    async def _schedule_task_memory(self, task: Task):
        """Schedule a task for future execution (memory backend)"""
        delay = task.scheduled_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Add to immediate queue
        task.scheduled_at = None
        await self._enqueue_task_memory(task)
    
    async def start_workers(self, queue_name: str = None, worker_count: int = None) -> bool:
        """Start workers for task processing"""
        try:
            queues_to_start = [queue_name] if queue_name else ['default']
            if not queue_name:
                # Start workers for all known queues
                queues_to_start.extend(self.queues.keys())
            
            for queue in queues_to_start:
                count = worker_count or self.config.get('max_workers_per_queue', 3)
                
                for i in range(count):
                    worker_id = f"{queue}_worker_{i}"
                    if worker_id not in self.workers:
                        self.workers[worker_id] = asyncio.create_task(
                            self._worker_loop(queue, worker_id)
                        )
            
            logger.info(f"Started workers for queues: {', '.join(queues_to_start)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start workers: {e}")
            return False
    
    async def _worker_loop(self, queue_name: str, worker_id: str):
        """Main worker loop for processing tasks"""
        logger.info(f"Worker {worker_id} started for queue {queue_name}")
        
        while self.is_running:
            try:
                # Get next task
                task = await self._get_next_task(queue_name)
                
                if not task:
                    await asyncio.sleep(1.0)  # No tasks available
                    continue
                
                # Process task
                await self._process_task(task, worker_id)
                
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")
                await asyncio.sleep(5.0)  # Error recovery delay
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _get_next_task(self, queue_name: str) -> Optional[Task]:
        """Get next task from queue"""
        try:
            if self.backend_type == 'redis':
                return await self._get_next_task_redis(queue_name)
            else:
                return await self._get_next_task_memory(queue_name)
                
        except Exception as e:
            logger.error(f"Failed to get next task from {queue_name}: {e}")
            return None
    
    async def _get_next_task_redis(self, queue_name: str) -> Optional[Task]:
        """Get next task from Redis queue"""
        try:
            # Check for scheduled tasks that are ready
            current_time = time.time()
            scheduled_tasks = await self.redis_client.zrangebyscore(
                f"scheduled:{queue_name}",
                0, current_time,
                start=0, num=1
            )
            
            if scheduled_tasks:
                # Move scheduled task to immediate queue
                task_data = scheduled_tasks[0]
                await self.redis_client.zrem(f"scheduled:{queue_name}", task_data)
                await self.redis_client.zadd(
                    f"queue:{queue_name}",
                    {task_data: current_time}
                )
            
            # Get highest priority task
            result = await self.redis_client.zpopmax(f"queue:{queue_name}")
            if result:
                task_data, _ = result[0]
                # Deserialize task using JSON instead of pickle for security
                task_dict = json.loads(task_data.decode('utf-8'))
                
                # Recreate Task object from JSON data
                task = Task(
                    task_id=task_dict['task_id'],
                    queue_name=task_dict['queue_name'],
                    function_name=task_dict['function_name'],
                    args=task_dict['args'],
                    kwargs=task_dict['kwargs'],
                    priority=TaskPriority(task_dict['priority']),
                    max_retries=task_dict['max_retries'],
                    retry_delay=task_dict['retry_delay'],
                    timeout=task_dict['timeout'],
                    scheduled_at=task_dict['scheduled_at'],
                    created_at=task_dict['created_at'],
                    started_at=task_dict['started_at'],
                    completed_at=task_dict['completed_at'],
                    status=TaskStatus(task_dict['status']),
                    result=task_dict['result'],
                    error=task_dict['error'],
                    retry_count=task_dict['retry_count'],
                    metadata=task_dict['metadata']
                )
                return task
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task from Redis queue {queue_name}: {e}")
            return None
    
    async def _get_next_task_memory(self, queue_name: str) -> Optional[Task]:
        """Get next task from memory queue"""
        try:
            if queue_name in self.queues and self.queues[queue_name]:
                return self.queues[queue_name].popleft()
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task from memory queue {queue_name}: {e}")
            return None
    
    async def _process_task(self, task: Task, worker_id: str):
        """Process a single task"""
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self.running_tasks[task.task_id] = task
            
            # Get task handler
            if task.function_name not in self.task_handlers:
                raise ValueError(f"No handler registered for function: {task.function_name}")
            
            handler = self.task_handlers[task.function_name]
            
            # Apply middleware (pre-processing)
            for middleware in self.middleware:
                await middleware(task, 'before')
            
            # Execute task with timeout
            try:
                result = await asyncio.wait_for(
                    handler(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                
                # Apply middleware (post-processing)
                for middleware in self.middleware:
                    await middleware(task, 'after')
                
                logger.debug(f"Task {task.task_id} completed successfully")
                
            except asyncio.TimeoutError:
                raise Exception(f"Task timeout after {task.timeout} seconds")
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            
        except Exception as e:
            # Task failed
            task.error = str(e)
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                # Retry task
                task.status = TaskStatus.RETRYING
                delay = task.retry_delay * (2 ** (task.retry_count - 1))  # Exponential backoff
                
                logger.warning(f"Task {task.task_id} failed (attempt {task.retry_count}), retrying in {delay}s: {e}")
                
                # Schedule retry
                task.scheduled_at = time.time() + delay
                await self._enqueue_task_memory(task) if self.backend_type == 'memory' else await self._enqueue_task_redis(task)
                
            else:
                # Max retries exceeded - move to failed tasks
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self.failed_tasks[task.task_id] = task
                
                logger.error(f"Task {task.task_id} failed permanently after {task.retry_count} attempts: {e}")
                
                # Send to dead letter queue
                await self._send_to_dead_letter_queue(task)
        
        finally:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    async def _send_to_dead_letter_queue(self, task: Task):
        """Send failed task to dead letter queue"""
        try:
            dead_letter_queue = self.config.get('dead_letter_queue', 'dead_letter')
            
            # Add error details to metadata
            task.metadata['original_queue'] = task.queue_name
            task.metadata['final_error'] = task.error
            task.metadata['failure_timestamp'] = time.time()
            
            # Change queue name to dead letter queue
            task.queue_name = dead_letter_queue
            
            if self.backend_type == 'redis':
                await self._enqueue_task_redis(task)
            else:
                self.queues[dead_letter_queue].append(task)
            
            logger.warning(f"Task {task.task_id} sent to dead letter queue")
            
        except Exception as e:
            logger.error(f"Failed to send task to dead letter queue: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and details"""
        try:
            # Check running tasks
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'queue_name': task.queue_name,
                    'function_name': task.function_name,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'retry_count': task.retry_count,
                    'progress': 'running'
                }
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'queue_name': task.queue_name,
                    'function_name': task.function_name,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'result': task.result,
                    'processing_time': task.completed_at - task.started_at if task.started_at else None
                }
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'status': task.status.value,
                    'queue_name': task.queue_name,
                    'function_name': task.function_name,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'error': task.error,
                    'retry_count': task.retry_count
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
    
    async def get_queue_stats(self, queue_name: str = None) -> Dict[str, QueueStats]:
        """Get queue statistics"""
        try:
            stats = {}
            
            queues_to_check = [queue_name] if queue_name else list(self.queues.keys())
            if not queues_to_check:
                queues_to_check = ['default']
            
            for queue in queues_to_check:
                pending = len(self.queues.get(queue, []))
                running = len([t for t in self.running_tasks.values() if t.queue_name == queue])
                completed = len([t for t in self.completed_tasks.values() if t.queue_name == queue])
                failed = len([t for t in self.failed_tasks.values() if t.queue_name == queue])
                
                # Calculate average processing time
                completed_tasks = [t for t in self.completed_tasks.values() if t.queue_name == queue]
                avg_time = 0.0
                if completed_tasks:
                    processing_times = [
                        t.completed_at - t.started_at 
                        for t in completed_tasks 
                        if t.started_at and t.completed_at
                    ]
                    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
                
                stats[queue] = QueueStats(
                    queue_name=queue,
                    pending_tasks=pending,
                    running_tasks=running,
                    completed_tasks=completed,
                    failed_tasks=failed,
                    total_processed=completed + failed,
                    avg_processing_time=avg_time,
                    last_updated=time.time()
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def publish_event(self, event_name: str, data: Any):
        """Publish an event to subscribers"""
        try:
            subscribers = self.event_subscribers.get(event_name, [])
            
            for subscriber in subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event_name, data)
                    else:
                        subscriber(event_name, data)
                except Exception as e:
                    logger.error(f"Error in event subscriber for {event_name}: {e}")
            
            logger.debug(f"Published event {event_name} to {len(subscribers)} subscribers")
            
        except Exception as e:
            logger.error(f"Failed to publish event {event_name}: {e}")
    
    def subscribe_to_event(self, event_name: str, callback: Callable):
        """Subscribe to an event"""
        self.event_subscribers[event_name].append(callback)
        logger.info(f"Subscribed to event: {event_name}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Update queue statistics
                await self._update_queue_stats()
                
                # Publish monitoring events
                stats = await self.get_queue_stats()
                await self.publish_event('queue_stats_updated', stats)
                
                try:
                    await asyncio.sleep(self.config.get('stats_update_interval', 60.0))
                except asyncio.CancelledError:
                    break
                
            except asyncio.CancelledError:
                logger.info("MessageQueue monitoring loop cancelled")
                break
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error in monitoring loop: {e}")
                    try:
                        await asyncio.sleep(60.0)
                    except asyncio.CancelledError:
                        break
                else:
                    break
    
    async def _update_queue_stats(self):
        """Update queue statistics"""
        try:
            self.queue_stats = await self.get_queue_stats()
        except Exception as e:
            logger.error(f"Failed to update queue stats: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_running:
            try:
                # Clean up old completed/failed tasks
                current_time = time.time()
                cleanup_age = self.config.get('cleanup_interval', 3600.0)
                
                # Clean completed tasks older than cleanup_age
                for task_id in list(self.completed_tasks.keys()):
                    task = self.completed_tasks[task_id]
                    if task.completed_at and (current_time - task.completed_at) > cleanup_age:
                        del self.completed_tasks[task_id]
                
                # Clean failed tasks older than cleanup_age
                for task_id in list(self.failed_tasks.keys()):
                    task = self.failed_tasks[task_id]
                    if task.completed_at and (current_time - task.completed_at) > cleanup_age:
                        del self.failed_tasks[task_id]
                
                try:
                    await asyncio.sleep(cleanup_age)
                except asyncio.CancelledError:
                    break
                
            except asyncio.CancelledError:
                logger.info("MessageQueue cleanup loop cancelled")
                break
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error in cleanup loop: {e}")
                    try:
                        await asyncio.sleep(3600.0)
                    except asyncio.CancelledError:
                        break
                else:
                    break
    
    async def stop_workers(self):
        """Stop all workers"""
        try:
            self.is_running = False
            
            # Cancel all worker tasks
            for worker_id, worker_task in self.workers.items():
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
            
            self.workers.clear()
            logger.info("All workers stopped")
            
        except Exception as e:
            logger.error(f"Error stopping workers: {e}")
    
    async def cleanup(self):
        """Cleanup message queue resources"""
        try:
            await self.stop_workers()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("MessageQueue cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during MessageQueue cleanup: {e}") 