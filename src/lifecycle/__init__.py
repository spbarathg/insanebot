"""
Lifecycle Management Module

This module handles the lifecycle management of ants including:
- Splitting logic when ants reach certain capital thresholds
- Merging logic when ants underperform
- Inheritance system for transferring knowledge and capital
"""

from .splitting_logic import SplittingLogic
from .merging_logic import MergingLogic
from .inheritance_system import InheritanceSystem

__all__ = [
    'SplittingLogic',
    'MergingLogic', 
    'InheritanceSystem'
] 