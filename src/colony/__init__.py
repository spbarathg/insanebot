"""
Ant Colony Architecture Module

This module implements the hierarchical ant colony system for the Ant Bot Ultimate Bot.
Provides clean separation of concerns and modular design for all ant types.
"""

from .founding_queen import FoundingAntQueen
from .ant_queen import AntQueen
from .worker_ant import WorkerAnt
from .ant_drone import AntDrone
from .accounting_ant import AccountingAnt
from .ant_princess import AntPrincess

__all__ = [
    'FoundingAntQueen',
    'AntQueen', 
    'WorkerAnt',
    'AntDrone',
    'AccountingAnt',
    'AntPrincess'
] 