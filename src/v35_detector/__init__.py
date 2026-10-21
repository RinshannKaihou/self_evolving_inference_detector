"""Streaming detectors of severe LLM inference errors (taxonomy v3.5). Repetition: RepDetector with the m4 configuration.

    from v35_detector import RepDetector, CONFIG_M4
    det = RepDetector('qwen3', CONFIG_M4)
"""
import os

from .stream import RepDetector, Result

__version__ = '0.1.0'
CONFIG_M4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'm4.json')

__all__ = ['RepDetector', 'Result', 'CONFIG_M4', '__version__']
