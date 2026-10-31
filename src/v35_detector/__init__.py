"""Streaming detectors of severe LLM inference errors (taxonomy v3.5).
Repetition: RepDetector with the m4 configuration (token ids only).
Garbled output (rare_chars): rc3 / rc4 use token ids and raw-distribution logprobs; rc1t uses generated token ids alone.

    from v35_detector import RepDetector, CONFIG_M4, RareDetector, CONFIG_RC3, CONFIG_RC1T
    det = RepDetector('qwen3', CONFIG_M4)
    rare = RareDetector('qwen3', CONFIG_RC3)                # or CONFIG_RC4 / CONFIG_RC1T
"""
import os

from .rare import RareDetector
from .stream import RepDetector, Result

__version__ = '0.5.0'
CONFIG_M4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'm4.json')
CONFIG_RC3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'rc3.json')
CONFIG_RC4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'rc4.json')
CONFIG_RC1T = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'rc1t.json')

__all__ = ['RepDetector', 'RareDetector', 'Result', 'CONFIG_M4', 'CONFIG_RC3', 'CONFIG_RC4', 'CONFIG_RC1T', '__version__']
