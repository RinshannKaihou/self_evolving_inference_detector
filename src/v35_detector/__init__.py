"""Streaming detectors of severe LLM inference errors (taxonomy v3.5).
Repetition: RepDetector with the m4 configuration (token ids only).
Garbled output (rare_chars): RareDetector with the rc3 or rc4 configuration (token ids, the top-1 logprob and the generated token's
logprob of the raw distribution; online they are computed in vLLM's sampler, serving/collect.py).

    from v35_detector import RepDetector, CONFIG_M4, RareDetector, CONFIG_RC3
    det = RepDetector('qwen3', CONFIG_M4)
    rare = RareDetector('qwen3', CONFIG_RC3)                # or CONFIG_RC4
"""
import os

from .rare import RareDetector
from .stream import RepDetector, Result

__version__ = '0.4.0'
CONFIG_M4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'm4.json')
CONFIG_RC3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'rc3.json')
CONFIG_RC4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'rc4.json')

__all__ = ['RepDetector', 'RareDetector', 'Result', 'CONFIG_M4', 'CONFIG_RC3', 'CONFIG_RC4', '__version__']
