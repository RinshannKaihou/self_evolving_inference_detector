"""The shipped config, tables and detection logic are the evaluated ones (sha256 recorded by tools/make_manifest.py)."""
import hashlib
import json
import os

import v35_detector

PKG = os.path.dirname(v35_detector.__file__)


def test_shipped_files_match_manifest():
    man = json.load(open(os.path.join(PKG, 'MANIFEST.json')))
    for rel, digest in man['sha256'].items():
        assert hashlib.sha256(open(os.path.join(PKG, rel), 'rb').read()).hexdigest() == digest, rel
