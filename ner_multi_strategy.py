#!/usr/bin/env python3
"""
Multi-Strategy NER System - Refactored Entry Point

This is the main entry point that uses the refactored modular architecture.
It provides the same functionality as the original script but with better organization.
"""

import sys
import os

# Add the ner_app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ner_app'))

from ner_app.main import main

if __name__ == "__main__":
    exit(main())
