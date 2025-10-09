"""
Vercel Serverless Function Entry Point
This file serves as the entry point for Vercel deployment
"""

import sys
import os

# Add the parent directory to the path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from main.py
from main import app

# Vercel expects the app to be named 'app' for ASGI applications
# No need for a separate handler
__all__ = ['app']

