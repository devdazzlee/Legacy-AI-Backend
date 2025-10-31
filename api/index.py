# api/index.py
import sys
import os

# Add the parent directory to the path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from main.py
from main import app

# Export the app for Vercel (ASGI app)
handler = app
