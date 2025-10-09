# Vercel Deployment Guide for AI-Backend

## Overview
Your AI-Backend is now configured for Vercel deployment with proper serverless function setup.

---

## What Was Fixed

→ Created proper `api/index.py` entry point that imports your FastAPI app
→ Updated `vercel.json` with correct configuration for Python serverless functions
→ Fixed duplicate `aiohttp` dependency in `requirements.txt`
→ Created `.vercelignore` to exclude unnecessary files
→ Added `runtime.txt` to specify Python 3.9

---

## Deployment Steps

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy to Vercel
Navigate to your project directory:
```bash
cd "/Users/mac/Desktop/Ahmed Work/AI-Backend"
vercel
```

Follow the prompts:
- Set up and deploy? **Yes**
- Which scope? Select your account
- Link to existing project? **No** (for first deployment)
- Project name? **ai-backend** (or your preferred name)
- Directory location? **./** (default)

### Step 4: Set Environment Variables (IMPORTANT!)

⚠️ **Security Warning**: Your Azure OpenAI API key is currently hardcoded in `main.py`. You should move it to environment variables.

In Vercel Dashboard:
1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add these variables:

```
AZURE_OPENAI_ENDPOINT=your-azure-endpoint-url
AZURE_OPENAI_KEY=your-azure-api-key-here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Step 5: Update main.py to Use Environment Variables

Replace lines 94-96 in `main.py`:

**Current (hardcoded - DON'T DO THIS):**
```python
endpoint = "https://your-endpoint.openai.azure.com/"
api_key = "your-hardcoded-key-here"  # BAD PRACTICE!
api_version = "2025-01-01-preview"
```

**Recommended (environment variables):**
```python
import os
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://hakeem-4411-resource.openai.azure.com/")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
```

### Step 6: Production Deployment
```bash
vercel --prod
```

---

## API Endpoints Available

Once deployed, your API will be available at: `https://your-project.vercel.app`

### Main Endpoints:
- `GET /` - Hello World
- `GET /health` - Health check
- `POST /threads/create` - Create conversation thread
- `POST /messages/send` - Send message to AI
- `POST /validate-response` - Validate response detail
- `POST /validate-service-relevance` - Validate service relevance
- `POST /chatbot/ask` - Ask chatbot with context
- `POST /prescreen-message` - AI Prescreener
- `POST /detect-conflicts` - Detect concept conflicts
- `POST /generate-narrative` - Generate shift narrative

---

## Configuration Files Explained

### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",          // Entry point
      "use": "@vercel/python",         // Python runtime
      "config": {
        "maxLambdaSize": "15mb"        // Increased for dependencies
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",                  // Route all requests
      "dest": "api/index.py"           // To the entry point
    }
  ]
}
```

### api/index.py
```python
# This file imports your FastAPI app and exports it for Vercel
from main import app
handler = app
```

---

## Important Notes

→ **Database Limitation**: SQLite database (`ai_prescreener.db`) won't work in Vercel's serverless environment. Consider using:
  - PostgreSQL (Vercel Postgres)
  - Supabase (you already have this in dependencies)
  - MongoDB Atlas

→ **Stateless Functions**: Vercel functions are stateless. In-memory conversation storage in `azure_service.conversations` will not persist between requests. Consider:
  - Using a database for conversation history
  - Using Redis for session storage
  - Using Vercel KV storage

→ **Cold Starts**: First request after inactivity may be slower due to function cold start

→ **Timeout**: Vercel has a 10-second timeout for Hobby plan, 60 seconds for Pro

---

## Testing Your Deployment

### Test Health Endpoint:
```bash
curl https://your-project.vercel.app/health
```

### Test Message Endpoint:
```bash
curl -X POST https://your-project.vercel.app/threads/create
```

---

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Make sure all dependencies are in `requirements.txt`

### Issue: "Function timeout" error
**Solution**: Optimize long-running operations or upgrade Vercel plan

### Issue: Database connection fails
**Solution**: Use a cloud database instead of SQLite

### Issue: CORS errors
**Solution**: Your CORS is already configured to allow all origins (`allow_origins=["*"]`)

---

## Next Steps

1. **Secure API Keys**: Move hardcoded credentials to environment variables
2. **Database Migration**: Switch from SQLite to cloud database
3. **Session Management**: Implement persistent storage for conversations
4. **Monitoring**: Set up logging and error tracking
5. **Rate Limiting**: Add API rate limiting for production

---

## Quick Deploy Command

For future deployments:
```bash
cd "/Users/mac/Desktop/Ahmed Work/AI-Backend"
vercel --prod
```

---

## Support Resources

- Vercel Documentation: https://vercel.com/docs
- FastAPI on Vercel: https://vercel.com/docs/frameworks/fastapi
- Vercel Python Runtime: https://vercel.com/docs/runtimes#official-runtimes/python

