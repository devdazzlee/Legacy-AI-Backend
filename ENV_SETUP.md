# Environment Variables Setup

## Create .env File

Create a file named `.env` in the AI-Backend directory with the following content:

```bash
# ============================================
# Azure OpenAI Configuration
# (Required - Main GPT model for text generation)
# ============================================
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_FALLBACK_DEPLOYMENTS=gpt-4o-mini,o4-mini

# ============================================
# Azure Whisper Configuration
# (Required - Speech-to-text transcription)
# ============================================
AZURE_WHISPER_DEPLOYMENT=whisper
AZURE_WHISPER_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/openai/deployments/whisper/audio/transcriptions
AZURE_WHISPER_KEY=your-whisper-api-key-here
AZURE_WHISPER_API_VERSION=2024-06-01

# ============================================
# Database Configuration
# (Required - PostgreSQL database URL)
# ============================================
DATABASE_URL=postgresql://user:password@host:port/database?sslmode=require

# ============================================
# Redis Configuration (Optional)
# (For caching and rate limiting - improves performance)
# ============================================
# Option 1: Standard Redis URL
REDIS_URL=redis://username:password@host:port
# Option 2: Upstash Redis (recommended for Vercel)
UPSTASH_REDIS_REST_URL=https://your-endpoint.upstash.io
```

## Vercel Deployment

For Vercel deployment, add these as environment variables in your Vercel dashboard:

1. Go to your Vercel project → Settings → Environment Variables
2. Add each variable with its value
3. Select the environment (Production, Preview, Development)
4. See `vercel-env-template.txt` for a complete list

**Required Variables for Vercel:**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_WHISPER_DEPLOYMENT`
- `AZURE_WHISPER_ENDPOINT`
- `AZURE_WHISPER_KEY`
- `AZURE_WHISPER_API_VERSION`
- `DATABASE_URL`

**Optional Variables:**
- `REDIS_URL` or `UPSTASH_REDIS_REST_URL` (for caching)

## Local Development

The application will automatically load these environment variables from the `.env` file when you run it locally.

## Important Notes

→ The `.env` file is already in `.gitignore` and will NOT be pushed to GitHub
→ Keep your API keys secure and never share them
→ Replace placeholder values with your actual credentials
→ For production, use environment variables in your deployment platform

## Security Warning

⚠️ NEVER commit actual API keys to Git
⚠️ Always use placeholder text in documentation files
⚠️ Keep real credentials ONLY in .env file or platform environment variables
