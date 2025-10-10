# Environment Variables Setup

## Create .env File

Create a file named `.env` in the AI-Backend directory with the following content:

```bash
# Azure OpenAI Configuration (for GPT-4 text generation)
AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint-url
AZURE_OPENAI_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Azure Whisper Configuration (for speech-to-text)
AZURE_WHISPER_DEPLOYMENT=whisper
AZURE_WHISPER_ENDPOINT=your-whisper-endpoint-url
AZURE_WHISPER_KEY=your-whisper-api-key-here
AZURE_WHISPER_API_VERSION=2024-06-01
```

## Important Notes

→ The `.env` file is already in `.gitignore` and will NOT be pushed to GitHub
→ Keep your API keys secure and never share them
→ Replace placeholder values with your actual credentials
→ For Vercel deployment, add these as environment variables in your Vercel dashboard

## Local Development

The application will automatically load these environment variables when you run it locally.

## Security Warning

⚠️ NEVER commit actual API keys to Git
⚠️ Always use placeholder text in documentation files
⚠️ Keep real credentials ONLY in .env file
