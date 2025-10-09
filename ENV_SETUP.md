# Environment Variables Setup

## Create .env File

Create a file named `.env` in the AI-Backend directory with the following content:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your-azure-endpoint-url
AZURE_OPENAI_KEY=your-azure-api-key-here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

## Important Notes

→ The `.env` file is already in `.gitignore` and will NOT be pushed to GitHub
→ Keep your API keys secure and never share them
→ For Vercel deployment, add these as environment variables in your Vercel dashboard

## Local Development

The application will automatically load these environment variables when you run it locally.

