# Jira AI Ticket Organizer

AI-powered ticket organizer for Jira Cloud using embeddings and NLP to group issues into themes, find similar tickets, and auto-tag.

## Overview

This tool helps organize Jira tickets by:
- **Grouping issues into themes/Epics** based on semantic similarity
- **Finding and linking similar historical tickets** using embeddings
- **Auto-tagging** with labels/components based on content analysis

## MVP Features (Phase 1)

- Manual trigger processing ("process this issue" or "process recent issues")
- Works with a single configured Jira project
- Local embeddings using Sentence Transformers (no external API required)
- REST API with health check endpoints

## Architecture

```
jira-ai-ticket-organizer/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── jira/
│   │   └── client.py        # Jira REST API client
│   ├── nlp/                 # NLP and embeddings (future)
│   ├── storage/             # Database layer (future)
│   └── services/            # Business logic (future)
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- Jira Cloud instance with admin access
- Jira API token

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/aYaN21406/jira-ai-ticket-organizer.git
cd jira-ai-ticket-organizer
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment**

Copy `.env.example` to `.env` and fill in your details:

```bash
cp .env.example .env
```

Edit `.env` with your Jira credentials:

```ini
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=YOURPROJECT
```

### Getting Jira API Token

1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Give it a name (e.g., "Jira AI Organizer")
4. Copy the token to your `.env` file

## Running the Application

### Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Basic health check
- `GET /health/jira` - Verify Jira connectivity
- `GET /docs` - Interactive API documentation (Swagger UI)

### Test the setup

```bash
# Basic health check
curl http://localhost:8000/health

# Test Jira connection
curl http://localhost:8000/health/jira
```

## Development

### Project Structure

- **app/config.py** - Manages environment variables and settings
- **app/jira/client.py** - Jira REST API wrapper with methods for searching, updating issues
- **app/main.py** - FastAPI application with health check endpoints

### Next Steps (Future Phases)

1. Implement NLP preprocessing and embeddings
2. Set up PostgreSQL + pgvector for similarity search
3. Build issue organization logic (themes, similarity, tagging)
4. Add API endpoints for processing issues
5. Optionally add webhooks for real-time processing

## Technology Stack

- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation and settings management
- **Requests** - HTTP client for Jira API
- **Sentence Transformers** - Local embeddings (future)
- **PostgreSQL + pgvector** - Vector database (future)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
