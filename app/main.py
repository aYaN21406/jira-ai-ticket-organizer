"""FastAPI application entry point for Jira AI Ticket Organizer."""
from fastapi import FastAPI

from app.config import settings
from app.jira.client import jira_client

app = FastAPI(
    title="Jira AI Ticket Organizer",
    version="0.1.0",
    description="AI-powered ticket organizer for Jira Cloud"
)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Jira AI Ticket Organizer API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Basic health check endpoint.
    
    Returns:
        Dictionary with health status and configuration info
    """
    return {
        "status": "ok",
        "jira_base_url": str(settings.jira_base_url),
        "project": settings.jira_project_key,
    }


@app.get("/health/jira")
def health_check_jira():
    """Deeper health check that verifies Jira credentials.
    
    Runs a lightweight query to verify API connectivity.
    
    Returns:
        Dictionary with Jira connection status
    """
    try:
        data = jira_client.search_project_recent(max_results=1)
        total = data.get("total", 0)
        return {
            "status": "ok",
            "jira_reachable": True,
            "sample_issue_count": total
        }
    except Exception as e:
        return {
            "status": "error",
            "jira_reachable": False,
            "error": str(e)
        }
