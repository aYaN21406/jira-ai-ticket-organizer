from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel
import logging
from app.storage.database import Database
from app.services.organizer import IssueOrganizerService
from app.config import settings

Add comprehensive API endpoints with database integration
router = APIRouter()
db = Database()
organizer_service = None

def get_organizer():
    global organizer_service
    if organizer_service is None:
        organizer_service = IssueOrganizerService()
    return organizer_service

# Request/Response models
class ProcessIssueRequest(BaseModel):
    issue_key: str

class ProcessIssueResponse(BaseModel):
    issue_key: str
    status: str
    similar_issues: List[dict]
    suggested_labels: List[str]
    suggested_epic: Optional[str] = None

class IssueResponse(BaseModel):
    issue_key: str
    summary: str
    description: Optional[str]
    labels: List[str]
    components: List[str]
    created_at: str

@router.post("/process", response_model=ProcessIssueResponse)
async def process_issue(request: ProcessIssueRequest):
    """
    Manually trigger processing of a specific Jira issue.
    
    This endpoint:
    - Fetches the issue from Jira
    - Generates embeddings
    - Finds similar historical issues
    - Suggests labels/components
    - Suggests grouping into Epics
    - Stores results in database
    """
    try:
        organizer = get_organizer()
        result = await organizer.process_issue(request.issue_key)
        
        return ProcessIssueResponse(
            issue_key=request.issue_key,
            status="success",
            similar_issues=result.get("similar_issues", []),
            suggested_labels=result.get("suggested_labels", []),
            suggested_epic=result.get("suggested_epic")
        )
    except Exception as e:
        logger.error(f"Error processing issue {request.issue_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/issues", response_model=List[IssueResponse])
async def get_processed_issues(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get list of processed issues from database.
    
    Args:
        limit: Maximum number of issues to return (1-100)
        offset: Number of issues to skip for pagination
    """
    try:
        issues = db.get_all_issues(limit=limit, offset=offset)
        return [
            IssueResponse(
                issue_key=issue["issue_key"],
                summary=issue["summary"],
                description=issue.get("description"),
                labels=issue.get("labels", []),
                components=issue.get("components", []),
                created_at=issue["created_at"]
            )
            for issue in issues
        ]
    except Exception as e:
        logger.error(f"Error fetching issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/issues/{issue_key}", response_model=IssueResponse)
async def get_issue(issue_key: str):
    """
    Get details of a specific processed issue from database.
    
    Args:
        issue_key: The Jira issue key (e.g., PROJ-123)
    """
    try:
        issue = db.get_issue(issue_key)
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        return IssueResponse(
            issue_key=issue["issue_key"],
            summary=issue["summary"],
            description=issue.get("description"),
            labels=issue.get("labels", []),
            components=issue.get("components", []),
            created_at=issue["created_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching issue {issue_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/issues/{issue_key}/similar")
async def get_similar_issues(issue_key: str, limit: int = Query(default=5, ge=1, le=20)):
    """
    Find similar issues to a given issue using vector similarity search.
    
    Args:
        issue_key: The Jira issue key (e.g., PROJ-123)
        limit: Maximum number of similar issues to return (1-20)
    """
    try:
        organizer = get_organizer()
        similar = await organizer.find_similar_issues(issue_key, limit=limit)
        return {
            "issue_key": issue_key,
            "similar_issues": similar
        }
    except Exception as e:
        logger.error(f"Error finding similar issues for {issue_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/process")
async def batch_process_issues(
    project_key: Optional[str] = None,
    jql: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=100)
):
    """
    Batch process multiple issues from Jira.
    
    Args:
        project_key: Process all issues from a specific project
        jql: Custom JQL query to fetch issues
        limit: Maximum number of issues to process (1-100)
    """
    try:
        organizer = get_organizer()
        
        # Use project_key from settings if not provided
        if not project_key and not jql:
            project_key = settings.jira_project_key
        
        results = await organizer.batch_process(
            project_key=project_key,
            jql=jql,
            limit=limit
        )
        
        return {
            "status": "success",
            "processed_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/issues/{issue_key}")
async def delete_issue(issue_key: str):
    """
    Delete a processed issue from database.
    
    Args:
        issue_key: The Jira issue key (e.g., PROJ-123)
    """
    try:
        success = db.delete_issue(issue_key)
        if not success:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        return {"status": "success", "message": f"Issue {issue_key} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting issue {issue_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
