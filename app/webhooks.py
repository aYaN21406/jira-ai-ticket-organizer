from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
from typing import Optional
from app.services.organizer.organizer import IssueOrganizerService

logger = logging.getLogger(__name__)

router = APIRouter()

class JiraWebhookPayload(BaseModel):
    webhookEvent: str
    issue: Optional[dict] = None
    
organizer_service = None

def get_organizer():
    global organizer_service
    if organizer_service is None:
        organizer_service = IssueOrganizerService()
    return organizer_service

async def process_issue_async(issue_key: str):
    """Process issue in background"""
    try:
        organizer = get_organizer()
        result = await organizer.process_issue(issue_key)
        logger.info(f"Processed issue {issue_key} via webhook: {result}")
    except Exception as e:
        logger.error(f"Error processing issue {issue_key}: {e}")

@router.post("/jira")
async def jira_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Jira webhook events"""
    try:
        payload = await request.json()
        event_type = payload.get("webhookEvent")
        
        logger.info(f"Received webhook event: {event_type}")
        
        # Handle issue created/updated events
        if event_type in ["jira:issue_created", "jira:issue_updated"]:
            issue = payload.get("issue", {})
            issue_key = issue.get("key")
            
            if issue_key:
                # Process in background
                background_tasks.add_task(process_issue_async, issue_key)
                return {"status": "accepted", "issue": issue_key}
        
        return {"status": "ignored", "event": event_type}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
