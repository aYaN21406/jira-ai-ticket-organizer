"""Pydantic models for request/response validation."""
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class IssueBase(BaseModel):
    """Base model for Jira issue data."""
    issue_key: str = Field(..., description="Jira issue key (e.g., PROJ-123)")
    project_key: str = Field(..., description="Jira project key")
    summary: str = Field(..., description="Issue summary/title")
    description: Optional[str] = Field(None, description="Issue description")
    issue_type: Optional[str] = Field(None, description="Issue type (Bug, Task, etc.)")
    status: Optional[str] = Field(None, description="Issue status")
    epic_key: Optional[str] = Field(None, description="Parent epic key")
    labels: List[str] = Field(default_factory=list, description="Issue labels")
    components: List[str] = Field(default_factory=list, description="Issue components")


class IssueCreate(IssueBase):
    """Model for creating a new issue in storage."""
    preprocessed_text: str = Field(..., description="Preprocessed text for embedding")
    embedding: List[float] = Field(..., description="Embedding vector")


class IssueInDB(IssueBase):
    """Model for issue stored in database."""
    id: int = Field(..., description="Internal database ID")
    preprocessed_text: str
    embedding: List[float]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class SimilarIssue(BaseModel):
    """Model for similar issue result."""
    issue_key: str
    summary: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    epic_key: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    components: List[str] = Field(default_factory=list)


class EpicSuggestion(BaseModel):
    """Model for epic suggestion result."""
    epic_key: Optional[str] = Field(None, description="Suggested epic key")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation for suggestion")


class OrganizationResult(BaseModel):
    """Model for issue organization result."""
    issue_key: str
    similar_issues: List[SimilarIssue] = Field(default_factory=list)
    suggested_epic: Optional[EpicSuggestion] = None
    suggested_labels: List[str] = Field(default_factory=list)
    suggested_components: List[str] = Field(default_factory=list)


class ProcessIssueRequest(BaseModel):
    """Request model for processing a single issue."""
    issue_key: str = Field(..., description="Issue key to process")
    fetch_from_jira: bool = Field(True, description="Fetch latest data from Jira")
    top_k_similar: int = Field(5, ge=1, le=20, description="Number of similar issues to find")


class ProcessRecentRequest(BaseModel):
    """Request model for processing recent issues."""
    max_results: int = Field(10, ge=1, le=100, description="Number of recent issues to process")
    top_k_similar: int = Field(5, ge=1, le=20, description="Number of similar issues per issue")


class SyncHistoryRequest(BaseModel):
    """Request model for syncing historical issues."""
    max_issues: int = Field(100, ge=1, le=500, description="Maximum issues to sync")
    jql_filter: Optional[str] = Field(None, description="Custom JQL filter")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    jira_reachable: Optional[bool] = None
    embedding_model_loaded: Optional[bool] = None
    indexed_issues_count: Optional[int] = None
