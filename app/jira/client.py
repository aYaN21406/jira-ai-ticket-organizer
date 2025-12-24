"""Jira REST API client for interacting with Jira Cloud."""
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

from app.config import settings


class JiraClient:
    """Client for interacting with Jira REST API v3."""

    def __init__(self) -> None:
        self.base_url = settings.jira_base_url.rstrip("/")
        self.auth = HTTPBasicAuth(settings.jira_email, settings.jira_api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def search_issues(self, jql: str, max_results: int = 50) -> Dict[str, Any]:
        """Search for issues using JQL.
        
        Args:
            jql: JQL query string
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        url = f"{self.base_url}/rest/api/3/search"
        params = {"jql": jql, "maxResults": max_results}
        resp = requests.get(url, headers=self.headers, params=params, auth=self.auth)
        resp.raise_for_status()
        return resp.json()

    def get_issue(self, issue_key: str) -> Dict[str, Any]:
        """Get a single issue by key.
        
        Args:
            issue_key: Issue key (e.g., 'PROJ-123')
            
        Returns:
            Dictionary containing issue data
        """
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        resp = requests.get(url, headers=self.headers, auth=self.auth)
        resp.raise_for_status()
        return resp.json()

    def update_issue_fields(self, issue_key: str, fields: Dict[str, Any]) -> None:
        """Update fields on an issue.
        
        Args:
            issue_key: Issue key to update
            fields: Dictionary of fields to update
        """
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        payload = {"fields": fields}
        resp = requests.put(url, headers=self.headers, json=payload, auth=self.auth)
        resp.raise_for_status()

    def add_comment(self, issue_key: str, body: str) -> None:
        """Add a comment to an issue.
        
        Args:
            issue_key: Issue key to comment on
            body: Comment text
        """
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}/comment"
        payload = {"body": body}
        resp = requests.post(url, headers=self.headers, json=payload, auth=self.auth)
        resp.raise_for_status()

    def search_project_recent(self, max_results: int = 50) -> Dict[str, Any]:
        """Search for recent issues in the configured project.
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        jql = f'project = "{settings.jira_project_key}" ORDER BY created DESC'
        return self.search_issues(jql=jql, max_results=max_results)


jira_client = JiraClient()
