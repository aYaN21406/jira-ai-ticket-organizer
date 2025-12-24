"""Core organizer service for processing Jira issues."""
from typing import List, Dict, Optional
from collections import Counter

from app.jira.client import jira_client
from app.nlp.preprocess import preprocess_issue_for_embedding
from app.nlp.embeddings import get_embedding_model
from app.storage.vector_store import get_vector_store, IssueRecord
from models.schemas import (
    OrganizationResult,
    SimilarIssue,
    EpicSuggestion
)


class IssueOrganizer:
    """Service for organizing and analyzing Jira issues."""
    
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
    
    def index_issue(self, issue_data: Dict) -> IssueRecord:
        """Index a single issue into the vector store.
        
        Args:
            issue_data: Raw Jira issue data
            
        Returns:
            Created IssueRecord
        """
        fields = issue_data.get('fields', {})
        
        # Extract key fields
        issue_key = issue_data.get('key')
        project_key = issue_key.split('-')[0] if issue_key else ''
        summary = fields.get('summary', '')
        description = fields.get('description', '')
        
        # Handle epic
        epic_key = None
        if 'parent' in fields and fields['parent']:
            epic_key = fields['parent'].get('key')
        
        # Handle labels
        labels = fields.get('labels', [])
        
        # Handle components
        components = [c.get('name', '') for c in fields.get('components', [])]
        
        # Issue type and status
        issue_type = fields.get('issuetype', {}).get('name')
        status = fields.get('status', {}).get('name')
        
        # Preprocess and generate embedding
        preprocessed_text = preprocess_issue_for_embedding(issue_data)
        embedding = self.embedding_model.encode_single(preprocessed_text)
        
        # Create record
        record = IssueRecord(
            issue_key=issue_key,
            project_key=project_key,
            summary=summary,
            description=description if isinstance(description, str) else '',
            preprocessed_text=preprocessed_text,
            embedding=embedding,
            epic_key=epic_key,
            labels=labels,
            components=components,
            issue_type=issue_type,
            status=status
        )
        
        # Add to store
        self.vector_store.add_issue(record)
        return record
    
    def sync_issues(self, max_issues: int = 100, jql_filter: Optional[str] = None) -> int:
        """Sync issues from Jira to vector store.
        
        Args:
            max_issues: Maximum number of issues to sync
            jql_filter: Optional JQL filter
            
        Returns:
            Number of issues synced
        """
        if jql_filter:
            data = jira_client.search_issues(jql=jql_filter, max_results=max_issues)
        else:
            data = jira_client.search_project_recent(max_results=max_issues)
        
        issues = data.get('issues', [])
        
        for issue in issues:
            self.index_issue(issue)
        
        return len(issues)
    
    def suggest_epic(self, similar_issues: List[tuple]) -> Optional[EpicSuggestion]:
        """Suggest an epic based on similar issues.
        
        Args:
            similar_issues: List of (IssueRecord, similarity_score) tuples
            
        Returns:
            EpicSuggestion or None
        """
        if not similar_issues:
            return None
        
        # Count epics from similar issues
        epic_votes = {}
        total_similarity = 0
        
        for record, similarity in similar_issues:
            if record.epic_key:
                if record.epic_key not in epic_votes:
                    epic_votes[record.epic_key] = 0
                epic_votes[record.epic_key] += similarity
                total_similarity += similarity
        
        if not epic_votes:
            return EpicSuggestion(
                epic_key=None,
                confidence=0.0,
                reasoning="No similar issues with assigned epics found."
            )
        
        # Get epic with highest weighted vote
        best_epic = max(epic_votes.items(), key=lambda x: x[1])
        epic_key, vote_sum = best_epic
        
        # Calculate confidence
        confidence = vote_sum / total_similarity if total_similarity > 0 else 0
        
        # Count how many similar issues share this epic
        epic_count = sum(1 for r, _ in similar_issues if r.epic_key == epic_key)
        
        reasoning = (
            f"{epic_count}/{len(similar_issues)} similar issues belong to this epic. "
            f"Confidence: {confidence:.2f}"
        )
        
        return EpicSuggestion(
            epic_key=epic_key,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def suggest_labels(self, similar_issues: List[tuple], top_k: int = 3) -> List[str]:
        """Suggest labels based on similar issues.
        
        Args:
            similar_issues: List of (IssueRecord, similarity_score) tuples
            top_k: Number of top labels to return
            
        Returns:
            List of suggested labels
        """
        if not similar_issues:
            return []
        
        # Weighted label counting
        label_weights = {}
        
        for record, similarity in similar_issues:
            for label in record.labels:
                if label not in label_weights:
                    label_weights[label] = 0
                label_weights[label] += similarity
        
        # Get top labels
        sorted_labels = sorted(label_weights.items(), key=lambda x: x[1], reverse=True)
        return [label for label, _ in sorted_labels[:top_k]]
    
    def suggest_components(self, similar_issues: List[tuple], top_k: int = 2) -> List[str]:
        """Suggest components based on similar issues.
        
        Args:
            similar_issues: List of (IssueRecord, similarity_score) tuples
            top_k: Number of top components to return
            
        Returns:
            List of suggested components
        """
        if not similar_issues:
            return []
        
        # Weighted component counting
        component_weights = {}
        
        for record, similarity in similar_issues:
            for component in record.components:
                if component not in component_weights:
                    component_weights[component] = 0
                component_weights[component] += similarity
        
        # Get top components
        sorted_components = sorted(component_weights.items(), key=lambda x: x[1], reverse=True)
        return [comp for comp, _ in sorted_components[:top_k]]
    
    def organize_issue(self, issue_key: str, top_k_similar: int = 5) -> OrganizationResult:
        """Organize a single issue by finding similar issues and making suggestions.
        
        Args:
            issue_key: Jira issue key to organize
            top_k_similar: Number of similar issues to find
            
        Returns:
            OrganizationResult with suggestions
        """
        # Fetch issue from Jira
        issue_data = jira_client.get_issue(issue_key)
        
        # Index it
        record = self.index_issue(issue_data)
        
        # Find similar issues (excluding itself)
        similar_results = self.vector_store.find_similar(
            record.embedding,
            top_k=top_k_similar + 1,  # +1 to account for the issue itself
            exclude_keys=[issue_key]
        )
        
        # Convert to SimilarIssue objects
        similar_issues_list = []
        for sim_record, similarity in similar_results[:top_k_similar]:
            similar_issues_list.append(SimilarIssue(
                issue_key=sim_record.issue_key,
                summary=sim_record.summary,
                similarity_score=similarity,
                epic_key=sim_record.epic_key,
                labels=sim_record.labels,
                components=sim_record.components
            ))
        
        # Generate suggestions
        epic_suggestion = self.suggest_epic(similar_results[:top_k_similar])
        suggested_labels = self.suggest_labels(similar_results[:top_k_similar])
        suggested_components = self.suggest_components(similar_results[:top_k_similar])
        
        return OrganizationResult(
            issue_key=issue_key,
            similar_issues=similar_issues_list,
            suggested_epic=epic_suggestion,
            suggested_labels=suggested_labels,
            suggested_components=suggested_components
        )


# Global instance
_organizer_instance = None


def get_organizer() -> IssueOrganizer:
    """Get or create the global organizer instance."""
    global _organizer_instance
    if _organizer_instance is None:
        _organizer_instance = IssueOrganizer()
    return _organizer_instance
