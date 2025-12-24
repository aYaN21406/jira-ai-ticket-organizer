"""In-memory vector store for issue embeddings."""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict


class IssueRecord:
    """Record for a stored issue."""
    
    def __init__(self,
                 issue_key: str,
                 project_key: str,
                 summary: str,
                 description: Optional[str],
                 preprocessed_text: str,
                 embedding: np.ndarray,
                 epic_key: Optional[str] = None,
                 labels: List[str] = None,
                 components: List[str] = None,
                 issue_type: Optional[str] = None,
                 status: Optional[str] = None):
        self.issue_key = issue_key
        self.project_key = project_key
        self.summary = summary
        self.description = description
        self.preprocessed_text = preprocessed_text
        self.embedding = embedding
        self.epic_key = epic_key
        self.labels = labels or []
        self.components = components or []
        self.issue_type = issue_type
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'issue_key': self.issue_key,
            'project_key': self.project_key,
            'summary': self.summary,
            'description': self.description,
            'preprocessed_text': self.preprocessed_text,
            'epic_key': self.epic_key,
            'labels': self.labels,
            'components': self.components,
            'issue_type': self.issue_type,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class InMemoryVectorStore:
    """Simple in-memory vector store for issue embeddings."""
    
    def __init__(self):
        self._issues: Dict[str, IssueRecord] = {}
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._issue_keys_order: List[str] = []
        self._needs_rebuild = True
    
    def add_issue(self, issue: IssueRecord) -> None:
        """Add or update an issue in the store."""
        self._issues[issue.issue_key] = issue
        self._needs_rebuild = True
    
    def get_issue(self, issue_key: str) -> Optional[IssueRecord]:
        """Get an issue by key."""
        return self._issues.get(issue_key)
    
    def delete_issue(self, issue_key: str) -> bool:
        """Delete an issue from the store."""
        if issue_key in self._issues:
            del self._issues[issue_key]
            self._needs_rebuild = True
            return True
        return False
    
    def count(self) -> int:
        """Get count of stored issues."""
        return len(self._issues)
    
    def _rebuild_matrix(self) -> None:
        """Rebuild the embeddings matrix."""
        if not self._issues:
            self._embeddings_matrix = None
            self._issue_keys_order = []
            return
        
        self._issue_keys_order = list(self._issues.keys())
        embeddings = [self._issues[key].embedding for key in self._issue_keys_order]
        self._embeddings_matrix = np.vstack(embeddings)
        self._needs_rebuild = False
    
    def find_similar(self,
                    query_embedding: np.ndarray,
                    top_k: int = 5,
                    exclude_keys: List[str] = None) -> List[Tuple[IssueRecord, float]]:
        """Find most similar issues to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            exclude_keys: Issue keys to exclude from results
            
        Returns:
            List of (IssueRecord, similarity_score) tuples
        """
        if self._needs_rebuild:
            self._rebuild_matrix()
        
        if self._embeddings_matrix is None or len(self._issues) == 0:
            return []
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = np.dot(self._embeddings_matrix, query_norm.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1]
        
        # Build results, excluding specified keys
        exclude_set = set(exclude_keys or [])
        results = []
        
        for idx in top_indices:
            issue_key = self._issue_keys_order[idx]
            if issue_key not in exclude_set:
                issue = self._issues[issue_key]
                similarity = float(similarities[idx])
                results.append((issue, similarity))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_by_epic(self, epic_key: str) -> List[IssueRecord]:
        """Get all issues belonging to an epic."""
        return [issue for issue in self._issues.values() 
                if issue.epic_key == epic_key]
    
    def get_all_epics(self) -> Dict[str, int]:
        """Get all epic keys and their issue counts."""
        epic_counts = defaultdict(int)
        for issue in self._issues.values():
            if issue.epic_key:
                epic_counts[issue.epic_key] += 1
        return dict(epic_counts)
    
    def get_all_labels(self) -> Dict[str, int]:
        """Get all labels and their usage counts."""
        label_counts = defaultdict(int)
        for issue in self._issues.values():
            for label in issue.labels:
                label_counts[label] += 1
        return dict(label_counts)
    
    def get_all_components(self) -> Dict[str, int]:
        """Get all components and their usage counts."""
        component_counts = defaultdict(int)
        for issue in self._issues.values():
            for component in issue.components:
                component_counts[component] += 1
        return dict(component_counts)
    
    def get_all_issues(self) -> List[IssueRecord]:
        """Get all stored issues."""
        return list(self._issues.values())
    
    def clear(self) -> None:
        """Clear all stored issues."""
        self._issues.clear()
        self._embeddings_matrix = None
        self._issue_keys_order = []
        self._needs_rebuild = True


# Global instance
_vector_store_instance = None


def get_vector_store() -> InMemoryVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = InMemoryVectorStore()
    return _vector_store_instance
