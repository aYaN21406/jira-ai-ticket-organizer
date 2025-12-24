"""Persistent database storage for vectors and embeddings using SQLite."""
import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import faiss
from datetime import datetime


class VectorDatabase:
    """Persistent vector database using SQLite + FAISS."""
    
    def __init__(self, db_path: str = "data/vectors.db"):
        """Initialize database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self._create_tables()
        
        # Initialize FAISS index
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = self._load_or_create_index()
        
    def _create_tables(self):
        """Create database tables."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_key TEXT UNIQUE NOT NULL,
                summary TEXT,
                description TEXT,
                issue_type TEXT,
                status TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                embedding_vector BLOB,
                metadata TEXT
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                issue_key TEXT,
                event_type TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_issue_key ON issues(issue_key)
        """)
        
        self.conn.commit()
        
    def _load_or_create_index(self) -> faiss.IndexFlatL2:
        """Load existing FAISS index or create new one."""
        index_path = self.db_path.parent / "faiss.index"
        
        if index_path.exists():
            return faiss.read_index(str(index_path))
        else:
            return faiss.IndexFlatL2(self.dimension)
    
    def save_index(self):
        """Save FAISS index to disk."""
        index_path = self.db_path.parent / "faiss.index"
        faiss.write_index(self.index, str(index_path))
    
    def add_issue(self, issue_key: str, embedding: np.ndarray, 
                  metadata: Dict) -> int:
        """Add or update issue with embedding.
        
        Args:
            issue_key: Jira issue key
            embedding: Vector embedding
            metadata: Issue metadata
            
        Returns:
            Issue database ID
        """
        # Convert embedding to bytes
        embedding_bytes = embedding.astype('float32').tobytes()
        
        # Insert or replace issue
        self.cursor.execute("""
            INSERT OR REPLACE INTO issues 
            (issue_key, summary, description, issue_type, status, 
             created_at, updated_at, embedding_vector, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            issue_key,
            metadata.get('summary'),
            metadata.get('description'),
            metadata.get('issue_type'),
            metadata.get('status'),
            metadata.get('created'),
            datetime.now().isoformat(),
            embedding_bytes,
            json.dumps(metadata)
        ))
        
        self.conn.commit()
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self.save_index()
        
        return self.cursor.lastrowid
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 5) -> List[Dict]:
        """Search for similar issues.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of similar issues with metadata
        """
        if self.index.ntotal == 0:
            return []
        
        # Search in FAISS
        query_vec = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        
        # Get issue details from SQLite
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:  # Invalid index
                continue
                
            self.cursor.execute("""
                SELECT issue_key, summary, description, metadata
                FROM issues
                LIMIT 1 OFFSET ?
            """, (int(idx),))
            
            row = self.cursor.fetchone()
            if row:
                results.append({
                    'issue_key': row[0],
                    'summary': row[1],
                    'description': row[2],
                    'metadata': json.loads(row[3]),
                    'similarity_score': float(1 / (1 + distance))  # Convert distance to similarity
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        self.cursor.execute("SELECT COUNT(*) FROM issues")
        total_issues = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM processed_events")
        total_events = self.cursor.fetchone()[0]
        
        return {
            'total_issues': total_issues,
            'total_events': total_events,
            'index_size': self.index.ntotal
        }
    
    def mark_event_processed(self, event_id: str, issue_key: str, 
                            event_type: str):
        """Mark webhook event as processed."""
        self.cursor.execute("""
            INSERT OR IGNORE INTO processed_events 
            (event_id, issue_key, event_type)
            VALUES (?, ?, ?)
        """, (event_id, issue_key, event_type))
        self.conn.commit()
    
    def is_event_processed(self, event_id: str) -> bool:
        """Check if event was already processed."""
        self.cursor.execute(
            "SELECT 1 FROM processed_events WHERE event_id = ?",
            (event_id,)
        )
        return self.cursor.fetchone() is not None
    
    def close(self):
        """Close database connections."""
        self.conn.close()
