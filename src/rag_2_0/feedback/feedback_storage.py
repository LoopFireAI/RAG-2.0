"""
Cost-efficient feedback storage with local persistence and batch processing.
Avoids expensive LLM calls for basic feedback operations.
"""
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import logging

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackStorage:
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = Path(db_path)
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with feedback tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    query_hash TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    satisfaction_score INTEGER,
                    relevance_score INTEGER,
                    feedback_text TEXT,
                    retrieved_docs TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    persona TEXT,
                    response_time_ms INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    response_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response_content TEXT NOT NULL,
                    retrieved_docs TEXT,
                    persona TEXT,
                    response_time_ms INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    doc_title TEXT,
                    query_hash TEXT NOT NULL,
                    relevance_score INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_patterns (
                    query_hash TEXT PRIMARY KEY,
                    query_normalized TEXT NOT NULL,
                    avg_satisfaction REAL,
                    feedback_count INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON feedback(query_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON document_feedback(doc_id)")

    def store_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Store feedback with minimal processing to reduce costs."""
        feedback_id = str(uuid.uuid4())
        query_hash = self._hash_query(feedback_data.get('query', ''))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback (
                    id, query_hash, query, response_id, satisfaction_score,
                    relevance_score, feedback_text, retrieved_docs, persona, response_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback_id,
                query_hash,
                feedback_data.get('query', ''),
                feedback_data.get('response_id', ''),
                feedback_data.get('satisfaction_score'),
                feedback_data.get('relevance_score'),
                feedback_data.get('feedback_text'),
                json.dumps(feedback_data.get('retrieved_docs', [])),
                feedback_data.get('persona', 'default'),
                feedback_data.get('response_time_ms', 0)
            ))

            # Store individual document feedback
            for doc in feedback_data.get('retrieved_docs', []):
                conn.execute("""
                    INSERT INTO document_feedback (doc_id, doc_title, query_hash, relevance_score)
                    VALUES (?, ?, ?, ?)
                """, (
                    doc.get('id', ''),
                    doc.get('title', ''),
                    query_hash,
                    feedback_data.get('relevance_score', 3)
                ))

            # Update query patterns for fast lookup
            self._update_query_patterns(conn, query_hash, feedback_data.get('query', ''),
                                      feedback_data.get('satisfaction_score'))

        return feedback_id

    def store_response(self, response_id: str, query: str, response_content: str,
                      retrieved_docs: List[Dict], persona: str = "default",
                      response_time_ms: int = 0) -> bool:
        """Store response data for later feedback collection."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO responses (
                        response_id, query, response_content, retrieved_docs, 
                        persona, response_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    response_id,
                    query,
                    response_content,
                    json.dumps(retrieved_docs),
                    persona,
                    response_time_ms
                ))
            return True
        except Exception as e:
            logger.error(f"Error storing response: {e}")
            return False

    def get_response(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored response data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT query, response_content, retrieved_docs, persona, response_time_ms
                    FROM responses WHERE response_id = ?
                """, (response_id,))

                result = cursor.fetchone()
                if result:
                    return {
                        'query': result[0],
                        'response': result[1],
                        'retrieved_docs': json.loads(result[2]) if result[2] else [],
                        'persona': result[3],
                        'response_time_ms': result[4]
                    }
                return None
        except Exception as e:
            logger.error(f"Error retrieving response: {e}")
            return None

    def get_document_feedback_scores(self, doc_ids: List[str]) -> Dict[str, float]:
        """Get average relevance scores for documents - used for retrieval weighting."""
        if not doc_ids:
            return {}

        placeholders = ','.join(['?' for _ in doc_ids])
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT doc_id, AVG(relevance_score) as avg_score, COUNT(*) as count
                FROM document_feedback 
                WHERE doc_id IN ({placeholders})
                GROUP BY doc_id
                HAVING count >= 2
            """, doc_ids)

            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_query_pattern_score(self, query: str) -> Optional[float]:
        """Get average satisfaction for similar queries - fast local lookup."""
        query_hash = self._hash_query(query)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT avg_satisfaction FROM query_patterns 
                WHERE query_hash = ? AND feedback_count >= 3
            """, (query_hash,))

            result = cursor.fetchone()
            return result[0] if result else None

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get basic feedback statistics for monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(satisfaction_score) as avg_satisfaction,
                    AVG(relevance_score) as avg_relevance,
                    COUNT(DISTINCT query_hash) as unique_queries
                FROM feedback 
                WHERE satisfaction_score IS NOT NULL
            """)

            stats = cursor.fetchone()
            return {
                'total_feedback': stats[0],
                'avg_satisfaction': round(stats[1], 2) if stats[1] else 0,
                'avg_relevance': round(stats[2], 2) if stats[2] else 0,
                'unique_queries': stats[3]
            }

    def get_low_performing_docs(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Identify documents with consistently low relevance scores."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT doc_id, doc_title, AVG(relevance_score) as avg_score, COUNT(*) as feedback_count
                FROM document_feedback
                GROUP BY doc_id, doc_title
                HAVING feedback_count >= 3 AND avg_score < ?
                ORDER BY avg_score ASC
            """, (threshold,))

            return [
                {
                    'doc_id': row[0],
                    'doc_title': row[1],
                    'avg_score': round(row[2], 2),
                    'feedback_count': row[3]
                }
                for row in cursor.fetchall()
            ]

    def _hash_query(self, query: str) -> str:
        """Create consistent hash for query normalization."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _update_query_patterns(self, conn, query_hash: str, query: str, satisfaction_score: Optional[int]):
        """Update query pattern statistics."""
        if satisfaction_score is None:
            return

        conn.execute("""
            INSERT INTO query_patterns (query_hash, query_normalized, avg_satisfaction, feedback_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(query_hash) DO UPDATE SET
                avg_satisfaction = (avg_satisfaction * feedback_count + ?) / (feedback_count + 1),
                feedback_count = feedback_count + 1,
                last_updated = CURRENT_TIMESTAMP
        """, (query_hash, query.lower().strip(), satisfaction_score, satisfaction_score))

    def export_feedback_batch(self, days: int = 7) -> List[Dict[str, Any]]:
        """Export recent feedback for batch analysis (cost-efficient)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT query, satisfaction_score, relevance_score, feedback_text, persona
                FROM feedback 
                WHERE timestamp >= datetime('now', '-{} days')
                AND (satisfaction_score <= 2 OR feedback_text IS NOT NULL)
            """.format(days))

            return [
                {
                    'query': row[0],
                    'satisfaction_score': row[1],
                    'relevance_score': row[2],
                    'feedback_text': row[3],
                    'persona': row[4]
                }
                for row in cursor.fetchall()
            ]
