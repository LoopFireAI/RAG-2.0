"""
Cost-efficient feedback collection with smart prompting and minimal LLM usage.
"""
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

class FeedbackCollector:
    def __init__(self, storage):
        self.storage = storage
        self.response_cache = {}  # Cache recent responses for feedback correlation

    def register_response(self, query: str, response: str, retrieved_docs: List[Dict],
                         persona: str = "default", response_time_ms: int = 0, response_id: str = None) -> str:
        """Register a response for potential feedback collection."""
        if response_id is None:
            response_id = str(uuid.uuid4())

        # Store in database instead of memory cache
        success = self.storage.store_response(
            response_id=response_id,
            query=query,
            response_content=response,
            retrieved_docs=retrieved_docs,
            persona=persona,
            response_time_ms=response_time_ms
        )

        if success:
            # Also keep in memory cache for immediate access
            self.response_cache[response_id] = {
                'query': query,
                'response': response,
                'retrieved_docs': retrieved_docs,
                'persona': persona,
                'response_time_ms': response_time_ms,
                'timestamp': datetime.now()
            }

            # Clean old cache entries (keep last 50)
            if len(self.response_cache) > 50:
                oldest_keys = sorted(self.response_cache.keys(),
                                   key=lambda k: self.response_cache[k]['timestamp'])[:10]
                for key in oldest_keys:
                    del self.response_cache[key]

        return response_id

    def collect_feedback_interactive(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Collect feedback through CLI prompts - cost-free interaction."""
        # Check memory cache first, then database
        if response_id in self.response_cache:
            response_data = self.response_cache[response_id]
        else:
            # Try to get from database
            response_data = self.storage.get_response(response_id)
            if not response_data:
                return None

        print("\n" + "="*50)
        print("📊 Quick Feedback (helps improve future responses)")
        print("="*50)

        # Simple satisfaction rating
        while True:
            try:
                print("\nHow satisfied were you with this response?")
                print("[1] Poor  [2] Fair  [3] Good  [4] Very Good  [5] Excellent  [s] Skip")
                satisfaction = input("Rating: ").strip().lower()

                if satisfaction == 's':
                    return None

                satisfaction_score = int(satisfaction)
                if 1 <= satisfaction_score <= 5:
                    break
                else:
                    print("Please enter a number between 1-5 or 's' to skip")
            except ValueError:
                print("Please enter a number between 1-5 or 's' to skip")

        # Optional relevance rating (only if satisfaction is low)
        relevance_score = None
        if satisfaction_score <= 3:
            while True:
                try:
                    print("\nWere the retrieved documents relevant to your question?")
                    print("[1] Not relevant  [2] Somewhat relevant  [3] Very relevant  [s] Skip")
                    relevance = input("Relevance: ").strip().lower()

                    if relevance == 's':
                        break

                    relevance_score = int(relevance)
                    if 1 <= relevance_score <= 3:
                        break
                    else:
                        print("Please enter a number between 1-3 or 's' to skip")
                except ValueError:
                    print("Please enter a number between 1-3 or 's' to skip")

        # Optional text feedback (only for poor responses)
        feedback_text = None
        if satisfaction_score <= 2:
            print("\nWhat could be improved? (optional, press Enter to skip)")
            feedback_text = input("Feedback: ").strip() or None

        # Prepare feedback data
        feedback_data = {
            'query': response_data['query'],
            'response_id': response_id,
            'satisfaction_score': satisfaction_score,
            'relevance_score': relevance_score,
            'feedback_text': feedback_text,
            'retrieved_docs': [
                {
                    'id': doc.get('id', ''),
                    'title': doc.get('title', doc.get('source', 'Unknown')),
                    'metadata': doc.get('metadata', {})
                }
                for doc in response_data['retrieved_docs']
            ],
            'persona': response_data['persona'],
            'response_time_ms': response_data['response_time_ms']
        }

        # Store feedback
        feedback_id = self.storage.store_feedback(feedback_data)

        print(f"\n✅ Thank you! Your feedback helps improve the system.")
        if satisfaction_score <= 2:
            print("💡 We'll work on addressing these issues in future responses.")

        return feedback_data

    def collect_feedback_simple(self, response_id: str, satisfaction: int,
                              relevance: Optional[int] = None,
                              text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Collect feedback programmatically (for API usage)."""
        # Check memory cache first, then database
        if response_id in self.response_cache:
            response_data = self.response_cache[response_id]
        else:
            # Try to get from database
            response_data = self.storage.get_response(response_id)
            if not response_data:
                return None

        feedback_data = {
            'query': response_data['query'],
            'response_id': response_id,
            'satisfaction_score': satisfaction,
            'relevance_score': relevance,
            'feedback_text': text,
            'retrieved_docs': [
                {
                    'id': doc.get('id', ''),
                    'title': doc.get('title', doc.get('source', 'Unknown')),
                    'metadata': doc.get('metadata', {})
                }
                for doc in response_data['retrieved_docs']
            ],
            'persona': response_data['persona'],
            'response_time_ms': response_data['response_time_ms']
        }

        self.storage.store_feedback(feedback_data)
        return feedback_data

    def should_prompt_feedback(self, response_id: str) -> bool:
        """Determine if we should ask for feedback - cost-aware decision."""
        if response_id not in self.response_cache:
            return False

        # Smart prompting: ask less frequently for similar queries
        response_data = self.response_cache[response_id]
        query_score = self.storage.get_query_pattern_score(response_data['query'])

        # If we have good data on similar queries, prompt less often
        if query_score and query_score >= 4.0:
            return False  # Skip feedback for consistently good query types

        # Always ask for feedback on new query patterns or poor performers
        return True

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback statistics without expensive operations."""
        return self.storage.get_feedback_stats()

    def get_improvement_suggestions(self) -> List[str]:
        """Get actionable improvement suggestions based on stored feedback."""
        suggestions = []

        stats = self.storage.get_feedback_stats()
        if stats['avg_satisfaction'] > 0:
            if stats['avg_satisfaction'] < 3.0:
                suggestions.append("🔴 Low satisfaction detected - review response quality")
            elif stats['avg_satisfaction'] < 4.0:
                suggestions.append("🟡 Room for improvement in response satisfaction")

        if stats['avg_relevance'] > 0 and stats['avg_relevance'] < 2.5:
            suggestions.append("🔍 Document relevance issues - review retrieval strategy")

        low_docs = self.storage.get_low_performing_docs()
        if low_docs:
            suggestions.append(f"📄 {len(low_docs)} documents consistently rated as irrelevant")

        if not suggestions:
            suggestions.append("✅ System performing well based on user feedback")

        return suggestions
