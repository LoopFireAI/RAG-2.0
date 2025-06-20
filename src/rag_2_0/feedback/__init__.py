"""
Feedback system for RAG 2.0 - Cost-efficient continuous learning.
"""

from .feedback_storage import FeedbackStorage
from .feedback_collector import FeedbackCollector

__all__ = ['FeedbackStorage', 'FeedbackCollector']
