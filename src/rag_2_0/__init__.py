"""RAG 2.0 - Advanced Retrieval-Augmented Generation System."""

__version__ = "1.0.0"
__author__ = "LoopFire AI"

from .agents.rag_agent import create_rag_graph, RAGState
from .ingestion.document_ingester import DocumentIngester

__all__ = [
    "create_rag_graph",
    "RAGState",
    "DocumentIngester",
]
