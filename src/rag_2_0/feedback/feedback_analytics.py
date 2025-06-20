"""
Cost-efficient feedback analytics - batch processing for insights.
"""
import json
from typing import Dict, Any
from datetime import datetime

class FeedbackAnalytics:
    def __init__(self, storage):
        self.storage = storage

    def generate_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive insights report for system improvement."""
        stats = self.storage.get_feedback_stats()
        low_docs = self.storage.get_low_performing_docs()

        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': stats,
            'insights': [],
            'recommendations': [],
            'low_performing_documents': low_docs
        }

        # Generate insights based on data
        if stats['total_feedback'] > 0:
            if stats['avg_satisfaction'] < 3.0:
                report['insights'].append({
                    'type': 'quality_issue',
                    'message': f"Low average satisfaction ({stats['avg_satisfaction']}/5.0)",
                    'priority': 'high'
                })
                report['recommendations'].append({
                    'action': 'review_response_generation',
                    'description': 'Review response generation prompts and context usage'
                })

            if stats['avg_relevance'] < 2.5:
                report['insights'].append({
                    'type': 'retrieval_issue',
                    'message': f"Low document relevance ({stats['avg_relevance']}/3.0)",
                    'priority': 'high'
                })
                report['recommendations'].append({
                    'action': 'optimize_retrieval',
                    'description': 'Review document chunking and embedding strategies'
                })

        if len(low_docs) > 0:
            report['insights'].append({
                'type': 'document_quality',
                'message': f"{len(low_docs)} documents consistently rated as irrelevant",
                'priority': 'medium'
            })
            report['recommendations'].append({
                'action': 'document_audit',
                'description': 'Review and potentially remove low-performing documents'
            })

        return report

    def get_improvement_tracking(self, days: int = 30) -> Dict[str, Any]:
        """Track improvement trends over time."""
        # This would require time-series data in the database
        # For now, return basic trending info
        stats = self.storage.get_feedback_stats()

        return {
            'period_days': days,
            'current_satisfaction': stats['avg_satisfaction'],
            'current_relevance': stats['avg_relevance'],
            'total_feedback': stats['total_feedback'],
            'trend_analysis': 'baseline_period'  # Would calculate actual trends with more data
        }

    def export_feedback_for_analysis(self, days: int = 7) -> str:
        """Export feedback data for external analysis tools."""
        feedback_data = self.storage.export_feedback_batch(days)

        export_data = {
            'export_date': datetime.now().isoformat(),
            'period_days': days,
            'feedback_count': len(feedback_data),
            'feedback': feedback_data
        }

        return json.dumps(export_data, indent=2)

    def get_persona_performance(self) -> Dict[str, Any]:
        """Analyze performance by detected persona/leader."""
        # This would require more sophisticated querying
        # For MVP, return placeholder data
        return {
            'analysis_note': 'Persona performance analysis requires more data collection',
            'available_personas': ['default', 'janelle', 'leader2']
        }
