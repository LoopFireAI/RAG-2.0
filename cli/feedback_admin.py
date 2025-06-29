#!/usr/bin/env python3
"""
Feedback Administration Tool - Cost-efficient feedback management.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_2_0.feedback.feedback_storage import FeedbackStorage
from rag_2_0.feedback.feedback_collector import FeedbackCollector
from rag_2_0.feedback.feedback_analytics import FeedbackAnalytics

def show_stats():
    """Show feedback statistics."""
    storage = FeedbackStorage()
    collector = FeedbackCollector(storage)
    analytics = FeedbackAnalytics(storage)
    
    print("=" * 50)
    print("FEEDBACK SYSTEM STATISTICS")
    print("=" * 50)
    
    # Basic stats
    stats = collector.get_feedback_summary()
    print(f"Total Feedback: {stats['total_feedback']}")
    print(f"Average Satisfaction: {stats['avg_satisfaction']}/5.0")
    print(f"Average Relevance: {stats['avg_relevance']}/3.0")
    print(f"Unique Queries: {stats['unique_queries']}")
    
    print("\n" + "-" * 30)
    print("IMPROVEMENT SUGGESTIONS")
    print("-" * 30)
    
    suggestions = collector.get_improvement_suggestions()
    for suggestion in suggestions:
        print(f"• {suggestion}")
    
    print("\n" + "-" * 30)
    print("LOW PERFORMING DOCUMENTS")
    print("-" * 30)
    
    low_docs = storage.get_low_performing_docs()
    if low_docs:
        for doc in low_docs[:5]:  # Show top 5
            print(f"• {doc['doc_title']} (avg: {doc['avg_score']}/3.0, count: {doc['feedback_count']})")
    else:
        print("• No consistently low-performing documents found")

def generate_report():
    """Generate detailed insights report."""
    storage = FeedbackStorage()
    analytics = FeedbackAnalytics(storage)
    
    report = analytics.generate_insights_report()
    
    print("=" * 50)
    print("FEEDBACK INSIGHTS REPORT")
    print("=" * 50)
    print(f"Generated: {report['generated_at']}")
    
    print(f"\nSummary:")
    summary = report['summary']
    print(f"  Total Feedback: {summary['total_feedback']}")
    print(f"  Avg Satisfaction: {summary['avg_satisfaction']}/5.0")
    print(f"  Avg Relevance: {summary['avg_relevance']}/3.0")
    
    if report['insights']:
        print(f"\nKey Insights:")
        for insight in report['insights']:
            priority_emoji = "🔴" if insight['priority'] == 'high' else "🟡" if insight['priority'] == 'medium' else "🟢"
            print(f"  {priority_emoji} {insight['message']}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec['action']}: {rec['description']}")

def export_data(days: int = 7):
    """Export feedback data for analysis."""
    storage = FeedbackStorage()
    analytics = FeedbackAnalytics(storage)
    
    export_data = analytics.export_feedback_for_analysis(days)
    
    filename = f"feedback_export_{days}days.json"
    with open(filename, 'w') as f:
        f.write(export_data)
    
    print(f"Feedback data exported to {filename}")

def reset_database():
    """Reset feedback database (use with caution)."""
    confirm = input("⚠️  This will DELETE all feedback data. Type 'CONFIRM' to proceed: ")
    if confirm == 'CONFIRM':
        # Reinitialize database (clears tables)
        storage = FeedbackStorage()
        storage.init_database()
        print("✅ Feedback database reset successfully")
    else:
        print("❌ Reset cancelled")

def main():
    parser = argparse.ArgumentParser(description="RAG 2.0 Feedback Administration")
    parser.add_argument('command', choices=['stats', 'report', 'export', 'reset'], 
                       help='Command to execute')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days for export (default: 7)')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'stats':
            show_stats()
        elif args.command == 'report':
            generate_report()
        elif args.command == 'export':
            export_data(args.days)
        elif args.command == 'reset':
            reset_database()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()