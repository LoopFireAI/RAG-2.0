#!/usr/bin/env python3
"""
KPI Dashboard - Command line interface for monitoring RAG system performance.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rag_2_0.feedback.feedback_storage import FeedbackStorage
from rag_2_0.feedback.kpi_monitor import KPIMonitor

def print_banner():
    """Print dashboard banner."""
    print("\n" + "="*70)
    print("🎯 RAG SYSTEM KPI DASHBOARD")
    print("="*70)

def print_current_kpis(monitor: KPIMonitor):
    """Print current KPI overview."""
    print("\n📊 CURRENT KPIs")
    print("-" * 50)
    
    kpis = monitor.get_current_kpis()
    
    print(f"Total Responses Generated: {kpis.total_responses:,}")
    print(f"Feedback Received: {kpis.total_feedback:,} ({kpis.response_rate}% response rate)")
    print(f"Average Satisfaction: {kpis.avg_satisfaction}/5.0")
    print(f"Average Relevance: {kpis.avg_relevance}/3.0")
    print(f"Success Rate: {kpis.success_rate}% (≥4.0 satisfaction)")
    print(f"Failure Rate: {kpis.failure_rate}% (≤2.0 satisfaction)")
    print(f"This Week's Avg: {kpis.weekly_avg_satisfaction}/5.0")
    print(f"Trend: {kpis.trend_direction} ({kpis.improvement_rate:+.1f}%)")

def print_weekly_breakdown(monitor: KPIMonitor, weeks: int = 4):
    """Print weekly performance breakdown."""
    print(f"\n📈 WEEKLY BREAKDOWN (Last {weeks} weeks)")
    print("-" * 50)
    
    weekly_data = monitor.get_weekly_metrics(weeks)
    
    if not weekly_data:
        print("No weekly data available yet.")
        return
    
    # Header
    print(f"{'Week':<8} {'Period':<20} {'Feedback':<10} {'Satisfaction':<12} {'Success%':<10} {'Failure%':<10}")
    print("-" * 80)
    
    # Data rows
    for week in weekly_data:
        period = f"{week['week_start'][5:]} to {week['week_end'][5:]}"
        print(f"{week['week_number']:<8} {period:<20} {week['feedback_count']:<10} "
              f"{week['avg_satisfaction']:<12.2f} {week['success_rate']:<10.1f} {week['failure_rate']:<10.1f}")

def print_pilot_summary(monitor: KPIMonitor, weeks: int = 4):
    """Print comprehensive pilot summary."""
    print(f"\n🎯 {weeks}-WEEK PILOT SUMMARY")
    print("-" * 50)
    
    summary = monitor.get_pilot_summary(weeks)
    
    print(f"Pilot Period: {summary['pilot_period']}")
    print(f"Total Responses: {summary['total_responses']:,}")
    print(f"Feedback Response Rate: {summary['feedback_response_rate']}%")
    print(f"Pilot Average Satisfaction: {summary['pilot_avg_satisfaction']}/5.0")
    print(f"Pilot Average Success Rate: {summary['pilot_avg_success_rate']}%")
    print(f"Performance Grade: {summary['performance_grade']}")
    print(f"Current Trend: {summary['current_trend']} ({summary['improvement_rate']:+.1f}%)")
    
    if summary['key_insights']:
        print("\n💡 Key Insights:")
        for insight in summary['key_insights']:
            print(f"  • {insight}")

def print_persona_performance(monitor: KPIMonitor):
    """Print performance by persona/leader."""
    print("\n👥 PERFORMANCE BY PERSONA")
    print("-" * 50)
    
    performance = monitor.get_persona_performance()
    
    if not performance['personas']:
        print("No persona performance data available yet.")
        return
    
    # Header
    print(f"{'Persona':<12} {'Feedback':<10} {'Satisfaction':<12} {'Success%':<10} {'Status':<15}")
    print("-" * 70)
    
    # Data rows
    for persona in performance['personas']:
        status = "🟢 Good" if persona['avg_satisfaction'] >= 3.5 else "🟡 Needs Work" if persona['avg_satisfaction'] >= 2.5 else "🔴 Critical"
        print(f"{persona['persona']:<12} {persona['feedback_count']:<10} "
              f"{persona['avg_satisfaction']:<12.2f} {persona['success_rate']:<10.1f} {status:<15}")
    
    if performance['best_performing']:
        print(f"\n🏆 Best Performing: {performance['best_performing']['persona']} "
              f"({performance['best_performing']['avg_satisfaction']}/5.0)")
    
    if performance['needs_improvement']:
        print("\n⚠️ Needs Improvement:")
        for persona in performance['needs_improvement']:
            print(f"  • {persona['persona']}: {persona['avg_satisfaction']}/5.0")

def print_alerts(monitor: KPIMonitor):
    """Print alert conditions."""
    print("\n⚠️ SYSTEM ALERTS")
    print("-" * 50)
    
    alerts = monitor.get_alert_conditions()
    
    if not alerts:
        print("✅ No alerts - system performing within normal parameters")
        return
    
    for alert in alerts:
        icon = "🔴" if alert['level'] == 'CRITICAL' else "🟡" if alert['level'] == 'HIGH' else "🟦" if alert['level'] == 'MEDIUM' else "⚪"
        print(f"\n{icon} {alert['level']}: {alert['type']}")
        print(f"   Issue: {alert['message']}")
        print(f"   Action: {alert['recommendation']}")

def save_report(monitor: KPIMonitor, filename: str = None):
    """Save full KPI report to file."""
    if filename is None:
        filename = f"kpi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    report = monitor.generate_kpi_report()
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"\n💾 Full KPI report saved to: {filename}")

def main():
    """Main dashboard function."""
    parser = argparse.ArgumentParser(description='RAG System KPI Dashboard')
    parser.add_argument('--weeks', type=int, default=4, help='Number of weeks to analyze (default: 4)')
    parser.add_argument('--report', action='store_true', help='Generate and save full report')
    parser.add_argument('--alerts-only', action='store_true', help='Show only alerts')
    parser.add_argument('--summary-only', action='store_true', help='Show only pilot summary')
    
    args = parser.parse_args()
    
    # Initialize monitoring system
    try:
        storage = FeedbackStorage()
        monitor = KPIMonitor(storage)
    except Exception as e:
        print(f"❌ Error initializing KPI monitor: {e}")
        sys.exit(1)
    
    # Print banner
    if not args.alerts_only and not args.summary_only:
        print_banner()
    
    # Handle different modes
    if args.alerts_only:
        print_alerts(monitor)
    elif args.summary_only:
        print_pilot_summary(monitor, args.weeks)
    else:
        # Full dashboard
        print_current_kpis(monitor)
        print_weekly_breakdown(monitor, args.weeks)
        print_pilot_summary(monitor, args.weeks)
        print_persona_performance(monitor)
        print_alerts(monitor)
    
    # Generate report if requested
    if args.report:
        save_report(monitor)
    
    print("\n" + "="*70)
    print(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()