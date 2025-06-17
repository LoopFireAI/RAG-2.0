"""
Comprehensive KPI monitoring system for RAG feedback tracking.
Tracks quantitative metrics, weekly averages, success rates, and trends.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import statistics
from dataclasses import dataclass

@dataclass
class KPIMetrics:
    """Data class for KPI metrics."""
    total_responses: int
    total_feedback: int
    avg_satisfaction: float
    avg_relevance: float
    success_rate: float
    failure_rate: float
    response_rate: float
    weekly_avg_satisfaction: float
    trend_direction: str
    improvement_rate: float

class KPIMonitor:
    """Monitors and tracks quantitative KPIs for the RAG system."""
    
    def __init__(self, storage):
        self.storage = storage
        self.db_path = storage.db_path
        self.success_threshold = 4.0  # 4+ out of 5 is considered success
        self.failure_threshold = 2.0  # 2 or below is considered failure
        
    def get_current_kpis(self) -> KPIMetrics:
        """Get current comprehensive KPI metrics."""
        with sqlite3.connect(self.db_path) as conn:
            # Basic stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(DISTINCT response_id) as total_responses,
                    COUNT(*) as total_feedback,
                    AVG(satisfaction_score) as avg_satisfaction,
                    AVG(relevance_score) as avg_relevance,
                    COUNT(CASE WHEN satisfaction_score >= ? THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(CASE WHEN satisfaction_score <= ? THEN 1 END) * 100.0 / COUNT(*) as failure_rate
                FROM feedback 
                WHERE satisfaction_score IS NOT NULL
            """, (self.success_threshold, self.failure_threshold))
            
            stats = cursor.fetchone()
            
            # Get total responses (including those without feedback)
            cursor = conn.execute("SELECT COUNT(*) FROM responses")
            total_responses = cursor.fetchone()[0]
            
            # Calculate response rate (feedback received / total responses)
            response_rate = (stats[1] / total_responses * 100.0) if total_responses > 0 else 0
            
            # Get weekly average for current week
            weekly_avg = self._get_weekly_average()
            
            # Get trend direction
            trend_direction, improvement_rate = self._calculate_trend()
            
            return KPIMetrics(
                total_responses=total_responses,
                total_feedback=stats[1] or 0,
                avg_satisfaction=round(stats[2] or 0, 2),
                avg_relevance=round(stats[3] or 0, 2),
                success_rate=round(stats[4] or 0, 2),
                failure_rate=round(stats[5] or 0, 2),
                response_rate=round(response_rate, 2),
                weekly_avg_satisfaction=round(weekly_avg, 2),
                trend_direction=trend_direction,
                improvement_rate=round(improvement_rate, 2)
            )
    
    def get_weekly_metrics(self, weeks_back: int = 4) -> List[Dict[str, Any]]:
        """Get weekly breakdown of metrics for the specified number of weeks."""
        weekly_data = []
        
        with sqlite3.connect(self.db_path) as conn:
            for week in range(weeks_back):
                start_date = datetime.now() - timedelta(weeks=week+1)
                end_date = datetime.now() - timedelta(weeks=week)
                
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as feedback_count,
                        AVG(satisfaction_score) as avg_satisfaction,
                        AVG(relevance_score) as avg_relevance,
                        COUNT(CASE WHEN satisfaction_score >= ? THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                        COUNT(CASE WHEN satisfaction_score <= ? THEN 1 END) * 100.0 / COUNT(*) as failure_rate,
                        MIN(satisfaction_score) as min_satisfaction,
                        MAX(satisfaction_score) as max_satisfaction
                    FROM feedback 
                    WHERE timestamp BETWEEN ? AND ?
                    AND satisfaction_score IS NOT NULL
                """, (self.success_threshold, self.failure_threshold, 
                      start_date.isoformat(), end_date.isoformat()))
                
                stats = cursor.fetchone()
                
                weekly_data.append({
                    'week_start': start_date.strftime('%Y-%m-%d'),
                    'week_end': end_date.strftime('%Y-%m-%d'),
                    'week_number': week + 1,
                    'feedback_count': stats[0] or 0,
                    'avg_satisfaction': round(stats[1] or 0, 2),
                    'avg_relevance': round(stats[2] or 0, 2),
                    'success_rate': round(stats[3] or 0, 2),
                    'failure_rate': round(stats[4] or 0, 2),
                    'min_satisfaction': stats[5] or 0,
                    'max_satisfaction': stats[6] or 0
                })
        
        return list(reversed(weekly_data))  # Most recent first
    
    def get_pilot_summary(self, pilot_weeks: int = 4) -> Dict[str, Any]:
        """Get comprehensive pilot program summary."""
        current_kpis = self.get_current_kpis()
        weekly_metrics = self.get_weekly_metrics(pilot_weeks)
        
        # Calculate pilot averages
        pilot_satisfactions = [week['avg_satisfaction'] for week in weekly_metrics if week['avg_satisfaction'] > 0]
        pilot_success_rates = [week['success_rate'] for week in weekly_metrics if week['feedback_count'] > 0]
        
        pilot_avg_satisfaction = statistics.mean(pilot_satisfactions) if pilot_satisfactions else 0
        pilot_avg_success_rate = statistics.mean(pilot_success_rates) if pilot_success_rates else 0
        
        # Performance assessment
        performance_grade = self._assess_performance(pilot_avg_satisfaction, pilot_avg_success_rate)
        
        return {
            'pilot_duration_weeks': pilot_weeks,
            'pilot_period': f"{weekly_metrics[-1]['week_start']} to {weekly_metrics[0]['week_end']}" if weekly_metrics else "No data",
            'total_responses': current_kpis.total_responses,
            'total_feedback_received': current_kpis.total_feedback,
            'feedback_response_rate': current_kpis.response_rate,
            'pilot_avg_satisfaction': round(pilot_avg_satisfaction, 2),
            'pilot_avg_success_rate': round(pilot_avg_success_rate, 2),
            'current_trend': current_kpis.trend_direction,
            'improvement_rate': current_kpis.improvement_rate,
            'performance_grade': performance_grade,
            'weekly_breakdown': weekly_metrics,
            'key_insights': self._generate_insights(weekly_metrics, current_kpis)
        }
    
    def get_persona_performance(self) -> Dict[str, Any]:
        """Analyze performance by persona/leader."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    persona,
                    COUNT(*) as feedback_count,
                    AVG(satisfaction_score) as avg_satisfaction,
                    AVG(relevance_score) as avg_relevance,
                    COUNT(CASE WHEN satisfaction_score >= ? THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(CASE WHEN satisfaction_score <= ? THEN 1 END) * 100.0 / COUNT(*) as failure_rate
                FROM feedback 
                WHERE satisfaction_score IS NOT NULL
                GROUP BY persona
                ORDER BY avg_satisfaction DESC
            """, (self.success_threshold, self.failure_threshold))
            
            persona_data = []
            for row in cursor.fetchall():
                persona_data.append({
                    'persona': row[0],
                    'feedback_count': row[1],
                    'avg_satisfaction': round(row[2], 2),
                    'avg_relevance': round(row[3], 2),
                    'success_rate': round(row[4], 2),
                    'failure_rate': round(row[5], 2)
                })
            
            return {
                'personas': persona_data,
                'best_performing': persona_data[0] if persona_data else None,
                'needs_improvement': [p for p in persona_data if p['avg_satisfaction'] < 3.5]
            }
    
    def get_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for alert conditions that require attention."""
        alerts = []
        current_kpis = self.get_current_kpis()
        weekly_metrics = self.get_weekly_metrics(2)  # Last 2 weeks
        
        # Critical satisfaction alert
        if current_kpis.avg_satisfaction < 2.5:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'LOW_SATISFACTION',
                'message': f'Average satisfaction critically low: {current_kpis.avg_satisfaction}/5.0',
                'recommendation': 'Immediate review of response quality and content needed'
            })
        
        # High failure rate alert
        if current_kpis.failure_rate > 25:
            alerts.append({
                'level': 'HIGH',
                'type': 'HIGH_FAILURE_RATE',
                'message': f'Failure rate too high: {current_kpis.failure_rate}%',
                'recommendation': 'Review and improve response generation processes'
            })
        
        # Declining trend alert
        if current_kpis.trend_direction == 'declining' and current_kpis.improvement_rate < -10:
            alerts.append({
                'level': 'MEDIUM',
                'type': 'DECLINING_PERFORMANCE',
                'message': f'Performance declining: {current_kpis.improvement_rate}% change',
                'recommendation': 'Investigate recent changes and user feedback patterns'
            })
        
        # Low response rate alert
        if current_kpis.response_rate < 10:  # Less than 10% feedback rate
            alerts.append({
                'level': 'LOW',
                'type': 'LOW_FEEDBACK_RATE',
                'message': f'Low feedback collection rate: {current_kpis.response_rate}%',
                'recommendation': 'Consider improving feedback collection prompts'
            })
        
        return alerts
    
    def generate_kpi_report(self) -> str:
        """Generate a comprehensive KPI report."""
        pilot_summary = self.get_pilot_summary()
        persona_performance = self.get_persona_performance()
        alerts = self.get_alert_conditions()
        
        report = f"""
# RAG System KPI Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Pilot Program Summary ({pilot_summary['pilot_duration_weeks']} weeks)
- **Period**: {pilot_summary['pilot_period']}
- **Total Responses**: {pilot_summary['total_responses']:,}
- **Feedback Received**: {pilot_summary['total_feedback_received']:,} ({pilot_summary['feedback_response_rate']}% response rate)
- **Overall Satisfaction**: {pilot_summary['pilot_avg_satisfaction']}/5.0
- **Success Rate**: {pilot_summary['pilot_avg_success_rate']}%
- **Performance Grade**: {pilot_summary['performance_grade']}
- **Trend**: {pilot_summary['current_trend']} ({pilot_summary['improvement_rate']:+.1f}%)

## 📈 Weekly Breakdown
"""
        
        for week in pilot_summary['weekly_breakdown']:
            report += f"""
### Week {week['week_number']} ({week['week_start']} to {week['week_end']})
- Feedback Count: {week['feedback_count']}
- Avg Satisfaction: {week['avg_satisfaction']}/5.0
- Success Rate: {week['success_rate']}%
- Failure Rate: {week['failure_rate']}%
"""
        
        report += f"""
## 👥 Performance by Persona
"""
        for persona in persona_performance['personas']:
            report += f"""
### {persona['persona'].title()}
- Satisfaction: {persona['avg_satisfaction']}/5.0
- Success Rate: {persona['success_rate']}%
- Feedback Count: {persona['feedback_count']}
"""
        
        if alerts:
            report += f"""
## ⚠️ Alerts & Recommendations
"""
            for alert in alerts:
                report += f"""
### {alert['level']}: {alert['type']}
- **Issue**: {alert['message']}
- **Recommendation**: {alert['recommendation']}
"""
        
        report += f"""
## 💡 Key Insights
"""
        for insight in pilot_summary['key_insights']:
            report += f"- {insight}\n"
        
        return report
    
    def _get_weekly_average(self) -> float:
        """Get current week's average satisfaction."""
        week_start = datetime.now() - timedelta(days=7)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT AVG(satisfaction_score)
                FROM feedback 
                WHERE timestamp >= ? AND satisfaction_score IS NOT NULL
            """, (week_start.isoformat(),))
            
            result = cursor.fetchone()
            return result[0] or 0
    
    def _calculate_trend(self) -> Tuple[str, float]:
        """Calculate trend direction and improvement rate."""
        weekly_metrics = self.get_weekly_metrics(4)
        
        if len(weekly_metrics) < 2:
            return "insufficient_data", 0.0
        
        # Get satisfaction scores for last 2 weeks vs previous 2 weeks
        recent_weeks = weekly_metrics[:2]  # Most recent 2 weeks
        older_weeks = weekly_metrics[2:4]  # Previous 2 weeks
        
        recent_satisfactions = [w['avg_satisfaction'] for w in recent_weeks if w['avg_satisfaction'] > 0]
        older_satisfactions = [w['avg_satisfaction'] for w in older_weeks if w['avg_satisfaction'] > 0]
        
        if not recent_satisfactions or not older_satisfactions:
            return "insufficient_data", 0.0
        
        recent_avg = statistics.mean(recent_satisfactions)
        older_avg = statistics.mean(older_satisfactions)
        
        if recent_avg > older_avg:
            improvement_rate = ((recent_avg - older_avg) / older_avg) * 100
            return "improving", improvement_rate
        elif recent_avg < older_avg:
            decline_rate = ((older_avg - recent_avg) / older_avg) * 100
            return "declining", -decline_rate
        else:
            return "stable", 0.0
    
    def _assess_performance(self, avg_satisfaction: float, success_rate: float) -> str:
        """Assess overall performance grade."""
        if avg_satisfaction >= 4.0 and success_rate >= 75:
            return "A - Excellent"
        elif avg_satisfaction >= 3.5 and success_rate >= 60:
            return "B - Good"
        elif avg_satisfaction >= 3.0 and success_rate >= 45:
            return "C - Average"
        elif avg_satisfaction >= 2.5 and success_rate >= 30:
            return "D - Below Average"
        else:
            return "F - Needs Immediate Attention"
    
    def _generate_insights(self, weekly_metrics: List[Dict], current_kpis: KPIMetrics) -> List[str]:
        """Generate actionable insights from the data."""
        insights = []
        
        if current_kpis.total_feedback < 20:
            insights.append("📊 Limited feedback data - consider encouraging more user participation")
        
        if current_kpis.success_rate > 70:
            insights.append("✅ High success rate indicates strong user satisfaction")
        elif current_kpis.success_rate < 40:
            insights.append("⚠️ Low success rate - focus on improving response quality")
        
        if current_kpis.trend_direction == "improving":
            insights.append("📈 Positive trend - system is improving over time")
        elif current_kpis.trend_direction == "declining":
            insights.append("📉 Declining performance - investigate recent changes")
        
        # Check consistency across weeks
        weekly_satisfactions = [w['avg_satisfaction'] for w in weekly_metrics if w['avg_satisfaction'] > 0]
        if len(weekly_satisfactions) >= 3:
            std_dev = statistics.stdev(weekly_satisfactions)
            if std_dev > 0.5:
                insights.append("🔄 High variability in weekly performance - focus on consistency")
            else:
                insights.append("🎯 Consistent performance across weeks")
        
        return insights