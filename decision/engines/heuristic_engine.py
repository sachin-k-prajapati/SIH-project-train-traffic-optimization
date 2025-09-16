from datetime import datetime, timedelta
from core.models import Train, Decision
from django.utils import timezone

class HeuristicEngine:
    def __init__(self, section):
        self.section = section
    
    def decide_precedence(self, active_trains, current_time):
        """Rule-based precedence decision making with enhanced logic"""
        if not active_trains:
            return []
        
        # Calculate metrics for each train
        train_metrics = []
        for train in active_trains:
            metrics = {
                'train': train,
                'lateness': self.calculate_lateness(train, current_time),
                'priority_score': 6 - train.priority,  # Higher priority = higher score
                'type_priority': self.get_type_priority(train.train_type),
                'estimated_impact': self.estimate_impact(train, active_trains)
            }
            metrics['total_score'] = (
                metrics['priority_score'] * 0.4 +
                metrics['type_priority'] * 0.3 +
                (metrics['lateness'] / 60) * 0.2 +  # Convert minutes to weight
                metrics['estimated_impact'] * 0.1
            )
            train_metrics.append(metrics)
        
        # Sort by total score (higher score = higher precedence)
        sorted_trains = sorted(train_metrics, key=lambda x: x['total_score'], reverse=True)
        
        decisions = []
        for i in range(1, len(sorted_trains)):
            current_train = sorted_trains[i]['train']
            preceding_train = sorted_trains[i-1]['train']
            
            decision_data = {
                'type': 'precedence',
                'trains': [preceding_train.id, current_train.id],
                'action': f'Train {preceding_train.train_id} should precede {current_train.train_id}',
                'reason': self.generate_reason(preceding_train, current_train, sorted_trains[i-1], sorted_trains[i])
            }
            
            decisions.append(decision_data)
            
            # Create decision record
            self.record_decision(decision_data, current_time)
        
        return decisions
    
    def calculate_lateness(self, train, current_time):
        """Calculate how late a train is in minutes"""
        if train.scheduled_arrival:
            scheduled_time = train.scheduled_arrival
            if timezone.is_naive(scheduled_time):
                scheduled_time = timezone.make_aware(scheduled_time)
            return max(0, (current_time - scheduled_time).total_seconds() / 60)
        return 0
    
    def get_type_priority(self, train_type):
        """Get priority based on train type"""
        type_priorities = {
            'express': 5,
            'special': 4,
            'local': 3,
            'freight': 2
        }
        return type_priorities.get(train_type, 1)
    
    def estimate_impact(self, train, all_trains):
        """Estimate the impact of delaying this train"""
        # Simple heuristic: express trains have higher impact when delayed
        impact_scores = {
            'express': 0.9,
            'special': 0.8,
            'local': 0.5,
            'freight': 0.3
        }
        return impact_scores.get(train.train_type, 0.1)
    
    def generate_reason(self, train1, train2, metrics1, metrics2):
        """Generate a human-readable reason for the precedence decision"""
        reasons = []
        
        if metrics1['priority_score'] > metrics2['priority_score']:
            reasons.append(f"Higher priority ({train1.priority} vs {train2.priority})")
        
        if metrics1['type_priority'] > metrics2['type_priority']:
            reasons.append(f"Train type precedence ({train1.train_type} vs {train2.train_type})")
        
        if metrics1['lateness'] > metrics2['lateness'] + 10:  # 10 minutes threshold
            reasons.append(f"More delayed ({metrics1['lateness']:.1f}min vs {metrics2['lateness']:.1f}min)")
        
        if reasons:
            return "; ".join(reasons)
        return "Operational efficiency consideration"
    
    def record_decision(self, decision_data, timestamp):
        """Record the decision in the database"""
        try:
            trains = Train.objects.filter(id__in=decision_data['trains'])
            decision = Decision.objects.create(
                decision_type=decision_data['type'],
                recommended_action=decision_data['action'],
                explanation=decision_data['reason'],
                timestamp=timestamp
            )
            decision.trains_involved.set(trains)
            return decision
        except Exception as e:
            print(f"Error recording decision: {e}")
            return None