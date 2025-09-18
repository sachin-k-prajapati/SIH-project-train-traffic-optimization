"""
AI-Powered Traffic Controller for Railway Optimization
Implements machine learning-based decision support for train traffic control
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import logging
from datetime import datetime, timedelta
from django.utils import timezone
from django.conf import settings
import os
import json

from core.models import Section, Train, Segment, WeatherCondition, EmergencyEvent
from .ilp_engine_enhanced import AdvancedILPEngine

logger = logging.getLogger(__name__)

class AITrafficController:
    """
    AI-powered traffic controller using machine learning for intelligent decisions
    """
    
    def __init__(self, section):
        self.section = section
        self.ilp_engine = AdvancedILPEngine(section)
        
        # ML Models
        self.precedence_model = None  # For train precedence decisions
        self.delay_predictor = None   # For delay prediction
        self.conflict_detector = None # For conflict detection
        self.throughput_optimizer = None # For throughput optimization
        
        # Feature scalers
        self.scaler_precedence = StandardScaler()
        self.scaler_delay = StandardScaler()
        self.scaler_conflict = StandardScaler()
        
        # Model paths
        self.model_dir = os.path.join(settings.BASE_DIR, 'ml_models', f'section_{section.id}')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Training data storage
        self.training_data = {
            'precedence': [],
            'delays': [],
            'conflicts': [],
            'throughput': []
        }
        
        # Performance metrics
        self.prediction_accuracy = {}
        self.decision_history = []
        
        logger.info(f"Initialized AI Traffic Controller for {section.name}")
        self._load_models()
    
    def make_intelligent_decision(self, trains, current_conditions=None, emergency_events=None):
        """
        Make intelligent traffic control decisions using AI and optimization
        """
        logger.info(f"Making AI-powered decision for {len(trains)} trains")
        
        start_time = timezone.now()
        
        # Analyze current situation
        situation_analysis = self._analyze_traffic_situation(trains, current_conditions, emergency_events)
        
        # Predict potential issues
        predictions = self._predict_issues(trains, situation_analysis)
        
        # Generate precedence recommendations
        precedence_decisions = self._generate_precedence_decisions(trains, situation_analysis)
        
        # Optimize using ILP with AI insights
        optimization_params = self._prepare_optimization_params(
            trains, precedence_decisions, predictions
        )
        
        # Run optimization
        optimization_result = self.ilp_engine.optimize_comprehensive_schedule(
            trains, emergency_events, **optimization_params
        )
        
        # Enhance with AI insights
        enhanced_solution = self._enhance_with_ai_insights(
            optimization_result, situation_analysis, predictions
        )
        
        # Learn from this decision
        self._record_decision(trains, enhanced_solution, situation_analysis)
        
        decision_time = (timezone.now() - start_time).total_seconds()
        enhanced_solution['ai_decision_time'] = decision_time
        
        logger.info(f"AI decision completed in {decision_time:.2f}s")
        return enhanced_solution
    
    def _analyze_traffic_situation(self, trains, current_conditions, emergency_events):
        """
        Analyze current traffic situation using pattern recognition
        """
        analysis = {
            'train_mix': self._analyze_train_mix(trains),
            'congestion_level': self._assess_congestion(trains),
            'weather_impact': self._assess_weather_impact(current_conditions),
            'emergency_impact': self._assess_emergency_impact(emergency_events),
            'historical_patterns': self._identify_patterns(trains),
            'bottleneck_risk': self._assess_bottleneck_risk(trains),
            'priority_conflicts': self._identify_priority_conflicts(trains)
        }
        
        return analysis
    
    def _analyze_train_mix(self, trains):
        """Analyze the composition of trains"""
        train_types = {}
        priority_distribution = {}
        
        for train in trains:
            train_types[train.train_type] = train_types.get(train.train_type, 0) + 1
            priority_distribution[train.priority] = priority_distribution.get(train.priority, 0) + 1
        
        return {
            'types': train_types,
            'priorities': priority_distribution,
            'total_count': len(trains),
            'diversity_score': len(train_types) / max(1, len(trains)),
            'high_priority_ratio': priority_distribution.get(1, 0) / max(1, len(trains))
        }
    
    def _assess_congestion(self, trains):
        """Assess current congestion level"""
        capacity_utilization = len(trains) / self.section.capacity if self.section.capacity > 0 else 1
        
        congestion_score = min(1.0, capacity_utilization)
        
        if congestion_score >= 0.9:
            level = 'critical'
        elif congestion_score >= 0.7:
            level = 'high'
        elif congestion_score >= 0.5:
            level = 'moderate'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': congestion_score,
            'utilization_percent': capacity_utilization * 100
        }
    
    def _assess_weather_impact(self, current_conditions):
        """Assess weather impact on operations"""
        if not current_conditions:
            # Use section's current weather
            weather_factor = self.section.weather_impact_factor
            condition = getattr(self.section, 'current_weather', 'clear')
        else:
            weather_factor = getattr(current_conditions, 'visibility_factor', 1.0)
            condition = getattr(current_conditions, 'condition', 'clear')
        
        impact_level = 'low'
        if weather_factor < 0.6:
            impact_level = 'severe'
        elif weather_factor < 0.8:
            impact_level = 'moderate'
        
        return {
            'condition': condition,
            'impact_factor': weather_factor,
            'impact_level': impact_level,
            'speed_reduction_percent': (1 - weather_factor) * 100
        }
    
    def _assess_emergency_impact(self, emergency_events):
        """Assess impact of emergency events"""
        if not emergency_events:
            return {'level': 'none', 'affected_segments': [], 'severity': 0}
        
        affected_segments = []
        total_severity = 0
        
        for event in emergency_events:
            if hasattr(event, 'affected_segment'):
                affected_segments.append(event.affected_segment.id)
            severity = getattr(event, 'severity', 1)
            total_severity += severity
        
        avg_severity = total_severity / len(emergency_events) if emergency_events else 0
        
        level = 'critical' if avg_severity >= 3 else 'moderate' if avg_severity >= 2 else 'minor'
        
        return {
            'level': level,
            'affected_segments': affected_segments,
            'severity': avg_severity,
            'event_count': len(emergency_events)
        }
    
    def _identify_patterns(self, trains):
        """Identify historical patterns that may affect decisions"""
        current_hour = timezone.now().hour
        
        # Simple pattern recognition based on historical data
        patterns = {
            'peak_hour': current_hour in [7, 8, 9, 17, 18, 19],  # Rush hours
            'freight_heavy': len([t for t in trains if t.train_type == 'freight']) > len(trains) * 0.3,
            'express_priority': len([t for t in trains if t.train_type == 'express' and t.priority <= 2]) > 0,
            'delay_cascade_risk': any(t.current_delay_minutes > 10 for t in trains)
        }
        
        return patterns
    
    def _assess_bottleneck_risk(self, trains):
        """Assess risk of bottlenecks forming"""
        bottleneck_segments = self.ilp_engine.bottleneck_segments
        
        risk_score = 0
        for segment in bottleneck_segments:
            # Calculate potential load on this segment
            segment_load = len([t for t in trains if self._train_uses_segment(t, segment)])
            capacity_ratio = segment_load / segment.max_occupancy if segment.max_occupancy > 0 else 1
            
            if capacity_ratio > 0.8:
                risk_score += 3
            elif capacity_ratio > 0.6:
                risk_score += 2
            elif capacity_ratio > 0.4:
                risk_score += 1
        
        risk_level = 'high' if risk_score >= 6 else 'medium' if risk_score >= 3 else 'low'
        
        return {
            'level': risk_level,
            'score': risk_score,
            'bottleneck_segments': [s.id for s in bottleneck_segments]
        }
    
    def _identify_priority_conflicts(self, trains):
        """Identify potential conflicts between high-priority trains"""
        high_priority_trains = [t for t in trains if t.priority <= 2]
        
        conflicts = []
        for i, train1 in enumerate(high_priority_trains):
            for train2 in high_priority_trains[i+1:]:
                # Check if trains might conflict based on timing and routes
                time_diff = abs(
                    (train1.scheduled_arrival - train2.scheduled_arrival).total_seconds() / 60
                )
                
                if time_diff <= 15:  # Within 15 minutes
                    conflicts.append({
                        'train1': train1.train_id,
                        'train2': train2.train_id,
                        'time_difference': time_diff,
                        'severity': 'high' if time_diff <= 5 else 'medium'
                    })
        
        return conflicts
    
    def _train_uses_segment(self, train, segment):
        """Check if train route uses a specific segment"""
        # Simplified: assume all trains use all segments in order
        # In real implementation, this would check actual routes
        return True
    
    def _predict_issues(self, trains, situation_analysis):
        """
        Predict potential issues using trained ML models
        """
        predictions = {
            'delay_predictions': {},
            'conflict_probabilities': {},
            'throughput_forecast': None,
            'bottleneck_alerts': []
        }
        
        # Predict delays for each train
        if self.delay_predictor:
            for train in trains:
                features = self._extract_delay_features(train, situation_analysis)
                predicted_delay = self.delay_predictor.predict([features])[0]
                
                predictions['delay_predictions'][train.id] = {
                    'predicted_delay': max(0, predicted_delay),
                    'confidence': self._calculate_prediction_confidence(features, 'delay'),
                    'factors': self._identify_delay_factors(features)
                }
        
        # Predict conflicts
        if self.conflict_detector:
            for i, train1 in enumerate(trains):
                for train2 in trains[i+1:]:
                    features = self._extract_conflict_features(train1, train2, situation_analysis)
                    conflict_prob = self.conflict_detector.predict_proba([features])[0][1]
                    
                    if conflict_prob > 0.3:  # Threshold for concerning conflicts
                        predictions['conflict_probabilities'][(train1.id, train2.id)] = {
                            'probability': conflict_prob,
                            'severity': 'high' if conflict_prob > 0.7 else 'medium',
                            'factors': self._identify_conflict_factors(features)
                        }
        
        # Predict throughput
        if situation_analysis['congestion_level']['score'] > 0.5:
            throughput_features = self._extract_throughput_features(trains, situation_analysis)
            if self.throughput_optimizer:
                predicted_throughput = self.throughput_optimizer.predict([throughput_features])[0]
                predictions['throughput_forecast'] = {
                    'predicted_rate': predicted_throughput,
                    'efficiency_score': min(100, predicted_throughput / len(trains) * 100),
                    'recommendations': self._generate_throughput_recommendations(throughput_features)
                }
        
        return predictions
    
    def _extract_delay_features(self, train, situation_analysis):
        """Extract features for delay prediction"""
        return [
            train.priority,
            train.current_delay_minutes,
            1 if train.train_type == 'express' else 0,
            1 if train.train_type == 'freight' else 0,
            situation_analysis['weather_impact']['impact_factor'],
            situation_analysis['congestion_level']['score'],
            1 if situation_analysis['historical_patterns']['peak_hour'] else 0,
            situation_analysis['emergency_impact']['severity']
        ]
    
    def _extract_conflict_features(self, train1, train2, situation_analysis):
        """Extract features for conflict prediction"""
        time_diff = abs(
            (train1.scheduled_arrival - train2.scheduled_arrival).total_seconds() / 60
        )
        
        return [
            min(train1.priority, train2.priority),  # Higher priority = lower number
            max(train1.priority, train2.priority),
            time_diff,
            train1.current_delay_minutes + train2.current_delay_minutes,
            1 if train1.train_type == train2.train_type else 0,
            situation_analysis['congestion_level']['score'],
            situation_analysis['bottleneck_risk']['score'],
            situation_analysis['weather_impact']['impact_factor']
        ]
    
    def _extract_throughput_features(self, trains, situation_analysis):
        """Extract features for throughput prediction"""
        return [
            len(trains),
            situation_analysis['train_mix']['diversity_score'],
            situation_analysis['congestion_level']['score'],
            situation_analysis['weather_impact']['impact_factor'],
            situation_analysis['emergency_impact']['severity'],
            situation_analysis['bottleneck_risk']['score'],
            len([t for t in trains if t.priority == 1]),  # High priority count
            len([t for t in trains if t.train_type == 'freight'])  # Freight count
        ]
    
    def _generate_precedence_decisions(self, trains, situation_analysis):
        """
        Generate intelligent precedence decisions using ML
        """
        decisions = {}
        
        # Sort trains by priority and delay
        sorted_trains = sorted(trains, key=lambda t: (t.priority, -t.current_delay_minutes))
        
        for i, train1 in enumerate(sorted_trains):
            for train2 in sorted_trains[i+1:]:
                # Use ML model if available, otherwise use heuristic
                if self.precedence_model:
                    features = self._extract_precedence_features(train1, train2, situation_analysis)
                    decision_prob = self.precedence_model.predict_proba([features])[0]
                    
                    # decision_prob[1] is probability that train1 should precede train2
                    should_precede = decision_prob[1] > 0.5
                    confidence = max(decision_prob)
                else:
                    # Heuristic decision
                    should_precede = self._heuristic_precedence(train1, train2, situation_analysis)
                    confidence = 0.7
                
                decisions[(train1.id, train2.id)] = {
                    'train1_precedes': should_precede,
                    'confidence': confidence,
                    'factors': self._explain_precedence_decision(train1, train2, should_precede)
                }
        
        return decisions
    
    def _extract_precedence_features(self, train1, train2, situation_analysis):
        """Extract features for precedence decision"""
        return [
            train1.priority - train2.priority,  # Priority difference
            train1.current_delay_minutes - train2.current_delay_minutes,  # Delay difference
            1 if train1.train_type == 'express' else 0,
            1 if train2.train_type == 'express' else 0,
            1 if train1.train_type == 'freight' else 0,
            1 if train2.train_type == 'freight' else 0,
            situation_analysis['congestion_level']['score'],
            situation_analysis['weather_impact']['impact_factor']
        ]
    
    def _heuristic_precedence(self, train1, train2, situation_analysis):
        """Heuristic precedence decision when ML model not available"""
        # Priority first
        if train1.priority != train2.priority:
            return train1.priority < train2.priority
        
        # Then by delay (more delayed goes first to catch up)
        if abs(train1.current_delay_minutes - train2.current_delay_minutes) > 5:
            return train1.current_delay_minutes > train2.current_delay_minutes
        
        # Express trains before others
        if train1.train_type != train2.train_type:
            if train1.train_type == 'express':
                return True
            if train2.train_type == 'express':
                return False
        
        # Default: first come, first served
        return train1.scheduled_arrival <= train2.scheduled_arrival
    
    def _prepare_optimization_params(self, trains, precedence_decisions, predictions):
        """Prepare parameters for ILP optimization based on AI insights"""
        params = {
            'time_horizon_minutes': 180
        }
        
        # Adjust time horizon based on congestion
        if len(trains) > self.section.capacity * 0.8:
            params['time_horizon_minutes'] = 240  # Extend for high load
        
        return params
    
    def _enhance_with_ai_insights(self, optimization_result, situation_analysis, predictions):
        """Enhance optimization result with AI insights"""
        enhanced = optimization_result.copy()
        
        # Add AI analysis
        enhanced['ai_analysis'] = {
            'situation_assessment': situation_analysis,
            'predictions': predictions,
            'confidence_level': self._calculate_overall_confidence(predictions),
            'risk_factors': self._identify_risk_factors(situation_analysis, predictions),
            'optimization_quality': self._assess_optimization_quality(optimization_result)
        }
        
        # Enhance recommendations with AI insights
        ai_recommendations = self._generate_ai_recommendations(
            situation_analysis, predictions, optimization_result
        )
        
        enhanced['recommendations'].extend(ai_recommendations)
        
        # Add explainability
        enhanced['decision_explanation'] = self._generate_decision_explanation(
            optimization_result, situation_analysis, predictions
        )
        
        return enhanced
    
    def _generate_ai_recommendations(self, situation_analysis, predictions, optimization_result):
        """Generate AI-specific recommendations"""
        recommendations = []
        
        # High delay risk recommendations
        for train_id, delay_pred in predictions.get('delay_predictions', {}).items():
            if delay_pred['predicted_delay'] > 15:
                recommendations.append({
                    'type': 'ai_delay_alert',
                    'priority': 'high',
                    'message': f"AI predicts {delay_pred['predicted_delay']:.1f} min delay for train {train_id}",
                    'action': 'Consider priority adjustment or alternative routing',
                    'confidence': delay_pred['confidence'],
                    'source': 'AI_Predictor'
                })
        
        # Conflict prevention recommendations
        for (t1, t2), conflict_data in predictions.get('conflict_probabilities', {}).items():
            if conflict_data['probability'] > 0.6:
                recommendations.append({
                    'type': 'ai_conflict_alert',
                    'priority': 'high',
                    'message': f"High conflict probability ({conflict_data['probability']:.1f}) between trains {t1} and {t2}",
                    'action': 'Implement strict separation timing',
                    'confidence': 0.8,
                    'source': 'AI_Conflict_Detector'
                })
        
        # Throughput optimization recommendations
        if predictions.get('throughput_forecast'):
            forecast = predictions['throughput_forecast']
            if forecast['efficiency_score'] < 70:
                recommendations.append({
                    'type': 'ai_throughput_warning',
                    'priority': 'medium',
                    'message': f"AI predicts low throughput efficiency: {forecast['efficiency_score']:.1f}%",
                    'action': 'Consider load balancing or schedule adjustment',
                    'confidence': 0.75,
                    'source': 'AI_Throughput_Optimizer'
                })
        
        return recommendations
    
    def _generate_decision_explanation(self, optimization_result, situation_analysis, predictions):
        """Generate human-readable explanation of AI decisions"""
        explanation = {
            'summary': f"AI analysis of {len(optimization_result.get('trains', {}))} trains",
            'key_factors': [],
            'decision_rationale': [],
            'confidence_factors': []
        }
        
        # Key factors that influenced the decision
        if situation_analysis['congestion_level']['level'] != 'low':
            explanation['key_factors'].append(
                f"High congestion level: {situation_analysis['congestion_level']['level']}"
            )
        
        if situation_analysis['weather_impact']['impact_level'] != 'low':
            explanation['key_factors'].append(
                f"Weather impact: {situation_analysis['weather_impact']['impact_level']}"
            )
        
        if situation_analysis['emergency_impact']['level'] != 'none':
            explanation['key_factors'].append(
                f"Emergency events: {situation_analysis['emergency_impact']['level']} impact"
            )
        
        # Decision rationale
        explanation['decision_rationale'].append(
            f"Optimization completed in {optimization_result.get('solve_time', 0):.2f}s"
        )
        
        if optimization_result.get('status') == 'optimal':
            explanation['decision_rationale'].append("Optimal solution found")
        else:
            explanation['decision_rationale'].append("Feasible solution found within time limit")
        
        return explanation
    
    def train_models(self, historical_data=None):
        """Train ML models using historical data"""
        logger.info("Training AI models...")
        
        if historical_data is None:
            historical_data = self._collect_historical_data()
        
        if not historical_data:
            logger.warning("No historical data available for training")
            return
        
        # Train precedence model
        self._train_precedence_model(historical_data)
        
        # Train delay predictor
        self._train_delay_predictor(historical_data)
        
        # Train conflict detector
        self._train_conflict_detector(historical_data)
        
        # Save models
        self._save_models()
        
        logger.info("AI model training completed")
    
    def _train_precedence_model(self, data):
        """Train the precedence decision model"""
        if 'precedence' not in data or len(data['precedence']) < 10:
            logger.warning("Insufficient data for precedence model training")
            return
        
        df = pd.DataFrame(data['precedence'])
        
        feature_columns = [
            'priority_diff', 'delay_diff', 'train1_express', 'train2_express',
            'train1_freight', 'train2_freight', 'congestion', 'weather_factor'
        ]
        
        X = df[feature_columns].values
        y = df['train1_precedes'].values
        
        X_scaled = self.scaler_precedence.fit_transform(X)
        
        self.precedence_model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        
        self.precedence_model.fit(X_scaled, y)
        
        # Calculate accuracy
        y_pred = self.precedence_model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        self.prediction_accuracy['precedence'] = accuracy
        
        logger.info(f"Precedence model trained with accuracy: {accuracy:.3f}")
    
    def _record_decision(self, trains, solution, situation_analysis):
        """Record decision for future learning"""
        decision_record = {
            'timestamp': timezone.now(),
            'num_trains': len(trains),
            'situation': situation_analysis,
            'solution_quality': solution.get('status'),
            'solve_time': solution.get('solve_time', 0),
            'throughput': solution.get('kpis', {}).get('throughput_per_hour', 0)
        }
        
        self.decision_history.append(decision_record)
        
        # Keep only last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            precedence_path = os.path.join(self.model_dir, 'precedence_model.pkl')
            if os.path.exists(precedence_path):
                self.precedence_model = joblib.load(precedence_path)
                logger.info("Loaded precedence model")
        except Exception as e:
            logger.warning(f"Could not load precedence model: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            if self.precedence_model:
                joblib.save(
                    self.precedence_model,
                    os.path.join(self.model_dir, 'precedence_model.pkl')
                )
                
            if self.scaler_precedence:
                joblib.save(
                    self.scaler_precedence,
                    os.path.join(self.model_dir, 'precedence_scaler.pkl')
                )
                
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_ai_insights(self):
        """Get current AI system insights and performance"""
        return {
            'model_status': {
                'precedence_model': 'trained' if self.precedence_model else 'not_trained',
                'delay_predictor': 'trained' if self.delay_predictor else 'not_trained',
                'conflict_detector': 'trained' if self.conflict_detector else 'not_trained'
            },
            'prediction_accuracy': self.prediction_accuracy,
            'decision_history_size': len(self.decision_history),
            'recent_performance': self._analyze_recent_performance()
        }
    
    def _analyze_recent_performance(self):
        """Analyze recent decision-making performance"""
        if len(self.decision_history) < 5:
            return {'status': 'insufficient_data'}
        
        recent_decisions = self.decision_history[-20:]  # Last 20 decisions
        
        avg_solve_time = np.mean([d['solve_time'] for d in recent_decisions])
        avg_throughput = np.mean([d['throughput'] for d in recent_decisions])
        optimal_rate = len([d for d in recent_decisions if d['solution_quality'] == 'optimal']) / len(recent_decisions)
        
        return {
            'avg_solve_time': avg_solve_time,
            'avg_throughput': avg_throughput,
            'optimal_solution_rate': optimal_rate,
            'performance_trend': 'improving' if avg_solve_time < 4 else 'stable'
        }
    
    # Placeholder methods for missing implementations
    def _calculate_prediction_confidence(self, features, model_type):
        return 0.8  # Default confidence
    
    def _identify_delay_factors(self, features):
        return ['weather', 'congestion']
    
    def _identify_conflict_factors(self, features):
        return ['timing_overlap', 'priority_conflict']
    
    def _generate_throughput_recommendations(self, features):
        return ['optimize_spacing', 'adjust_priorities']
    
    def _explain_precedence_decision(self, train1, train2, should_precede):
        if should_precede:
            return f"Train {train1.train_id} has higher priority or greater delay"
        else:
            return f"Train {train2.train_id} should proceed first"
    
    def _calculate_overall_confidence(self, predictions):
        return 0.8  # Default overall confidence
    
    def _identify_risk_factors(self, situation_analysis, predictions):
        risks = []
        if situation_analysis['congestion_level']['level'] == 'critical':
            risks.append('critical_congestion')
        if situation_analysis['weather_impact']['impact_level'] != 'low':
            risks.append('weather_impact')
        return risks
    
    def _assess_optimization_quality(self, optimization_result):
        status = optimization_result.get('status', 'unknown')
        solve_time = optimization_result.get('solve_time', 10)
        
        if status == 'optimal' and solve_time <= 5:
            return 'excellent'
        elif status in ['optimal', 'feasible'] and solve_time <= 5:
            return 'good'
        else:
            return 'acceptable'
    
    def _collect_historical_data(self):
        # Placeholder for collecting historical data
        return {'precedence': []}
    
    def _train_delay_predictor(self, data):
        # Placeholder for delay predictor training
        pass
    
    def _train_conflict_detector(self, data):
        # Placeholder for conflict detector training
        pass
