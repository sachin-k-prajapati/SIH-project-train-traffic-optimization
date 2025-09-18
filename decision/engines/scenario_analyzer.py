"""
Scenario Analysis Engine for Railway Optimization
Implements what-if analysis for alternative routing, holding strategies,
platform allocation, emergency response, and capacity analysis scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from django.utils import timezone
import json
import logging
from dataclasses import dataclass
from enum import Enum

from core.models import Section, Train, Segment, Signal, Platform, WeatherCondition, EmergencyEvent
from decision.engines.ai_traffic_controller import AITrafficController
from decision.engines.ilp_engine_enhanced import AdvancedILPEngine
from simulator.real_time_engine import RealTimeSimulationEngine

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of scenarios that can be analyzed"""
    ALTERNATIVE_ROUTING = "alternative_routing"
    HOLDING_STRATEGY = "holding_strategy"
    PLATFORM_ALLOCATION = "platform_allocation"
    EMERGENCY_RESPONSE = "emergency_response"
    CAPACITY_ANALYSIS = "capacity_analysis"
    SPEED_OPTIMIZATION = "speed_optimization"
    WEATHER_IMPACT = "weather_impact"
    DEMAND_VARIATION = "demand_variation"

@dataclass
class ScenarioParameter:
    """Parameter for scenario analysis"""
    name: str
    value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    step_size: Optional[Any] = None
    description: str = ""

@dataclass
class ScenarioResult:
    """Result of scenario analysis"""
    scenario_id: str
    scenario_type: ScenarioType
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    success: bool
    execution_time: float
    recommendations: List[Dict]
    detailed_results: Optional[Dict] = None

class ScenarioAnalyzer:
    """
    Advanced scenario analyzer for railway optimization what-if analysis
    """
    
    def __init__(self, section):
        self.section = section
        self.ai_controller = AITrafficController(section)
        self.ilp_engine = AdvancedILPEngine(section)
        
        # Scenario tracking
        self.scenario_history = []
        self.baseline_metrics = None
        
        # Analysis configuration
        self.analysis_config = {
            'max_scenarios': 100,
            'parallel_execution': True,
            'confidence_threshold': 0.8,
            'significance_threshold': 0.05
        }
        
        logger.info(f"Initialized Scenario Analyzer for {section.name}")
    
    async def analyze_scenario(self, scenario_type: ScenarioType, 
                              trains: List[Train], 
                              parameters: Dict[str, Any] = None,
                              baseline_comparison: bool = True) -> ScenarioResult:
        """
        Analyze a specific scenario type with given parameters
        """
        logger.info(f"Analyzing scenario: {scenario_type.value}")
        
        start_time = timezone.now()
        scenario_id = f"{scenario_type.value}_{int(start_time.timestamp())}"
        
        try:
            # Set baseline if not exists
            if baseline_comparison and self.baseline_metrics is None:
                await self._establish_baseline(trains)
            
            # Execute scenario analysis
            result = await self._execute_scenario_analysis(
                scenario_id, scenario_type, trains, parameters or {}
            )
            
            # Compare with baseline if requested
            if baseline_comparison and self.baseline_metrics:
                result = self._compare_with_baseline(result)
            
            # Store result
            self.scenario_history.append(result)
            
            execution_time = (timezone.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"Scenario analysis completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Scenario analysis failed: {e}")
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                parameters=parameters or {},
                metrics={},
                success=False,
                execution_time=(timezone.now() - start_time).total_seconds(),
                recommendations=[{
                    'type': 'error',
                    'message': f"Analysis failed: {str(e)}",
                    'priority': 'high'
                }]
            )
    
    async def batch_scenario_analysis(self, scenario_configs: List[Dict]) -> List[ScenarioResult]:
        """
        Run multiple scenario analyses in batch
        """
        logger.info(f"Running batch analysis of {len(scenario_configs)} scenarios")
        
        results = []
        
        for i, config in enumerate(scenario_configs):
            logger.info(f"Running scenario {i+1}/{len(scenario_configs)}")
            
            scenario_type = ScenarioType(config['type'])
            trains = config['trains']
            parameters = config.get('parameters', {})
            
            result = await self.analyze_scenario(scenario_type, trains, parameters)
            results.append(result)
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(results)
        
        return results, comparative_analysis
    
    async def _execute_scenario_analysis(self, scenario_id: str, 
                                        scenario_type: ScenarioType,
                                        trains: List[Train], 
                                        parameters: Dict[str, Any]) -> ScenarioResult:
        """Execute specific scenario analysis"""
        
        if scenario_type == ScenarioType.ALTERNATIVE_ROUTING:
            return await self._analyze_alternative_routing(scenario_id, trains, parameters)
        
        elif scenario_type == ScenarioType.HOLDING_STRATEGY:
            return await self._analyze_holding_strategy(scenario_id, trains, parameters)
        
        elif scenario_type == ScenarioType.PLATFORM_ALLOCATION:
            return await self._analyze_platform_allocation(scenario_id, trains, parameters)
        
        elif scenario_type == ScenarioType.EMERGENCY_RESPONSE:
            return await self._analyze_emergency_response(scenario_id, trains, parameters)
        
        elif scenario_type == ScenarioType.CAPACITY_ANALYSIS:
            return await self._analyze_capacity_scenarios(scenario_id, trains, parameters)
        
        elif scenario_type == ScenarioType.SPEED_OPTIMIZATION:
            return await self._analyze_speed_optimization(scenario_id, trains, parameters)
        
        elif scenario_type == ScenarioType.WEATHER_IMPACT:
            return await self._analyze_weather_impact(scenario_id, trains, parameters)
        
        elif scenario_type == ScenarioType.DEMAND_VARIATION:
            return await self._analyze_demand_variation(scenario_id, trains, parameters)
        
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    async def _analyze_alternative_routing(self, scenario_id: str, 
                                          trains: List[Train], 
                                          parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze alternative routing scenarios"""
        logger.info("Analyzing alternative routing scenarios")
        
        # Define routing alternatives
        routing_alternatives = parameters.get('alternatives', [
            {'name': 'express_priority', 'description': 'Prioritize express trains'},
            {'name': 'balanced_flow', 'description': 'Balance all train types'},
            {'name': 'freight_optimization', 'description': 'Optimize for freight'}
        ])
        
        results = []
        
        for alternative in routing_alternatives:
            # Modify train priorities based on routing strategy
            modified_trains = self._apply_routing_strategy(trains, alternative)
            
            # Run optimization
            optimization_result = self.ilp_engine.optimize_comprehensive_schedule(
                modified_trains, time_horizon_minutes=180
            )
            
            metrics = self._extract_routing_metrics(optimization_result)
            
            results.append({
                'strategy': alternative['name'],
                'description': alternative['description'],
                'metrics': metrics,
                'feasible': optimization_result.get('status') in ['optimal', 'feasible']
            })
        
        # Find best alternative
        best_alternative = max(results, key=lambda x: x['metrics'].get('throughput_efficiency', 0))
        
        recommendations = self._generate_routing_recommendations(results, best_alternative)
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.ALTERNATIVE_ROUTING,
            parameters=parameters,
            metrics=best_alternative['metrics'],
            success=True,
            execution_time=0,  # Will be set by caller
            recommendations=recommendations,
            detailed_results={'alternatives': results, 'best': best_alternative}
        )
    
    async def _analyze_holding_strategy(self, scenario_id: str,
                                       trains: List[Train],
                                       parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze different holding strategies"""
        logger.info("Analyzing holding strategies")
        
        holding_strategies = parameters.get('strategies', [
            {'name': 'no_holding', 'max_hold_time': 0},
            {'name': 'minimal_holding', 'max_hold_time': 5},
            {'name': 'moderate_holding', 'max_hold_time': 15},
            {'name': 'aggressive_holding', 'max_hold_time': 30}
        ])
        
        results = []
        
        for strategy in holding_strategies:
            # Simulate holding strategy
            modified_scenario = self._apply_holding_strategy(trains, strategy)
            
            # Run optimization with holding constraints
            optimization_result = self.ilp_engine.optimize_comprehensive_schedule(
                modified_scenario['trains'],
                time_horizon_minutes=modified_scenario['time_horizon']
            )
            
            metrics = self._extract_holding_metrics(optimization_result, strategy)
            
            results.append({
                'strategy': strategy['name'],
                'max_hold_time': strategy['max_hold_time'],
                'metrics': metrics,
                'passenger_impact': self._calculate_passenger_impact(metrics),
                'feasible': optimization_result.get('status') in ['optimal', 'feasible']
            })
        
        # Analyze trade-offs
        best_strategy = self._select_best_holding_strategy(results)
        recommendations = self._generate_holding_recommendations(results, best_strategy)
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.HOLDING_STRATEGY,
            parameters=parameters,
            metrics=best_strategy['metrics'],
            success=True,
            execution_time=0,
            recommendations=recommendations,
            detailed_results={'strategies': results, 'best': best_strategy}
        )
    
    async def _analyze_platform_allocation(self, scenario_id: str,
                                          trains: List[Train],
                                          parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze platform allocation strategies"""
        logger.info("Analyzing platform allocation strategies")
        
        platforms = Platform.objects.filter(segment__section=self.section)
        
        allocation_strategies = parameters.get('strategies', [
            {'name': 'first_available', 'description': 'First available platform'},
            {'name': 'shortest_walk', 'description': 'Minimize passenger walking'},
            {'name': 'type_specific', 'description': 'Dedicate platforms by train type'},
            {'name': 'dynamic_optimal', 'description': 'AI-optimized allocation'}
        ])
        
        results = []
        
        for strategy in allocation_strategies:
            # Apply platform allocation strategy
            allocation_result = self._simulate_platform_allocation(trains, platforms, strategy)
            
            metrics = {
                'platform_utilization': allocation_result['utilization'],
                'average_turnaround_time': allocation_result['turnaround_time'],
                'conflicts': allocation_result['conflicts'],
                'passenger_satisfaction': allocation_result['satisfaction_score']
            }
            
            results.append({
                'strategy': strategy['name'],
                'description': strategy['description'],
                'metrics': metrics,
                'feasible': allocation_result['feasible']
            })
        
        best_strategy = max(results, key=lambda x: x['metrics']['passenger_satisfaction'])
        recommendations = self._generate_platform_recommendations(results, best_strategy)
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.PLATFORM_ALLOCATION,
            parameters=parameters,
            metrics=best_strategy['metrics'],
            success=True,
            execution_time=0,
            recommendations=recommendations,
            detailed_results={'strategies': results, 'best': best_strategy}
        )
    
    async def _analyze_emergency_response(self, scenario_id: str,
                                         trains: List[Train],
                                         parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze emergency response scenarios"""
        logger.info("Analyzing emergency response scenarios")
        
        emergency_scenarios = parameters.get('emergencies', [
            {
                'type': 'signal_failure',
                'affected_segments': [1],  # Would use actual segment IDs
                'duration_minutes': 20,
                'severity': 'moderate'
            },
            {
                'type': 'medical_emergency',
                'affected_segments': [],
                'duration_minutes': 15,
                'severity': 'high'
            },
            {
                'type': 'security_alert',
                'affected_segments': [2, 3],
                'duration_minutes': 45,
                'severity': 'major'
            }
        ])
        
        results = []
        
        for emergency in emergency_scenarios:
            # Simulate emergency response
            response_result = await self._simulate_emergency_response(trains, emergency)
            
            metrics = {
                'response_time_minutes': response_result['response_time'],
                'affected_trains': response_result['affected_trains'],
                'total_delay_minutes': response_result['total_delay'],
                'recovery_time_minutes': response_result['recovery_time'],
                'passenger_impact_score': response_result['passenger_impact']
            }
            
            results.append({
                'emergency_type': emergency['type'],
                'severity': emergency['severity'],
                'metrics': metrics,
                'response_actions': response_result['actions'],
                'effectiveness': response_result['effectiveness']
            })
        
        # Analyze response effectiveness
        avg_effectiveness = np.mean([r['effectiveness'] for r in results])
        recommendations = self._generate_emergency_recommendations(results)
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.EMERGENCY_RESPONSE,
            parameters=parameters,
            metrics={'average_effectiveness': avg_effectiveness},
            success=True,
            execution_time=0,
            recommendations=recommendations,
            detailed_results={'emergency_scenarios': results}
        )
    
    async def _analyze_capacity_scenarios(self, scenario_id: str,
                                         trains: List[Train],
                                         parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze capacity utilization scenarios"""
        logger.info("Analyzing capacity scenarios")
        
        capacity_levels = parameters.get('capacity_levels', [0.5, 0.7, 0.85, 0.95, 1.0, 1.1])
        base_train_count = len(trains)
        
        results = []
        
        for capacity_level in capacity_levels:
            # Adjust train count based on capacity level
            target_train_count = int(base_train_count * capacity_level)
            
            if target_train_count > base_train_count:
                # Generate additional trains
                scenario_trains = trains + self._generate_additional_trains(
                    target_train_count - base_train_count
                )
            else:
                # Use subset of trains
                scenario_trains = trains[:target_train_count]
            
            # Run capacity analysis
            optimization_result = self.ilp_engine.optimize_comprehensive_schedule(
                scenario_trains, time_horizon_minutes=240  # Longer horizon for capacity test
            )
            
            metrics = self._extract_capacity_metrics(optimization_result, capacity_level)
            
            results.append({
                'capacity_level': capacity_level,
                'train_count': len(scenario_trains),
                'metrics': metrics,
                'feasible': optimization_result.get('status') in ['optimal', 'feasible'],
                'bottlenecks': self._identify_capacity_bottlenecks(optimization_result)
            })
        
        # Find optimal capacity point
        optimal_capacity = self._find_optimal_capacity(results)
        recommendations = self._generate_capacity_recommendations(results, optimal_capacity)
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.CAPACITY_ANALYSIS,
            parameters=parameters,
            metrics=optimal_capacity['metrics'],
            success=True,
            execution_time=0,
            recommendations=recommendations,
            detailed_results={'capacity_analysis': results, 'optimal': optimal_capacity}
        )
    
    async def _analyze_speed_optimization(self, scenario_id: str,
                                         trains: List[Train],
                                         parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze speed optimization scenarios"""
        logger.info("Analyzing speed optimization scenarios")
        
        speed_strategies = parameters.get('strategies', [
            {'name': 'uniform_speed', 'description': 'All trains at same speed'},
            {'name': 'priority_based', 'description': 'Speed based on priority'},
            {'name': 'type_optimized', 'description': 'Speed optimized by train type'},
            {'name': 'dynamic_adaptive', 'description': 'AI-adaptive speed control'}
        ])
        
        results = []
        
        for strategy in speed_strategies:
            # Apply speed strategy
            speed_result = self._simulate_speed_strategy(trains, strategy)
            
            metrics = {
                'average_speed': speed_result['avg_speed'],
                'fuel_efficiency': speed_result['fuel_efficiency'],
                'travel_time_variance': speed_result['time_variance'],
                'throughput': speed_result['throughput'],
                'safety_score': speed_result['safety_score']
            }
            
            results.append({
                'strategy': strategy['name'],
                'description': strategy['description'],
                'metrics': metrics,
                'feasible': speed_result['feasible']
            })
        
        best_strategy = self._select_best_speed_strategy(results)
        recommendations = self._generate_speed_recommendations(results, best_strategy)
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.SPEED_OPTIMIZATION,
            parameters=parameters,
            metrics=best_strategy['metrics'],
            success=True,
            execution_time=0,
            recommendations=recommendations,
            detailed_results={'strategies': results, 'best': best_strategy}
        )
    
    async def _analyze_weather_impact(self, scenario_id: str,
                                     trains: List[Train],
                                     parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze weather impact scenarios"""
        logger.info("Analyzing weather impact scenarios")
        
        weather_conditions = parameters.get('conditions', [
            {'condition': 'clear', 'visibility_factor': 1.0, 'speed_reduction': 0},
            {'condition': 'light_rain', 'visibility_factor': 0.9, 'speed_reduction': 10},
            {'condition': 'heavy_rain', 'visibility_factor': 0.7, 'speed_reduction': 25},
            {'condition': 'fog', 'visibility_factor': 0.5, 'speed_reduction': 40},
            {'condition': 'storm', 'visibility_factor': 0.3, 'speed_reduction': 60}
        ])
        
        results = []
        
        for condition in weather_conditions:
            # Simulate weather impact
            weather_result = self._simulate_weather_impact(trains, condition)
            
            metrics = {
                'average_delay': weather_result['avg_delay'],
                'throughput_reduction': weather_result['throughput_reduction'],
                'safety_incidents': weather_result['safety_incidents'],
                'operational_efficiency': weather_result['efficiency']
            }
            
            results.append({
                'condition': condition['condition'],
                'visibility_factor': condition['visibility_factor'],
                'metrics': metrics,
                'mitigation_actions': weather_result['mitigations']
            })
        
        # Analyze weather preparedness
        recommendations = self._generate_weather_recommendations(results)
        worst_case = min(results, key=lambda x: x['metrics']['operational_efficiency'])
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.WEATHER_IMPACT,
            parameters=parameters,
            metrics=worst_case['metrics'],
            success=True,
            execution_time=0,
            recommendations=recommendations,
            detailed_results={'weather_scenarios': results, 'worst_case': worst_case}
        )
    
    async def _analyze_demand_variation(self, scenario_id: str,
                                       trains: List[Train],
                                       parameters: Dict[str, Any]) -> ScenarioResult:
        """Analyze demand variation scenarios"""
        logger.info("Analyzing demand variation scenarios")
        
        demand_scenarios = parameters.get('scenarios', [
            {'name': 'peak_hour', 'multiplier': 1.5, 'duration_hours': 2},
            {'name': 'off_peak', 'multiplier': 0.6, 'duration_hours': 4},
            {'name': 'special_event', 'multiplier': 2.0, 'duration_hours': 1},
            {'name': 'holiday_weekend', 'multiplier': 0.3, 'duration_hours': 8}
        ])
        
        results = []
        
        for scenario in demand_scenarios:
            # Adjust demand based on scenario
            demand_result = self._simulate_demand_scenario(trains, scenario)
            
            metrics = {
                'passenger_load': demand_result['passenger_load'],
                'service_quality': demand_result['service_quality'],
                'resource_utilization': demand_result['resource_utilization'],
                'revenue_impact': demand_result['revenue_impact']
            }
            
            results.append({
                'scenario': scenario['name'],
                'demand_multiplier': scenario['multiplier'],
                'metrics': metrics,
                'capacity_stress': demand_result['capacity_stress']
            })
        
        recommendations = self._generate_demand_recommendations(results)
        peak_scenario = max(results, key=lambda x: x['demand_multiplier'])
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=ScenarioType.DEMAND_VARIATION,
            parameters=parameters,
            metrics=peak_scenario['metrics'],
            success=True,
            execution_time=0,
            recommendations=recommendations,
            detailed_results={'demand_scenarios': results, 'peak': peak_scenario}
        )
    
    async def _establish_baseline(self, trains: List[Train]):
        """Establish baseline metrics for comparison"""
        logger.info("Establishing baseline metrics")
        
        # Run standard optimization
        baseline_result = self.ilp_engine.optimize_comprehensive_schedule(trains)
        
        self.baseline_metrics = {
            'throughput': baseline_result.get('kpis', {}).get('throughput_per_hour', 0),
            'average_delay': baseline_result.get('kpis', {}).get('avg_delay_minutes', 0),
            'punctuality': baseline_result.get('kpis', {}).get('punctuality_percent', 0),
            'utilization': baseline_result.get('kpis', {}).get('capacity_utilization', 0),
            'efficiency': baseline_result.get('kpis', {}).get('fuel_efficiency_score', 0)
        }
        
        logger.info(f"Baseline established: {self.baseline_metrics}")
    
    def _compare_with_baseline(self, result: ScenarioResult) -> ScenarioResult:
        """Compare scenario result with baseline"""
        if not self.baseline_metrics:
            return result
        
        improvements = {}
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in result.metrics:
                current_value = result.metrics[metric]
                improvement = ((current_value - baseline_value) / baseline_value * 100 
                              if baseline_value > 0 else 0)
                improvements[f"{metric}_improvement_percent"] = improvement
        
        result.metrics.update(improvements)
        
        # Add baseline comparison to recommendations
        significant_improvements = [
            metric for metric, improvement in improvements.items()
            if abs(improvement) > 5  # 5% threshold
        ]
        
        if significant_improvements:
            result.recommendations.append({
                'type': 'baseline_comparison',
                'message': f"Significant changes vs baseline: {', '.join(significant_improvements)}",
                'priority': 'medium',
                'details': improvements
            })
        
        return result
    
    def get_scenario_insights(self) -> Dict[str, Any]:
        """Get insights from scenario analysis history"""
        if not self.scenario_history:
            return {'message': 'No scenario analysis history available'}
        
        # Analyze patterns
        scenario_types = {}
        success_rates = {}
        
        for result in self.scenario_history:
            stype = result.scenario_type.value
            scenario_types[stype] = scenario_types.get(stype, 0) + 1
            
            if stype not in success_rates:
                success_rates[stype] = {'success': 0, 'total': 0}
            
            success_rates[stype]['total'] += 1
            if result.success:
                success_rates[stype]['success'] += 1
        
        # Calculate success rates
        for stype, data in success_rates.items():
            data['rate'] = data['success'] / data['total'] if data['total'] > 0 else 0
        
        # Find best performing scenarios
        best_scenarios = sorted(
            [r for r in self.scenario_history if r.success],
            key=lambda x: x.metrics.get('throughput', 0),
            reverse=True
        )[:5]
        
        return {
            'total_scenarios_analyzed': len(self.scenario_history),
            'scenario_type_distribution': scenario_types,
            'success_rates_by_type': success_rates,
            'top_performing_scenarios': [
                {
                    'id': s.scenario_id,
                    'type': s.scenario_type.value,
                    'metrics': s.metrics
                } for s in best_scenarios
            ],
            'recommendations': self._generate_meta_recommendations()
        }
    
    def _generate_meta_recommendations(self) -> List[Dict]:
        """Generate recommendations based on all scenario analyses"""
        recommendations = []
        
        if len(self.scenario_history) < 5:
            recommendations.append({
                'type': 'analysis_volume',
                'message': 'Run more scenario analyses for better insights',
                'priority': 'medium'
            })
        
        # Check for consistently poor performance in certain scenarios
        weather_scenarios = [r for r in self.scenario_history 
                           if r.scenario_type == ScenarioType.WEATHER_IMPACT]
        
        if weather_scenarios and all(not r.success for r in weather_scenarios[-3:]):
            recommendations.append({
                'type': 'weather_preparedness',
                'message': 'Weather impact scenarios showing poor performance - review preparedness',
                'priority': 'high'
            })
        
        return recommendations
    
    # Helper methods for scenario-specific logic
    def _apply_routing_strategy(self, trains, strategy):
        # Placeholder implementation
        return trains
    
    def _extract_routing_metrics(self, optimization_result):
        return {
            'throughput_efficiency': optimization_result.get('kpis', {}).get('throughput_efficiency_percent', 0),
            'average_delay': optimization_result.get('kpis', {}).get('avg_delay_minutes', 0),
            'conflicts': optimization_result.get('throughput_analysis', {}).get('total_conflicts', 0)
        }
    
    def _generate_routing_recommendations(self, results, best_alternative):
        return [{
            'type': 'routing_optimization',
            'message': f"Best routing strategy: {best_alternative['strategy']}",
            'priority': 'medium',
            'action': f"Implement {best_alternative['description']} for optimal throughput"
        }]
    
    def _apply_holding_strategy(self, trains, strategy):
        return {'trains': trains, 'time_horizon': 180}
    
    def _extract_holding_metrics(self, optimization_result, strategy):
        return {
            'total_holding_time': strategy['max_hold_time'] * len(optimization_result.get('trains', {})),
            'passenger_wait_time': strategy['max_hold_time'] * 0.8,  # Estimated
            'throughput': optimization_result.get('kpis', {}).get('throughput_per_hour', 0)
        }
    
    def _calculate_passenger_impact(self, metrics):
        return max(0, 100 - metrics.get('passenger_wait_time', 0) * 2)
    
    def _select_best_holding_strategy(self, results):
        return max(results, key=lambda x: x['passenger_impact'])
    
    def _generate_holding_recommendations(self, results, best_strategy):
        return [{
            'type': 'holding_strategy',
            'message': f"Optimal holding strategy: {best_strategy['strategy']}",
            'priority': 'medium'
        }]
    
    def _simulate_platform_allocation(self, trains, platforms, strategy):
        return {
            'utilization': np.random.uniform(0.6, 0.9),
            'turnaround_time': np.random.uniform(5, 15),
            'conflicts': np.random.randint(0, 3),
            'satisfaction_score': np.random.uniform(70, 95),
            'feasible': True
        }
    
    def _generate_platform_recommendations(self, results, best_strategy):
        return [{
            'type': 'platform_allocation',
            'message': f"Best platform strategy: {best_strategy['strategy']}",
            'priority': 'medium'
        }]
    
    async def _simulate_emergency_response(self, trains, emergency):
        return {
            'response_time': np.random.uniform(5, 15),
            'affected_trains': np.random.randint(1, len(trains)//2),
            'total_delay': np.random.uniform(20, 60),
            'recovery_time': np.random.uniform(30, 90),
            'passenger_impact': np.random.uniform(50, 200),
            'actions': ['reroute_trains', 'notify_passengers', 'deploy_backup'],
            'effectiveness': np.random.uniform(0.6, 0.9)
        }
    
    def _generate_emergency_recommendations(self, results):
        return [{
            'type': 'emergency_preparedness',
            'message': 'Review emergency response procedures',
            'priority': 'high'
        }]
    
    def _generate_additional_trains(self, count):
        # Generate dummy trains for capacity testing
        return []
    
    def _extract_capacity_metrics(self, optimization_result, capacity_level):
        return {
            'capacity_utilization': capacity_level * 100,
            'throughput_per_hour': optimization_result.get('kpis', {}).get('throughput_per_hour', 0),
            'average_delay': optimization_result.get('kpis', {}).get('avg_delay_minutes', 0),
            'bottleneck_score': capacity_level * 10  # Simplified
        }
    
    def _identify_capacity_bottlenecks(self, optimization_result):
        return ['segment_1', 'signal_junction_a']  # Placeholder
    
    def _find_optimal_capacity(self, results):
        return max(results, key=lambda x: x['metrics']['throughput_per_hour'])
    
    def _generate_capacity_recommendations(self, results, optimal_capacity):
        return [{
            'type': 'capacity_optimization',
            'message': f"Optimal capacity level: {optimal_capacity['capacity_level']:.1%}",
            'priority': 'medium'
        }]
    
    def _simulate_speed_strategy(self, trains, strategy):
        return {
            'avg_speed': np.random.uniform(60, 100),
            'fuel_efficiency': np.random.uniform(70, 95),
            'time_variance': np.random.uniform(5, 20),
            'throughput': np.random.uniform(8, 15),
            'safety_score': np.random.uniform(85, 98),
            'feasible': True
        }
    
    def _select_best_speed_strategy(self, results):
        return max(results, key=lambda x: x['metrics']['fuel_efficiency'])
    
    def _generate_speed_recommendations(self, results, best_strategy):
        return [{
            'type': 'speed_optimization',
            'message': f"Best speed strategy: {best_strategy['strategy']}",
            'priority': 'medium'
        }]
    
    def _simulate_weather_impact(self, trains, condition):
        visibility_factor = condition['visibility_factor']
        return {
            'avg_delay': (1 - visibility_factor) * 30,
            'throughput_reduction': (1 - visibility_factor) * 40,
            'safety_incidents': int((1 - visibility_factor) * 5),
            'efficiency': visibility_factor * 100,
            'mitigations': ['reduce_speed', 'increase_spacing', 'enhance_signaling']
        }
    
    def _generate_weather_recommendations(self, results):
        return [{
            'type': 'weather_preparedness',
            'message': 'Implement weather-adaptive operations',
            'priority': 'high'
        }]
    
    def _simulate_demand_scenario(self, trains, scenario):
        multiplier = scenario['multiplier']
        return {
            'passenger_load': multiplier * 100,
            'service_quality': max(0, 100 - (multiplier - 1) * 50),
            'resource_utilization': min(100, multiplier * 80),
            'revenue_impact': (multiplier - 1) * 20,
            'capacity_stress': max(0, multiplier - 1)
        }
    
    def _generate_demand_recommendations(self, results):
        return [{
            'type': 'demand_management',
            'message': 'Implement dynamic capacity management',
            'priority': 'medium'
        }]
    
    def _generate_comparative_analysis(self, results):
        """Generate comparative analysis of multiple scenarios"""
        return {
            'best_overall': max(results, key=lambda x: x.metrics.get('throughput', 0) if x.success else 0),
            'success_rate': len([r for r in results if r.success]) / len(results) if results else 0,
            'average_execution_time': np.mean([r.execution_time for r in results]) if results else 0
        }