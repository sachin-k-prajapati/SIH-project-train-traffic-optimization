from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
from core.models import Segment, Signal, Platform, WeatherCondition, EmergencyEvent
import logging

logger = logging.getLogger(__name__)

class AdvancedILPEngine:
    """
    Advanced Integer Linear Programming engine for railway optimization
    with realistic constraints, multi-objective optimization, and ML integration
    """
    
    def __init__(self, section):
        self.section = section
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 30.0  # Real-time requirement
        
        # Load section topology
        self.segments = list(Segment.objects.filter(section=section).order_by('sequence'))
        self.signals = Signal.objects.filter(segment__section=section)
        self.platforms = Platform.objects.filter(segment__section=section)
        
        # Performance tracking
        self.solve_history = []
        
        logger.info(f"Initialized Advanced ILP Engine for {section.name}")
    
    def optimize_comprehensive_schedule(self, trains, disruptions=None, time_horizon_minutes=180):
        """
        Comprehensive optimization considering multiple objectives and constraints
        """
        logger.info(f"Starting comprehensive optimization for {len(trains)} trains")
        
        if not trains:
            return {'status': 'no_trains', 'message': 'No trains to optimize'}
        
        start_time = timezone.now()
        
        # Initialize model
        self.model = cp_model.CpModel()
        
        # Decision variables
        variables = self._create_decision_variables(trains, time_horizon_minutes)
        
        # Add comprehensive constraints
        self._add_capacity_constraints(trains, variables)
        self._add_precedence_constraints(trains, variables)
        self._add_safety_constraints(trains, variables)
        self._add_infrastructure_constraints(trains, variables)
        self._add_weather_constraints(trains, variables)
        
        if disruptions:
            self._add_disruption_constraints(trains, variables, disruptions)
        
        # Multi-objective optimization
        objective = self._create_multi_objective(trains, variables)
        self.model.Minimize(objective)
        
        # Solve with time limit
        status = self.solver.Solve(self.model)
        solve_time = (timezone.now() - start_time).total_seconds()
        
        # Extract and return solution
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = self._extract_comprehensive_solution(trains, variables, status, solve_time)
            logger.info(f"Optimization completed in {solve_time:.2f}s")
            return solution
        else:
            logger.error(f"Optimization failed with status: {status}")
            return {
                'status': 'failed',
                'solve_time': solve_time,
                'message': 'No feasible solution found'
            }
    
    def optimize_precedence(self, active_trains, current_time):
        """Legacy method for compatibility - calls comprehensive optimization"""
        return self.optimize_comprehensive_schedule(active_trains, time_horizon_minutes=120)
    
    def _create_decision_variables(self, trains, time_horizon):
        """Create comprehensive decision variables"""
        variables = {
            'arrival_times': {},      # When train arrives at section
            'departure_times': {},    # When train departs section
            'segment_entry': {},      # When train enters each segment
            'segment_exit': {},       # When train exits each segment
            'platform_assignment': {},  # Which platform assigned to train
            'speed_profile': {},      # Speed for each segment
            'precedence': {},         # Train precedence decisions
            'route_choice': {},       # Route choice for trains with alternatives
            'delay_vars': {},         # Delay variables for soft constraints
        }
        
        # Time variables for each train
        for train in trains:
            variables['arrival_times'][train.id] = self.model.NewIntVar(
                0, time_horizon, f'arrival_{train.id}'
            )
            variables['departure_times'][train.id] = self.model.NewIntVar(
                0, time_horizon, f'departure_{train.id}'
            )
            variables['delay_vars'][train.id] = self.model.NewIntVar(
                0, time_horizon // 2, f'delay_{train.id}'
            )
            
            # Segment timing variables
            variables['segment_entry'][train.id] = {}
            variables['segment_exit'][train.id] = {}
            variables['speed_profile'][train.id] = {}
            
            for segment in self.segments:
                variables['segment_entry'][train.id][segment.id] = self.model.NewIntVar(
                    0, time_horizon, f'seg_entry_{train.id}_{segment.id}'
                )
                variables['segment_exit'][train.id][segment.id] = self.model.NewIntVar(
                    0, time_horizon, f'seg_exit_{train.id}_{segment.id}'
                )
                
                # Speed as percentage of max speed (20% to 100%)
                variables['speed_profile'][train.id][segment.id] = self.model.NewIntVar(
                    20, 100, f'speed_{train.id}_{segment.id}'
                )
            
            # Platform assignment for passenger trains
            if train.train_type in ['express', 'local']:
                variables['platform_assignment'][train.id] = {}
                for platform in self.platforms:
                    variables['platform_assignment'][train.id][platform.id] = self.model.NewBoolVar(
                        f'platform_{train.id}_{platform.id}'
                    )
        
        # Precedence variables between train pairs
        variables['precedence'] = {}
        for i, train1 in enumerate(trains):
            for train2 in trains[i+1:]:
                for segment in self.segments:
                    var_name = f'prec_{train1.id}_{train2.id}_{segment.id}'
                    variables['precedence'][(train1.id, train2.id, segment.id)] = \
                        self.model.NewBoolVar(var_name)
        
        return variables
    
    def _add_capacity_constraints(self, trains, variables):
        """Add realistic capacity constraints for segments, signals, and platforms"""
        logger.debug("Adding capacity constraints")
        
        # Segment capacity constraints
        for segment in self.segments:
            for time_slot in range(0, 180, 5):  # Check every 5 minutes
                trains_in_segment = []
                
                for train in trains:
                    # Create boolean variable: is train in segment at time_slot?
                    in_segment_var = self.model.NewBoolVar(
                        f'in_seg_{train.id}_{segment.id}_{time_slot}'
                    )
                    
                    # Link to entry/exit times
                    entry_time = variables['segment_entry'][train.id][segment.id]
                    exit_time = variables['segment_exit'][train.id][segment.id]
                    
                    # Train is in segment if entry_time <= time_slot < exit_time
                    self.model.Add(entry_time <= time_slot).OnlyEnforceIf(in_segment_var)
                    self.model.Add(exit_time > time_slot).OnlyEnforceIf(in_segment_var)
                    self.model.Add(entry_time > time_slot).OnlyEnforceIf(in_segment_var.Not())
                    
                    trains_in_segment.append(in_segment_var)
                
                # Capacity constraint
                self.model.Add(sum(trains_in_segment) <= segment.max_occupancy)
        
        # Platform capacity constraints
        for platform in self.platforms:
            if platform.segment.platform_count > 0:
                for time_slot in range(0, 180, 1):  # Every minute for platforms
                    platform_occupancy = []
                    
                    for train in trains:
                        if train.id in variables['platform_assignment']:
                            if platform.id in variables['platform_assignment'][train.id]:
                                platform_occupancy.append(
                                    variables['platform_assignment'][train.id][platform.id]
                                )
                    
                    if platform_occupancy:
                        self.model.Add(sum(platform_occupancy) <= 1)
    
    def _add_precedence_constraints(self, trains, variables):
        """Add train precedence constraints with realistic separation times"""
        logger.debug("Adding precedence constraints")
        
        for i, train1 in enumerate(trains):
            for train2 in trains[i+1:]:
                for segment in self.segments:
                    prec_var = variables['precedence'][(train1.id, train2.id, segment.id)]
                    
                    # Calculate minimum separation based on train types and speeds
                    min_separation = self._calculate_min_separation(train1, train2, segment)
                    
                    entry1 = variables['segment_entry'][train1.id][segment.id]
                    entry2 = variables['segment_entry'][train2.id][segment.id]
                    exit1 = variables['segment_exit'][train1.id][segment.id]
                    exit2 = variables['segment_exit'][train2.id][segment.id]
                    
                    # Big M for conditional constraints
                    M = 1000
                    
                    # If train1 precedes train2 in this segment
                    self.model.Add(
                        entry2 >= exit1 + min_separation - M * (1 - prec_var)
                    )
                    
                    # If train2 precedes train1 in this segment
                    self.model.Add(
                        entry1 >= exit2 + min_separation - M * prec_var
                    )
    
    def _add_safety_constraints(self, trains, variables):
        """Add comprehensive safety constraints"""
        logger.debug("Adding safety constraints")
        
        for train in trains:
            # Minimum dwell time constraints
            arrival = variables['arrival_times'][train.id]
            departure = variables['departure_times'][train.id]
            
            if train.train_type == 'express':
                min_dwell = 2  # 2 minutes minimum
            elif train.train_type == 'local':
                min_dwell = 1  # 1 minute minimum
            else:  # freight
                min_dwell = 5  # 5 minutes minimum
            
            self.model.Add(departure >= arrival + min_dwell)
            
            # Sequential segment constraints
            for i, segment in enumerate(self.segments[:-1]):
                next_segment = self.segments[i + 1]
                
                exit_current = variables['segment_exit'][train.id][segment.id]
                entry_next = variables['segment_entry'][train.id][next_segment.id]
                
                # Must exit current before entering next (with minimal gap)
                self.model.Add(entry_next >= exit_current + 1)
            
            # Speed-distance-time relationship
            for segment in self.segments:
                entry_time = variables['segment_entry'][train.id][segment.id]
                exit_time = variables['segment_exit'][train.id][segment.id]
                speed_percent = variables['speed_profile'][train.id][segment.id]
                
                # Calculate travel time based on speed
                max_speed = min(train.max_speed, segment.effective_max_speed)
                distance_km = segment.length_km
                
                # Travel time = distance / (speed * speed_percent/100)
                # Simplified: travel_time = base_time * 100 / speed_percent
                base_time_minutes = (distance_km / max_speed) * 60
                
                # Approximate linear constraint (simplified)
                min_travel_time = int(base_time_minutes * 100 / 100)  # At 100% speed
                max_travel_time = int(base_time_minutes * 100 / 20)   # At 20% speed
                
                travel_time = self.model.NewIntVar(
                    min_travel_time, max_travel_time, f'travel_{train.id}_{segment.id}'
                )
                
                self.model.Add(exit_time == entry_time + travel_time)
                
                # Link speed to travel time (approximation)
                # travel_time * speed_percent = base_time_minutes * 100
                # This is non-linear, so we use approximation
                self.model.Add(travel_time >= int(base_time_minutes * 100 / 100))
    
    def _add_infrastructure_constraints(self, trains, variables):
        """Add infrastructure-specific constraints"""
        logger.debug("Adding infrastructure constraints")
        
        # Signal constraints
        for signal in self.signals:
            segment = signal.segment
            
            # Only one train can control a signal at a time
            for time_slot in range(0, 180, 2):  # Every 2 minutes
                signal_users = []
                
                for train in trains:
                    entry = variables['segment_entry'][train.id][segment.id]
                    exit = variables['segment_exit'][train.id][segment.id]
                    
                    uses_signal = self.model.NewBoolVar(
                        f'signal_{signal.id}_{train.id}_{time_slot}'
                    )
                    
                    # Train uses signal if it's in the segment
                    self.model.Add(entry <= time_slot).OnlyEnforceIf(uses_signal)
                    self.model.Add(exit > time_slot).OnlyEnforceIf(uses_signal)
                    
                    signal_users.append(uses_signal)
                
                # At most one train per signal
                if signal_users:
                    self.model.Add(sum(signal_users) <= 1)
        
        # Electrification constraints
        for train in trains:
            for segment in self.segments:
                if not segment.electrified and train.train_type == 'electric':
                    # Electric trains cannot use non-electrified segments
                    entry = variables['segment_entry'][train.id][segment.id]
                    exit = variables['segment_exit'][train.id][segment.id]
                    self.model.Add(entry == exit)  # No time spent in segment
    
    def _add_weather_constraints(self, trains, variables):
        """Add weather-based constraints"""
        weather_factor = self.section.weather_impact_factor
        
        if weather_factor < 1.0:  # Adverse weather
            logger.debug(f"Adding weather constraints (factor: {weather_factor})")
            
            for train in trains:
                for segment in self.segments:
                    speed_var = variables['speed_profile'][train.id][segment.id]
                    
                    # Reduce maximum allowed speed due to weather
                    max_weather_speed = int(100 * weather_factor)
                    self.model.Add(speed_var <= max_weather_speed)
    
    def _add_disruption_constraints(self, trains, variables, disruptions):
        """Add constraints for known disruptions"""
        logger.debug(f"Adding constraints for {len(disruptions)} disruptions")
        
        for disruption in disruptions:
            if hasattr(disruption, 'affected_segment'):
                segment = disruption.affected_segment
                
                # Block segment during disruption time
                disruption_start = 0  # Convert to simulation time
                disruption_end = 60   # Convert to simulation time
                
                for train in trains:
                    entry = variables['segment_entry'][train.id][segment.id]
                    exit = variables['segment_exit'][train.id][segment.id]
                    
                    # Train cannot be in segment during disruption
                    not_in_disruption = self.model.NewBoolVar(
                        f'not_in_disruption_{train.id}_{segment.id}'
                    )
                    
                    # Either train finishes before disruption or starts after
                    self.model.Add(exit <= disruption_start).OnlyEnforceIf(not_in_disruption)
                    self.model.Add(entry >= disruption_end).OnlyEnforceIf(not_in_disruption.Not())
    
    def _create_multi_objective(self, trains, variables):
        """Create multi-objective function balancing multiple goals"""
        objective_terms = []
        
        # 1. Minimize total delay (highest priority)
        delay_weight = 1000
        for train in trains:
            delay_var = variables['delay_vars'][train.id]
            priority_factor = 6 - train.priority  # Higher priority = higher weight
            objective_terms.append(delay_var * delay_weight * priority_factor)
        
        # 2. Minimize total travel time
        travel_time_weight = 100
        for train in trains:
            arrival = variables['arrival_times'][train.id]
            departure = variables['departure_times'][train.id]
            total_time = departure - arrival
            objective_terms.append(total_time * travel_time_weight)
        
        # 3. Maximize throughput (minimize makespan)
        makespan_weight = 50
        latest_departure = self.model.NewIntVar(0, 180, 'makespan')
        for train in trains:
            departure = variables['departure_times'][train.id]
            self.model.Add(latest_departure >= departure)
        objective_terms.append(latest_departure * makespan_weight)
        
        # 4. Minimize fuel consumption (favor efficient speeds)
        fuel_weight = 10
        for train in trains:
            for segment in self.segments:
                speed_var = variables['speed_profile'][train.id][segment.id]
                # Quadratic relationship approximated linearly
                # Optimal speed around 60-80%, penalty for extremes
                fuel_penalty = self.model.NewIntVar(0, 100, f'fuel_{train.id}_{segment.id}')
                
                # Penalty increases as speed deviates from 70%
                optimal_speed = 70
                self.model.Add(fuel_penalty >= speed_var - optimal_speed)
                self.model.Add(fuel_penalty >= optimal_speed - speed_var)
                
                objective_terms.append(fuel_penalty * fuel_weight)
        
        return sum(objective_terms)
    
    def _calculate_min_separation(self, train1, train2, segment):
        """Calculate minimum separation time between trains"""
        base_separation = 3  # 3 minutes base
        
        # Adjust based on train types
        if train1.train_type == 'freight' or train2.train_type == 'freight':
            base_separation += 2
        
        if train1.priority == 1 or train2.priority == 1:  # High priority train
            base_separation += 1
        
        # Adjust for segment characteristics
        if segment.has_siding:
            base_separation -= 1  # Can use siding for closer spacing
        
        if segment.gradient_percent > 2:  # Steep gradient
            base_separation += 1
        
        return max(2, base_separation)  # Minimum 2 minutes
    
    def _extract_comprehensive_solution(self, trains, variables, status, solve_time):
        """Extract comprehensive solution with detailed recommendations"""
        solution = {
            'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',
            'solve_time': solve_time,
            'objective_value': self.solver.ObjectiveValue(),
            'trains': {},
            'recommendations': [],
            'kpis': {},
            'algorithm_confidence': 'high' if status == cp_model.OPTIMAL else 'medium'
        }
        
        # Extract train schedules
        total_delay = 0
        for train in trains:
            train_solution = {
                'train_id': train.train_id,
                'train_type': train.train_type,
                'priority': train.priority,
                'arrival_time': self.solver.Value(variables['arrival_times'][train.id]),
                'departure_time': self.solver.Value(variables['departure_times'][train.id]),
                'delay_minutes': self.solver.Value(variables['delay_vars'][train.id]),
                'segment_schedule': {},
                'speed_profile': {},
                'platform_assignment': None
            }
            
            total_delay += train_solution['delay_minutes']
            
            # Extract segment schedule
            for segment in self.segments:
                entry_time = self.solver.Value(variables['segment_entry'][train.id][segment.id])
                exit_time = self.solver.Value(variables['segment_exit'][train.id][segment.id])
                speed_percent = self.solver.Value(variables['speed_profile'][train.id][segment.id])
                
                train_solution['segment_schedule'][segment.id] = {
                    'segment_name': segment.name,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'travel_time': exit_time - entry_time
                }
                
                train_solution['speed_profile'][segment.id] = {
                    'speed_percent': speed_percent,
                    'actual_speed_kmh': int(min(train.max_speed, segment.effective_max_speed) * speed_percent / 100)
                }
            
            # Extract platform assignment
            if train.id in variables['platform_assignment']:
                for platform_id, platform_var in variables['platform_assignment'][train.id].items():
                    if self.solver.Value(platform_var):
                        platform = Platform.objects.get(id=platform_id)
                        train_solution['platform_assignment'] = {
                            'platform_id': platform_id,
                            'platform_name': platform.name
                        }
                        break
            
            solution['trains'][train.id] = train_solution
        
        # Generate actionable recommendations
        solution['recommendations'] = self._generate_recommendations(trains, variables, solution)
        
        # Calculate KPIs
        avg_delay = total_delay / len(trains) if trains else 0
        max_departure = max([sol['departure_time'] for sol in solution['trains'].values()])
        throughput = len(trains) / (max_departure / 60) if max_departure > 0 else 0
        
        solution['kpis'] = {
            'avg_delay_minutes': avg_delay,
            'total_throughput': throughput,
            'punctuality_percent': len([t for t in solution['trains'].values() 
                                      if t['delay_minutes'] <= 5]) / len(trains) * 100,
            'fuel_efficiency_score': self._calculate_fuel_efficiency(solution),
            'capacity_utilization': min(100, len(trains) / self.section.capacity * 100)
        }
        
        # Store solve history for learning
        self.solve_history.append({
            'timestamp': timezone.now(),
            'num_trains': len(trains),
            'solve_time': solve_time,
            'objective_value': solution['objective_value'],
            'status': solution['status']
        })
        
        return solution
    
    def _generate_recommendations(self, trains, variables, solution):
        """Generate actionable recommendations for traffic controllers"""
        recommendations = []
        
        # Priority-based recommendations
        for train_id, train_sol in solution['trains'].items():
            train = next(t for t in trains if t.id == train_id)
            
            if train_sol['delay_minutes'] > 10:
                recommendations.append({
                    'type': 'delay_warning',
                    'priority': 'high',
                    'train_id': train.train_id,
                    'message': f"Train {train.train_id} expected delay: {train_sol['delay_minutes']:.1f} minutes",
                    'action': 'Consider priority routing or platform reallocation',
                    'confidence': 0.9
                })
            
            # Speed recommendations
            for segment_id, speed_data in train_sol['speed_profile'].items():
                if speed_data['speed_percent'] < 50:
                    segment = next(s for s in self.segments if s.id == segment_id)
                    recommendations.append({
                        'type': 'speed_restriction',
                        'priority': 'medium',
                        'train_id': train.train_id,
                        'message': f"Reduce speed to {speed_data['actual_speed_kmh']} km/h in {segment.name}",
                        'action': f"Set speed restriction due to optimization",
                        'confidence': 0.8
                    })
        
        # System-wide recommendations
        if solution['kpis']['capacity_utilization'] > 90:
            recommendations.append({
                'type': 'capacity_warning',
                'priority': 'high',
                'message': "Section approaching capacity limit",
                'action': "Consider delaying non-critical trains or using alternative routes",
                'confidence': 0.95
            })
        
        return recommendations
    
    def _calculate_fuel_efficiency(self, solution):
        """Calculate fuel efficiency score based on speed profiles"""
        efficiency_scores = []
        
        for train_data in solution['trains'].values():
            train_score = 0
            segment_count = 0
            
            for segment_data in train_data['speed_profile'].values():
                speed_percent = segment_data['speed_percent']
                # Optimal efficiency around 60-80% speed
                if 60 <= speed_percent <= 80:
                    train_score += 100
                else:
                    # Penalty for inefficient speeds
                    deviation = min(abs(speed_percent - 60), abs(speed_percent - 80))
                    train_score += max(0, 100 - deviation * 2)
                
                segment_count += 1
            
            if segment_count > 0:
                efficiency_scores.append(train_score / segment_count)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0
    
    def get_optimization_insights(self):
        """Get insights from optimization history for continuous improvement"""
        if len(self.solve_history) < 5:
            return {'message': 'Insufficient data for insights'}
        
        recent_solves = self.solve_history[-10:]
        
        avg_solve_time = np.mean([h['solve_time'] for h in recent_solves])
        success_rate = len([h for h in recent_solves if h['status'] == 'optimal']) / len(recent_solves)
        
        insights = {
            'avg_solve_time': avg_solve_time,
            'success_rate': success_rate,
            'recommendations': []
        }
        
        if avg_solve_time > 20:
            insights['recommendations'].append(
                "Consider reducing time horizon or simplifying constraints for faster solutions"
            )
        
        if success_rate < 0.8:
            insights['recommendations'].append(
                "Frequent infeasible solutions detected. Review constraint parameters."
            )
        
        return insights

# Legacy alias for backward compatibility
ILPEngine = AdvancedILPEngine