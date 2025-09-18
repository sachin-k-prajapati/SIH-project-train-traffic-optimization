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
        self.solver.parameters.max_time_in_seconds = 5.0  # Real-time requirement: 5 seconds max
        self.solver.parameters.num_search_workers = 4  # Parallel search for speed
        self.solver.parameters.log_search_progress = False  # Reduce logging overhead
        
        # Load section topology
        self.segments = list(Segment.objects.filter(section=section).order_by('sequence'))
        self.signals = Signal.objects.filter(segment__section=section)
        self.platforms = Platform.objects.filter(segment__section=section)
        
        # Performance tracking for continuous improvement
        self.solve_history = []
        self.throughput_history = []
        self.decision_cache = {}  # Cache for similar scenarios
        
        # Throughput optimization parameters
        self.throughput_target = section.capacity * 0.85  # Target 85% utilization
        self.bottleneck_segments = self._identify_bottlenecks()
        
        logger.info(f"Initialized Advanced ILP Engine for {section.name} - Real-time mode (5s limit)")
    
    def optimize_comprehensive_schedule(self, trains, disruptions=None, time_horizon_minutes=180):
        """
        Comprehensive optimization focusing on throughput maximization with 5-second response time
        """
        logger.info(f"Starting real-time optimization for {len(trains)} trains (5s limit)")
        
        if not trains:
            return {'status': 'no_trains', 'message': 'No trains to optimize'}
        
        start_time = timezone.now()
        
        # Check cache for similar scenarios
        scenario_key = self._generate_scenario_key(trains, disruptions)
        if scenario_key in self.decision_cache:
            cached_solution = self.decision_cache[scenario_key]
            logger.info("Using cached solution for similar scenario")
            return self._adapt_cached_solution(cached_solution, trains)
        
        # Initialize model
        self.model = cp_model.CpModel()
        
        # Decision variables optimized for throughput
        variables = self._create_throughput_optimized_variables(trains, time_horizon_minutes)
        
        # Add constraints with priority on throughput-critical ones
        self._add_throughput_constraints(trains, variables)
        self._add_safety_constraints_optimized(trains, variables)
        self._add_conflict_resolution_constraints(trains, variables)
        
        if disruptions:
            self._add_disruption_constraints(trains, variables, disruptions)
        
        # Throughput-focused multi-objective optimization
        objective = self._create_throughput_objective(trains, variables)
        self.model.Minimize(objective)
        
        # Solve with strict 5-second limit
        status = self.solver.Solve(self.model)
        solve_time = (timezone.now() - start_time).total_seconds()
        
        # Extract and return solution
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = self._extract_throughput_solution(trains, variables, status, solve_time)
            
            # Cache solution for future use
            self.decision_cache[scenario_key] = solution
            
            # Update throughput tracking
            self._update_throughput_metrics(solution)
            
            logger.info(f"Real-time optimization completed in {solve_time:.2f}s")
            return solution
        else:
            logger.warning(f"Optimization failed in {solve_time:.2f}s - using fallback strategy")
            return self._generate_fallback_solution(trains, solve_time)
    
    def _identify_bottlenecks(self):
        """Identify bottleneck segments that limit throughput"""
        bottlenecks = []
        for segment in self.segments:
            # Check capacity, signal density, gradient
            bottleneck_score = 0
            
            if segment.max_occupancy <= 2:
                bottleneck_score += 3
            if segment.gradient_percent > 2:
                bottleneck_score += 2
            if not hasattr(segment, 'signals') or len(segment.signals.all()) < 2:
                bottleneck_score += 2
            
            if bottleneck_score >= 4:
                bottlenecks.append(segment)
        
        return bottlenecks
    
    def _generate_scenario_key(self, trains, disruptions):
        """Generate cache key for scenario similarity"""
        train_signature = tuple(sorted([
            (t.train_type, t.priority, t.current_delay_minutes // 5)  # Round to 5-min intervals
            for t in trains
        ]))
        
        disruption_signature = tuple(sorted([
            (d.type, getattr(d, 'affected_segment_id', None))
            for d in (disruptions or [])
        ]))
        
        return hash((train_signature, disruption_signature, self.section.current_weather))
    
    def _adapt_cached_solution(self, cached_solution, current_trains):
        """Adapt cached solution to current trains"""
        adapted = cached_solution.copy()
        adapted['status'] = 'cached_adapted'
        adapted['solve_time'] = 0.1  # Minimal adaptation time
        
        # Update train mappings and timing
        train_mapping = {}
        for i, train in enumerate(current_trains):
            if i < len(cached_solution.get('trains', {})):
                train_mapping[train.id] = list(cached_solution['trains'].keys())[i]
        
        # Adapt recommendations
        adapted['recommendations'] = [
            {**rec, 'confidence': rec.get('confidence', 0.8) * 0.9}  # Slightly lower confidence
            for rec in cached_solution.get('recommendations', [])
        ]
        
        return adapted
    
    def _generate_fallback_solution(self, trains, solve_time):
        """Generate heuristic solution when optimization fails"""
        logger.info("Generating heuristic fallback solution")
        
        # Simple priority-based scheduling
        sorted_trains = sorted(trains, key=lambda t: (t.priority, t.current_delay_minutes))
        
        solution = {
            'status': 'heuristic_fallback',
            'solve_time': solve_time,
            'trains': {},
            'recommendations': [],
            'kpis': {},
            'algorithm_confidence': 'low'
        }
        
        current_time = 0
        for train in sorted_trains:
            # Simple sequential scheduling
            arrival_time = current_time + train.current_delay_minutes
            departure_time = arrival_time + (5 if train.train_type == 'express' else 
                                           3 if train.train_type == 'local' else 8)
            
            solution['trains'][train.id] = {
                'train_id': train.train_id,
                'arrival_time': arrival_time,
                'departure_time': departure_time,
                'delay_minutes': train.current_delay_minutes,
                'platform_assignment': None
            }
            
            current_time = departure_time + 2  # 2-minute gap
        
        solution['recommendations'].append({
            'type': 'system_warning',
            'priority': 'high',
            'message': 'Optimization timeout - using heuristic solution',
            'action': 'Monitor closely and consider reducing train load',
            'confidence': 0.6
        })
        
        return solution
    
    def optimize_precedence(self, active_trains, current_time):
        """Legacy method for compatibility - calls comprehensive optimization"""
        return self.optimize_comprehensive_schedule(active_trains, time_horizon_minutes=120)
    
    def _create_throughput_optimized_variables(self, trains, time_horizon):
        """Create decision variables optimized for throughput maximization"""
        variables = {
            'arrival_times': {},      # When train arrives at section
            'departure_times': {},    # When train departs section
            'segment_slots': {},      # Time slots for segment usage
            'precedence': {},         # Train precedence decisions
            'throughput_vars': {},    # Variables for throughput calculation
            'conflict_resolution': {},  # Variables for conflict resolution
        }
        
        # Discretize time into 1-minute slots for throughput optimization
        time_slots = list(range(0, time_horizon, 1))
        
        # Time variables for each train
        for train in trains:
            variables['arrival_times'][train.id] = self.model.NewIntVar(
                0, time_horizon, f'arrival_{train.id}'
            )
            variables['departure_times'][train.id] = self.model.NewIntVar(
                0, time_horizon, f'departure_{train.id}'
            )
            
            # Segment slot assignment (binary variables for each time slot)
            variables['segment_slots'][train.id] = {}
            for segment in self.segments:
                variables['segment_slots'][train.id][segment.id] = {}
                for slot in time_slots:
                    variables['segment_slots'][train.id][segment.id][slot] = \
                        self.model.NewBoolVar(f'slot_{train.id}_{segment.id}_{slot}')
        
        # Throughput measurement variables
        total_throughput = self.model.NewIntVar(0, len(trains), 'total_throughput')
        variables['throughput_vars']['total'] = total_throughput
        
        # Bottleneck utilization variables
        for segment in self.bottleneck_segments:
            utilization = self.model.NewIntVar(0, time_horizon, f'util_{segment.id}')
            variables['throughput_vars'][f'bottleneck_{segment.id}'] = utilization
        
        # Conflict resolution variables
        for i, train1 in enumerate(trains):
            for train2 in trains[i+1:]:
                conflict_var = self.model.NewBoolVar(f'conflict_{train1.id}_{train2.id}')
                variables['conflict_resolution'][(train1.id, train2.id)] = conflict_var
        
        return variables
    
    def _add_throughput_constraints(self, trains, variables):
        """Add constraints focused on maximizing throughput"""
        logger.debug("Adding throughput optimization constraints")
        
        # Capacity constraints with throughput focus
        for segment in self.segments:
            max_concurrent = segment.max_occupancy
            
            for time_slot in range(0, 180, 1):  # Every minute
                slot_usage = []
                
                for train in trains:
                    if segment.id in variables['segment_slots'][train.id]:
                        if time_slot in variables['segment_slots'][train.id][segment.id]:
                            slot_usage.append(
                                variables['segment_slots'][train.id][segment.id][time_slot]
                            )
                
                if slot_usage:
                    self.model.Add(sum(slot_usage) <= max_concurrent)
        
        # Bottleneck optimization
        for segment in self.bottleneck_segments:
            total_usage = []
            for train in trains:
                for time_slot in range(0, 180, 1):
                    if (segment.id in variables['segment_slots'][train.id] and 
                        time_slot in variables['segment_slots'][train.id][segment.id]):
                        total_usage.append(
                            variables['segment_slots'][train.id][segment.id][time_slot]
                        )
            
            if total_usage and f'bottleneck_{segment.id}' in variables['throughput_vars']:
                self.model.Add(
                    variables['throughput_vars'][f'bottleneck_{segment.id}'] == sum(total_usage)
                )
        
        # Link slot variables to arrival/departure times
        for train in trains:
            for segment in self.segments:
                if segment.id in variables['segment_slots'][train.id]:
                    # Train must use consecutive slots
                    slot_vars = [
                        variables['segment_slots'][train.id][segment.id].get(t, 
                            self.model.NewBoolVar(f'dummy_{train.id}_{segment.id}_{t}'))
                        for t in range(0, 180, 1)
                    ]
                    
                    # At least one slot must be used
                    self.model.Add(sum(slot_vars) >= 1)
    
    def _add_conflict_resolution_constraints(self, trains, variables):
        """Add sophisticated conflict resolution constraints"""
        logger.debug("Adding conflict resolution constraints")
        
        for i, train1 in enumerate(trains):
            for train2 in trains[i+1:]:
                conflict_var = variables['conflict_resolution'][(train1.id, train2.id)]
                
                # Check for potential conflicts across all segments
                segment_conflicts = []
                
                for segment in self.segments:
                    for time_slot in range(0, 180, 1):
                        if (segment.id in variables['segment_slots'][train1.id] and 
                            segment.id in variables['segment_slots'][train2.id]):
                            
                            slot1 = variables['segment_slots'][train1.id][segment.id].get(time_slot)
                            slot2 = variables['segment_slots'][train2.id][segment.id].get(time_slot)
                            
                            if slot1 and slot2:
                                # Both trains cannot use same segment at same time
                                segment_conflict = self.model.NewBoolVar(
                                    f'seg_conflict_{train1.id}_{train2.id}_{segment.id}_{time_slot}'
                                )
                                
                                self.model.Add(slot1 + slot2 <= 1 + segment_conflict)
                                segment_conflicts.append(segment_conflict)
                
                # If any segment conflict exists, mark overall conflict
                if segment_conflicts:
                    self.model.Add(conflict_var >= max(segment_conflicts) if segment_conflicts else 0)
    
    def _create_throughput_objective(self, trains, variables):
        """Create objective function focusing on throughput maximization"""
        objective_terms = []
        
        # 1. Maximize throughput (highest priority weight: 10000)
        throughput_weight = 10000
        completed_trains = []
        
        for train in trains:
            # Train is completed if it has departed before time horizon
            completed = self.model.NewBoolVar(f'completed_{train.id}')
            departure = variables['departure_times'][train.id]
            
            self.model.Add(departure <= 179).OnlyEnforceIf(completed)
            self.model.Add(departure > 179).OnlyEnforceIf(completed.Not())
            
            completed_trains.append(completed)
        
        total_completed = sum(completed_trains)
        variables['throughput_vars']['total'] = total_completed
        
        # Maximize completed trains (negative because we minimize)
        objective_terms.append(-total_completed * throughput_weight)
        
        # 2. Minimize makespan (total time to complete all trains)
        makespan_weight = 5000
        makespan = self.model.NewIntVar(0, 180, 'makespan')
        for train in trains:
            departure = variables['departure_times'][train.id]
            self.model.Add(makespan >= departure)
        
        objective_terms.append(makespan * makespan_weight)
        
        # 3. Maximize bottleneck utilization
        utilization_weight = 1000
        for segment in self.bottleneck_segments:
            if f'bottleneck_{segment.id}' in variables['throughput_vars']:
                utilization = variables['throughput_vars'][f'bottleneck_{segment.id}']
                # Negative because we want to maximize utilization
                objective_terms.append(-utilization * utilization_weight)
        
        # 4. Minimize conflicts
        conflict_weight = 2000
        for conflict_var in variables['conflict_resolution'].values():
            objective_terms.append(conflict_var * conflict_weight)
        
        # 5. Priority-based scheduling
        priority_weight = 1500
        for train in trains:
            arrival = variables['arrival_times'][train.id]
            priority_factor = 6 - train.priority  # Higher priority = lower cost
            objective_terms.append(arrival * priority_weight * priority_factor)
        
        return sum(objective_terms)
    
    def _add_safety_constraints_optimized(self, trains, variables):
        """Add essential safety constraints optimized for speed"""
        logger.debug("Adding optimized safety constraints")
        
        for train in trains:
            # Minimum time constraints (simplified for speed)
            arrival = variables['arrival_times'][train.id]
            departure = variables['departure_times'][train.id]
            
            # Minimum dwell time based on train type
            min_dwell = 3 if train.train_type == 'express' else 2 if train.train_type == 'local' else 5
            self.model.Add(departure >= arrival + min_dwell)
            
            # Sequential constraints for adjacent segments only (optimization)
            for i, segment in enumerate(self.segments[:-1]):
                next_segment = self.segments[i + 1]
                
                # Simplified: train must finish current segment before next
                current_slots = variables['segment_slots'][train.id].get(segment.id, {})
                next_slots = variables['segment_slots'][train.id].get(next_segment.id, {})
                
                if current_slots and next_slots:
                    # Find last slot used in current segment and first in next
                    for t1 in range(179, -1, -1):  # Latest first
                        if t1 in current_slots:
                            for t2 in range(0, 180):  # Earliest first
                                if t2 in next_slots:
                                    self.model.Add(t2 >= t1 + 1).OnlyEnforceIf(
                                        [current_slots[t1], next_slots[t2]]
                                    )
                                    break
                            break
    
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
    
    def _extract_throughput_solution(self, trains, variables, status, solve_time):
        """Extract solution with focus on throughput metrics and real-time decisions"""
        solution = {
            'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',
            'solve_time': solve_time,
            'objective_value': self.solver.ObjectiveValue(),
            'trains': {},
            'recommendations': [],
            'kpis': {},
            'algorithm_confidence': 'high' if status == cp_model.OPTIMAL else 'medium',
            'throughput_analysis': {}
        }
        
        # Extract train schedules with throughput focus
        completed_trains = 0
        total_conflicts = 0
        
        for train in trains:
            arrival_time = self.solver.Value(variables['arrival_times'][train.id])
            departure_time = self.solver.Value(variables['departure_times'][train.id])
            
            # Check if train completes within time horizon
            is_completed = departure_time <= 179
            if is_completed:
                completed_trains += 1
            
            train_solution = {
                'train_id': train.train_id,
                'train_type': train.train_type,
                'priority': train.priority,
                'arrival_time': arrival_time,
                'departure_time': departure_time,
                'total_time': departure_time - arrival_time,
                'completed': is_completed,
                'segment_utilization': {},
                'conflicts': []
            }
            
            # Extract segment utilization
            total_segment_time = 0
            for segment in self.segments:
                segment_usage = 0
                if segment.id in variables['segment_slots'][train.id]:
                    for time_slot in range(0, 180):
                        if (time_slot in variables['segment_slots'][train.id][segment.id] and
                            self.solver.Value(variables['segment_slots'][train.id][segment.id][time_slot])):
                            segment_usage += 1
                
                train_solution['segment_utilization'][segment.id] = {
                    'segment_name': segment.name,
                    'time_used': segment_usage,
                    'utilization_rate': segment_usage / 60 if segment_usage > 0 else 0  # Convert to hours
                }
                total_segment_time += segment_usage
            
            train_solution['total_segment_time'] = total_segment_time
            
            # Check for conflicts with other trains
            for other_train in trains:
                if other_train.id != train.id:
                    conflict_key = (min(train.id, other_train.id), max(train.id, other_train.id))
                    if conflict_key in variables['conflict_resolution']:
                        if self.solver.Value(variables['conflict_resolution'][conflict_key]):
                            train_solution['conflicts'].append(other_train.train_id)
                            total_conflicts += 1
            
            solution['trains'][train.id] = train_solution
        
        # Calculate throughput metrics
        time_horizon_hours = 180 / 60  # 3 hours
        actual_throughput = completed_trains / time_horizon_hours
        max_possible_throughput = len(trains) / time_horizon_hours
        throughput_efficiency = (actual_throughput / max_possible_throughput) * 100 if max_possible_throughput > 0 else 0
        
        # Bottleneck analysis
        bottleneck_utilization = {}
        for segment in self.bottleneck_segments:
            if f'bottleneck_{segment.id}' in variables['throughput_vars']:
                utilization = self.solver.Value(variables['throughput_vars'][f'bottleneck_{segment.id}'])
                bottleneck_utilization[segment.id] = {
                    'segment_name': segment.name,
                    'utilization_minutes': utilization,
                    'utilization_percent': (utilization / 180) * 100
                }
        
        solution['throughput_analysis'] = {
            'completed_trains': completed_trains,
            'total_trains': len(trains),
            'completion_rate': (completed_trains / len(trains)) * 100,
            'actual_throughput_per_hour': actual_throughput,
            'max_possible_throughput_per_hour': max_possible_throughput,
            'throughput_efficiency_percent': throughput_efficiency,
            'total_conflicts': total_conflicts // 2,  # Each conflict counted twice
            'bottleneck_utilization': bottleneck_utilization
        }
        
        # Generate throughput-focused recommendations
        solution['recommendations'] = self._generate_throughput_recommendations(trains, solution)
        
        # Calculate comprehensive KPIs
        solution['kpis'] = self._calculate_throughput_kpis(solution)
        
        return solution
    
    def _generate_throughput_recommendations(self, trains, solution):
        """Generate recommendations focused on throughput optimization"""
        recommendations = []
        
        throughput_analysis = solution['throughput_analysis']
        
        # Throughput efficiency recommendations
        if throughput_analysis['throughput_efficiency_percent'] < 70:
            recommendations.append({
                'type': 'throughput_warning',
                'priority': 'high',
                'message': f"Low throughput efficiency: {throughput_analysis['throughput_efficiency_percent']:.1f}%",
                'action': 'Consider optimizing train schedules or reducing non-essential trains',
                'confidence': 0.9,
                'impact': 'high'
            })
        
        # Bottleneck recommendations
        for segment_id, bottleneck_data in throughput_analysis['bottleneck_utilization'].items():
            if bottleneck_data['utilization_percent'] > 90:
                recommendations.append({
                    'type': 'bottleneck_alert',
                    'priority': 'critical',
                    'message': f"Bottleneck at {bottleneck_data['segment_name']}: {bottleneck_data['utilization_percent']:.1f}% utilization",
                    'action': 'Immediate intervention required - consider alternative routing',
                    'confidence': 0.95,
                    'impact': 'critical'
                })
            elif bottleneck_data['utilization_percent'] > 75:
                recommendations.append({
                    'type': 'bottleneck_warning',
                    'priority': 'high',
                    'message': f"High utilization at {bottleneck_data['segment_name']}: {bottleneck_data['utilization_percent']:.1f}%",
                    'action': 'Monitor closely and prepare contingency plans',
                    'confidence': 0.85,
                    'impact': 'medium'
                })
        
        # Conflict resolution recommendations
        if throughput_analysis['total_conflicts'] > 0:
            recommendations.append({
                'type': 'conflict_resolution',
                'priority': 'high',
                'message': f"{throughput_analysis['total_conflicts']} train conflicts detected",
                'action': 'Implement precedence decisions and monitor train separation',
                'confidence': 0.8,
                'impact': 'medium'
            })
        
        # Completion rate recommendations
        if throughput_analysis['completion_rate'] < 90:
            incomplete_trains = [
                train_data for train_data in solution['trains'].values()
                if not train_data['completed']
            ]
            
            recommendations.append({
                'type': 'completion_warning',
                'priority': 'medium',
                'message': f"{len(incomplete_trains)} trains may not complete within time horizon",
                'action': 'Extend operating hours or defer low-priority trains',
                'confidence': 0.75,
                'impact': 'medium'
            })
        
        # Real-time decision recommendations
        current_time = 0  # Would be actual current time in real implementation
        immediate_actions = []
        
        for train_data in solution['trains'].values():
            if train_data['arrival_time'] <= current_time + 10:  # Within next 10 minutes
                immediate_actions.append({
                    'train_id': train_data['train_id'],
                    'action': 'proceed' if not train_data['conflicts'] else 'hold',
                    'timing': train_data['arrival_time']
                })
        
        if immediate_actions:
            recommendations.append({
                'type': 'immediate_action',
                'priority': 'urgent',
                'message': f"{len(immediate_actions)} trains require immediate decisions",
                'action': 'Execute precedence decisions as calculated',
                'confidence': 0.9,
                'impact': 'high',
                'details': immediate_actions
            })
        
        return recommendations
    
    def _calculate_throughput_kpis(self, solution):
        """Calculate comprehensive KPIs with focus on throughput"""
        throughput_analysis = solution['throughput_analysis']
        
        # Basic throughput KPIs
        kpis = {
            'throughput_per_hour': throughput_analysis['actual_throughput_per_hour'],
            'throughput_efficiency_percent': throughput_analysis['throughput_efficiency_percent'],
            'completion_rate_percent': throughput_analysis['completion_rate'],
            'conflict_rate': throughput_analysis['total_conflicts'] / len(solution['trains']) if solution['trains'] else 0,
            'optimization_quality': 'optimal' if solution['status'] == 'optimal' else 'good',
            'real_time_performance': 'excellent' if solution['solve_time'] <= 3 else 'good' if solution['solve_time'] <= 5 else 'poor'
        }
        
        # Advanced KPIs
        total_travel_time = sum([
            train_data['total_time'] for train_data in solution['trains'].values()
        ])
        avg_travel_time = total_travel_time / len(solution['trains']) if solution['trains'] else 0
        
        kpis.update({
            'avg_travel_time_minutes': avg_travel_time,
            'section_utilization_percent': min(100, throughput_analysis['actual_throughput_per_hour'] / self.section.capacity * 100),
            'bottleneck_efficiency': self._calculate_bottleneck_efficiency(throughput_analysis),
            'decision_confidence': solution['algorithm_confidence']
        })
        
        return kpis
    
    def _calculate_bottleneck_efficiency(self, throughput_analysis):
        """Calculate efficiency of bottleneck segment usage"""
        if not throughput_analysis['bottleneck_utilization']:
            return 100  # No bottlenecks identified
        
        total_utilization = sum([
            bottleneck['utilization_percent'] 
            for bottleneck in throughput_analysis['bottleneck_utilization'].values()
        ])
        
        avg_utilization = total_utilization / len(throughput_analysis['bottleneck_utilization'])
        
        # Efficiency is optimal around 75-80% utilization
        if 75 <= avg_utilization <= 80:
            return 100
        elif avg_utilization < 75:
            return avg_utilization / 75 * 100
        else:
            return max(0, 100 - (avg_utilization - 80) * 2)
    
    def _update_throughput_metrics(self, solution):
        """Update historical throughput metrics for continuous improvement"""
        throughput_data = {
            'timestamp': timezone.now(),
            'throughput_per_hour': solution['kpis']['throughput_per_hour'],
            'efficiency_percent': solution['kpis']['throughput_efficiency_percent'],
            'solve_time': solution['solve_time'],
            'conflicts': solution['throughput_analysis']['total_conflicts']
        }
        
        self.throughput_history.append(throughput_data)
        
        # Keep only last 100 records
        if len(self.throughput_history) > 100:
            self.throughput_history = self.throughput_history[-100:]
    
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
