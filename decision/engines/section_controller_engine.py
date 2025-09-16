"""
Section Controller Decision Support System
Core optimization engine for train precedence and crossing decisions
"""

from ortools.sat.python import cp_model
from typing import List, Dict, Tuple, Optional
import simpy
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum

class DecisionType(Enum):
    PROCEED = "proceed"
    HALT = "halt"
    REROUTE = "reroute"
    PLATFORM_CHANGE = "platform_change"

@dataclass
class TrainState:
    train_id: str
    current_segment: str
    priority: int  # 1 = highest, 5 = lowest
    train_type: str  # express, local, freight
    schedule_adherence: float  # minutes behind/ahead (-ve = early, +ve = late)
    destination_platform: Optional[str]
    expected_arrival: datetime
    passenger_count: int
    freight_value: Optional[float]

@dataclass
class SectionConstraints:
    track_capacity: Dict[str, int]  # segment_id -> max concurrent trains
    platform_capacity: Dict[str, int]  # platform_id -> capacity
    signal_blocks: List[str]  # blocked signal segments
    maintenance_windows: List[Tuple[str, datetime, datetime]]  # segment, start, end
    weather_impact_factor: float  # 0.0 to 1.0
    minimum_headway_minutes: Dict[str, int]  # train_type -> min gap in minutes

@dataclass
class OptimizationObjectives:
    maximize_throughput: float = 0.4
    minimize_delays: float = 0.3
    respect_priorities: float = 0.2
    minimize_platform_conflicts: float = 0.1

class SectionControllerOptimizer:
    """
    Core optimization engine for section controller decisions
    Uses Constraint Programming (CP-SAT) for optimal train precedence
    """
    
    def __init__(self, section_id: str, time_horizon_minutes: int = 120):
        self.section_id = section_id
        self.time_horizon = time_horizon_minutes
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
    def optimize_train_precedence(
        self, 
        trains: List[TrainState], 
        constraints: SectionConstraints,
        objectives: OptimizationObjectives
    ) -> Dict[str, DecisionType]:
        """
        Solve train precedence optimization problem
        Returns decisions for each train
        """
        
        # Decision variables
        train_decisions = {}
        platform_assignments = {}
        crossing_times = {}
        
        # Create decision variables for each train
        for train in trains:
            # Binary variables for decision type
            train_decisions[train.train_id] = {
                'proceed': self.model.NewBoolVar(f'proceed_{train.train_id}'),
                'halt': self.model.NewBoolVar(f'halt_{train.train_id}'),
                'reroute': self.model.NewBoolVar(f'reroute_{train.train_id}')
            }
            
            # Exactly one decision per train
            self.model.AddExactlyOne([
                train_decisions[train.train_id]['proceed'],
                train_decisions[train.train_id]['halt'],
                train_decisions[train.train_id]['reroute']
            ])
            
            # Platform assignment variables
            if train.destination_platform:
                platform_assignments[train.train_id] = self.model.NewIntVar(
                    0, len(constraints.platform_capacity), 
                    f'platform_{train.train_id}'
                )
            
            # Crossing time variables (minutes from now)
            crossing_times[train.train_id] = self.model.NewIntVar(
                0, self.time_horizon, 
                f'crossing_time_{train.train_id}'
            )
        
        # SAFETY CONSTRAINTS
        self._add_safety_constraints(trains, constraints, train_decisions, crossing_times)
        
        # CAPACITY CONSTRAINTS  
        self._add_capacity_constraints(trains, constraints, train_decisions, platform_assignments)
        
        # PRIORITY CONSTRAINTS
        self._add_priority_constraints(trains, train_decisions, crossing_times)
        
        # HEADWAY CONSTRAINTS
        self._add_headway_constraints(trains, constraints, crossing_times)
        
        # OBJECTIVE FUNCTION
        self._set_objective_function(trains, objectives, train_decisions, crossing_times)
        
        # Solve the model
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._extract_decisions(trains, train_decisions)
        else:
            # Fallback to heuristic if no optimal solution
            return self._heuristic_fallback(trains, constraints)
    
    def _add_safety_constraints(self, trains, constraints, train_decisions, crossing_times):
        """Add safety-related constraints"""
        
        # Minimum separation between trains on same track
        for i, train1 in enumerate(trains):
            for j, train2 in enumerate(trains[i+1:], i+1):
                if train1.current_segment == train2.current_segment:
                    # Ensure minimum 5-minute separation
                    self.model.Add(
                        crossing_times[train1.train_id] + 5 <= crossing_times[train2.train_id]
                    ).OnlyEnforceIf([
                        train_decisions[train1.train_id]['proceed'],
                        train_decisions[train2.train_id]['proceed']
                    ])
        
        # Block trains from proceeding through maintenance areas
        for train in trains:
            for segment, start_time, end_time in constraints.maintenance_windows:
                if train.current_segment == segment:
                    # Force halt if maintenance is active
                    self.model.Add(
                        train_decisions[train.train_id]['halt'] == 1
                    )
    
    def _add_capacity_constraints(self, trains, constraints, train_decisions, platform_assignments):
        """Add track and platform capacity constraints"""
        
        # Track capacity constraints
        for segment, max_capacity in constraints.track_capacity.items():
            trains_on_segment = [t for t in trains if t.current_segment == segment]
            if len(trains_on_segment) > max_capacity:
                # Force some trains to halt or reroute
                halt_vars = []
                for train in trains_on_segment:
                    halt_vars.append(train_decisions[train.train_id]['halt'])
                    halt_vars.append(train_decisions[train.train_id]['reroute'])
                
                # At least (count - capacity) trains must halt/reroute
                self.model.Add(
                    sum(halt_vars) >= len(trains_on_segment) - max_capacity
                )
        
        # Platform capacity constraints
        for platform, capacity in constraints.platform_capacity.items():
            platform_users = []
            for train in trains:
                if train.destination_platform == platform:
                    platform_users.append(train)
            
            if len(platform_users) > capacity:
                # Some trains must be rerouted to different platforms
                reroute_vars = [
                    train_decisions[train.train_id]['reroute'] 
                    for train in platform_users
                ]
                self.model.Add(
                    sum(reroute_vars) >= len(platform_users) - capacity
                )
    
    def _add_priority_constraints(self, trains, train_decisions, crossing_times):
        """Add train priority constraints"""
        
        # Higher priority trains should cross first
        for i, train1 in enumerate(trains):
            for j, train2 in enumerate(trains[i+1:], i+1):
                if train1.priority < train2.priority:  # Lower number = higher priority
                    # If both trains proceed, train1 should cross first
                    self.model.Add(
                        crossing_times[train1.train_id] <= crossing_times[train2.train_id]
                    ).OnlyEnforceIf([
                        train_decisions[train1.train_id]['proceed'],
                        train_decisions[train2.train_id]['proceed']
                    ])
    
    def _add_headway_constraints(self, trains, constraints, crossing_times):
        """Add minimum headway constraints based on train types"""
        
        for i, train1 in enumerate(trains):
            for j, train2 in enumerate(trains[i+1:], i+1):
                if train1.current_segment == train2.current_segment:
                    min_headway = max(
                        constraints.minimum_headway_minutes.get(train1.train_type, 3),
                        constraints.minimum_headway_minutes.get(train2.train_type, 3)
                    )
                    
                    self.model.Add(
                        crossing_times[train1.train_id] + min_headway <= crossing_times[train2.train_id]
                    ).OnlyEnforceIf([
                        train_decisions[train1.train_id]['proceed'],
                        train_decisions[train2.train_id]['proceed']
                    ])
    
    def _set_objective_function(self, trains, objectives, train_decisions, crossing_times):
        """Set multi-objective optimization function"""
        
        objective_terms = []
        
        # Maximize throughput (minimize halted trains)
        halted_trains = sum([
            train_decisions[train.train_id]['halt'] 
            for train in trains
        ])
        objective_terms.append(-objectives.maximize_throughput * 1000 * halted_trains)
        
        # Minimize delays (prioritize on-time trains)
        for train in trains:
            delay_penalty = max(0, train.schedule_adherence)  # Only penalize late trains
            proceed_var = train_decisions[train.train_id]['proceed']
            objective_terms.append(
                -objectives.minimize_delays * delay_penalty * proceed_var
            )
        
        # Respect priorities (higher priority trains get preference)
        for train in trains:
            priority_bonus = (6 - train.priority) * 100  # Higher bonus for higher priority
            proceed_var = train_decisions[train.train_id]['proceed']
            objective_terms.append(objectives.respect_priorities * priority_bonus * proceed_var)
        
        # Minimize platform conflicts
        rerouted_trains = sum([
            train_decisions[train.train_id]['reroute'] 
            for train in trains
        ])
        objective_terms.append(-objectives.minimize_platform_conflicts * 500 * rerouted_trains)
        
        # Set the objective
        self.model.Maximize(sum(objective_terms))
    
    def _extract_decisions(self, trains, train_decisions) -> Dict[str, DecisionType]:
        """Extract decisions from solved model"""
        
        decisions = {}
        for train in trains:
            if self.solver.Value(train_decisions[train.train_id]['proceed']):
                decisions[train.train_id] = DecisionType.PROCEED
            elif self.solver.Value(train_decisions[train.train_id]['halt']):
                decisions[train.train_id] = DecisionType.HALT
            elif self.solver.Value(train_decisions[train.train_id]['reroute']):
                decisions[train.train_id] = DecisionType.REROUTE
        
        return decisions
    
    def _heuristic_fallback(self, trains, constraints) -> Dict[str, DecisionType]:
        """Fallback heuristic when optimization fails"""
        
        decisions = {}
        
        # Sort trains by priority and schedule adherence
        sorted_trains = sorted(trains, key=lambda t: (t.priority, t.schedule_adherence))
        
        # Simple greedy allocation
        for train in sorted_trains:
            # Check capacity constraints
            segment_capacity = constraints.track_capacity.get(train.current_segment, 1)
            trains_on_segment = sum(1 for t in sorted_trains[:sorted_trains.index(train)] 
                                  if t.current_segment == train.current_segment and 
                                  decisions.get(t.train_id) == DecisionType.PROCEED)
            
            if trains_on_segment < segment_capacity:
                decisions[train.train_id] = DecisionType.PROCEED
            else:
                decisions[train.train_id] = DecisionType.HALT
        
        return decisions

class RealTimeDecisionEngine:
    """
    Real-time decision engine that monitors section state and provides recommendations
    """
    
    def __init__(self, section_id: str):
        self.section_id = section_id
        self.optimizer = SectionControllerOptimizer(section_id)
        self.last_optimization = None
        self.active_decisions = {}
    
    def get_realtime_recommendations(
        self, 
        current_trains: List[TrainState],
        section_constraints: SectionConstraints,
        disruption_event: Optional[str] = None
    ) -> Dict:
        """
        Get real-time recommendations for section controller
        """
        
        # Re-optimize if significant change or disruption
        should_reoptimize = (
            self.last_optimization is None or
            disruption_event is not None or
            self._significant_state_change(current_trains)
        )
        
        if should_reoptimize:
            objectives = OptimizationObjectives()
            decisions = self.optimizer.optimize_train_precedence(
                current_trains, section_constraints, objectives
            )
            self.active_decisions = decisions
            self.last_optimization = datetime.now()
        
        # Generate human-readable recommendations
        recommendations = self._generate_recommendations(current_trains, self.active_decisions)
        
        # Calculate expected outcomes
        performance_metrics = self._calculate_performance_metrics(current_trains, self.active_decisions)
        
        return {
            'timestamp': datetime.now(),
            'decisions': self.active_decisions,
            'recommendations': recommendations,
            'performance_metrics': performance_metrics,
            'next_review_time': datetime.now() + timedelta(minutes=5)
        }
    
    def _significant_state_change(self, current_trains: List[TrainState]) -> bool:
        """Check if there's been a significant change requiring re-optimization"""
        
        # Check if new trains have appeared or disappeared
        current_train_ids = {train.train_id for train in current_trains}
        previous_train_ids = set(self.active_decisions.keys())
        
        if current_train_ids != previous_train_ids:
            return True
        
        # Check if any high-priority train is significantly delayed
        for train in current_trains:
            if train.priority <= 2 and train.schedule_adherence > 10:  # High priority, >10 min late
                return True
        
        return False
    
    def _generate_recommendations(self, trains: List[TrainState], decisions: Dict) -> List[Dict]:
        """Generate human-readable recommendations"""
        
        recommendations = []
        
        for train in trains:
            decision = decisions.get(train.train_id, DecisionType.HALT)
            
            if decision == DecisionType.PROCEED:
                recommendations.append({
                    'train_id': train.train_id,
                    'action': 'PROCEED',
                    'explanation': f'Train {train.train_id} can proceed. Priority {train.priority}, on schedule.',
                    'confidence': 0.95,
                    'expected_crossing_time': datetime.now() + timedelta(minutes=2)
                })
            
            elif decision == DecisionType.HALT:
                recommendations.append({
                    'train_id': train.train_id,
                    'action': 'HALT',
                    'explanation': f'Hold Train {train.train_id} due to capacity/safety constraints.',
                    'confidence': 0.90,
                    'expected_wait_time': timedelta(minutes=5)
                })
            
            elif decision == DecisionType.REROUTE:
                recommendations.append({
                    'train_id': train.train_id,
                    'action': 'REROUTE',
                    'explanation': f'Reroute Train {train.train_id} to alternative platform/track.',
                    'confidence': 0.85,
                    'alternative_options': ['Platform 2', 'Platform 3']
                })
        
        return recommendations
    
    def _calculate_performance_metrics(self, trains: List[TrainState], decisions: Dict) -> Dict:
        """Calculate expected performance metrics"""
        
        proceeding_trains = sum(1 for decision in decisions.values() if decision == DecisionType.PROCEED)
        total_trains = len(trains)
        
        # Calculate expected throughput
        expected_throughput = proceeding_trains  # trains per decision cycle
        
        # Calculate expected delays
        halted_trains = [train for train in trains if decisions.get(train.train_id) == DecisionType.HALT]
        expected_delay = sum(5 for _ in halted_trains)  # 5 min delay per halted train
        
        # Calculate priority adherence
        high_priority_proceeding = sum(
            1 for train in trains 
            if train.priority <= 2 and decisions.get(train.train_id) == DecisionType.PROCEED
        )
        high_priority_total = sum(1 for train in trains if train.priority <= 2)
        priority_adherence = high_priority_proceeding / max(high_priority_total, 1)
        
        return {
            'expected_throughput': expected_throughput,
            'expected_avg_delay_minutes': expected_delay / max(total_trains, 1),
            'priority_adherence_percent': priority_adherence * 100,
            'capacity_utilization_percent': (proceeding_trains / max(total_trains, 1)) * 100
        }
