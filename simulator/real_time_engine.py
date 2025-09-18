"""
Real-time Railway Simulation Engine
Implements comprehensive simulation with disruption handling, dynamic re-optimization,
and emergency event modeling for the railway optimization system.
"""

import asyncio
import simpy
import numpy as np
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from django.utils import timezone
from django.db import transaction
from django.core.cache import cache
import logging

from core.models import (
    Section, Train, Segment, Signal, Platform, TrainEvent, 
    ResourceLock, KPIRecord, EmergencyEvent, WeatherCondition
)
from decision.engines.ai_traffic_controller import AITrafficController
from decision.engines.ilp_engine_enhanced import AdvancedILPEngine

logger = logging.getLogger(__name__)

class RealTimeSimulationEngine:
    """
    Comprehensive real-time simulation engine for railway operations
    """
    
    def __init__(self, section, config=None):
        self.section = section
        self.config = config or self._default_config()
        
        # Simulation environment
        self.env = simpy.Environment()
        self.simulation_time = 0
        self.real_start_time = time.time()
        
        # Core components
        self.ai_controller = AITrafficController(section)
        self.ilp_engine = AdvancedILPEngine(section)
        
        # Simulation state
        self.trains = {}           # train_id -> Train object
        self.segments = {}         # segment_id -> SimPy Resource
        self.signals = {}          # signal_id -> Signal state
        self.platforms = {}        # platform_id -> Platform state
        
        # Real-time tracking
        self.train_positions = {}  # train_id -> position data
        self.active_disruptions = []
        self.emergency_events = []
        self.weather_conditions = []
        
        # Performance monitoring
        self.kpi_tracker = KPITracker()
        self.event_log = []
        
        # Simulation control
        self.is_running = False
        self.should_stop = False
        self.update_callbacks = []
        
        # Re-optimization control
        self.last_optimization = time.time()
        self.optimization_interval = self.config['optimization_interval_seconds']
        
        logger.info(f"Initialized Real-time Simulation Engine for {section.name}")
    
    def _default_config(self):
        """Default simulation configuration"""
        return {
            'simulation_speed': 1.0,           # Real-time multiplier
            'optimization_interval_seconds': 30,  # Re-optimize every 30 seconds
            'disruption_probability': 0.02,    # 2% chance per minute
            'weather_change_probability': 0.01, # 1% chance per minute
            'emergency_probability': 0.005,    # 0.5% chance per minute
            'real_time_updates': True,
            'enable_ai_decisions': True,
            'max_reoptimization_time': 5.0,   # Max 5 seconds for re-optimization
            'bottleneck_monitoring': True,
            'predictive_analysis': True
        }
    
    async def start_simulation(self, trains, simulation_duration_hours=8):
        """Start real-time simulation with given trains"""
        logger.info(f"Starting real-time simulation with {len(trains)} trains for {simulation_duration_hours}h")
        
        # Initialize simulation
        await self._initialize_simulation(trains)
        
        # Set simulation end time
        simulation_end = simulation_duration_hours * 60  # Convert to minutes
        
        # Start main processes
        processes = [
            self.env.process(self._simulation_controller(simulation_end)),
            self.env.process(self._real_time_optimizer()),
            self.env.process(self._disruption_generator()),
            self.env.process(self._weather_simulator()),
            self.env.process(self._emergency_event_generator()),
            self.env.process(self._kpi_monitor()),
            self.env.process(self._train_movement_monitor())
        ]
        
        # Add train processes
        for train in trains:
            processes.append(self.env.process(self._simulate_train(train)))
        
        self.is_running = True
        
        try:
            # Run simulation
            await asyncio.gather(*[self._run_simpy_process(p) for p in processes])
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.is_running = False
            await self._cleanup_simulation()
        
        return self._generate_simulation_report()
    
    async def _initialize_simulation(self, trains):
        """Initialize simulation environment and resources"""
        logger.info("Initializing simulation environment...")
        
        # Initialize segments
        segments = Segment.objects.filter(section=self.section)
        for segment in segments:
            self.segments[segment.id] = {
                'resource': simpy.Resource(self.env, capacity=segment.max_occupancy),
                'segment': segment,
                'occupancy': 0,
                'current_trains': set(),
                'average_speed': segment.effective_max_speed,
                'blocked': False,
                'disruption_end_time': None
            }
        
        # Initialize signals
        signals = Signal.objects.filter(segment__section=self.section)
        for signal in signals:
            self.signals[signal.id] = {
                'state': signal.current_state,
                'controlled_by': None,
                'next_change_time': None,
                'signal': signal
            }
        
        # Initialize platforms
        platforms = Platform.objects.filter(segment__section=self.section)
        for platform in platforms:
            self.platforms[platform.id] = {
                'resource': simpy.Resource(self.env, capacity=1),
                'platform': platform,
                'occupying_train': None,
                'reservation_time': None
            }
        
        # Initialize trains
        for train in trains:
            self.trains[train.id] = train
            self.train_positions[train.id] = {
                'current_segment': None,
                'position_in_segment': 0.0,  # 0.0 to 1.0
                'speed': train.current_speed,
                'status': 'pending',
                'delay_minutes': train.current_delay_minutes,
                'next_decision_time': self.env.now
            }
        
        # Initialize weather
        current_weather = WeatherCondition.objects.filter(section=self.section).first()
        if current_weather:
            self.weather_conditions.append(current_weather)
        
        logger.info("Simulation environment initialized successfully")
    
    async def _simulation_controller(self, simulation_end):
        """Main simulation controller process"""
        logger.info(f"Starting simulation controller (duration: {simulation_end} minutes)")
        
        while self.env.now < simulation_end and not self.should_stop:
            current_time = self.env.now
            
            # Update simulation time
            self.simulation_time = current_time
            
            # Check for trains requiring immediate decisions
            await self._handle_immediate_decisions()
            
            # Update KPIs
            await self._update_kpis()
            
            # Send real-time updates
            if self.config['real_time_updates']:
                await self._send_real_time_updates()
            
            # Wait before next iteration (1 minute simulation time)
            yield self.env.timeout(1)
        
        logger.info("Simulation controller completed")
    
    async def _real_time_optimizer(self):
        """Real-time optimization process"""
        while not self.should_stop:
            try:
                current_time = time.time()
                
                # Check if optimization is due
                if (current_time - self.last_optimization) >= self.optimization_interval:
                    await self._perform_reoptimization()
                    self.last_optimization = current_time
                
                # Wait before next check
                yield self.env.timeout(5)  # Check every 5 simulation minutes
                
            except Exception as e:
                logger.error(f"Real-time optimization error: {e}")
                yield self.env.timeout(10)  # Wait longer on error
    
    async def _perform_reoptimization(self):
        """Perform real-time re-optimization"""
        logger.info("Performing real-time re-optimization...")
        
        start_time = time.time()
        
        try:
            # Get current active trains
            active_trains = [
                train for train in self.trains.values()
                if self.train_positions[train.id]['status'] in ['running', 'pending', 'delayed']
            ]
            
            if not active_trains:
                return
            
            # Get current conditions
            current_weather = self.weather_conditions[-1] if self.weather_conditions else None
            current_emergencies = [e for e in self.emergency_events if e.is_active]
            
            # Make AI-powered decision
            if self.config['enable_ai_decisions']:
                decision = self.ai_controller.make_intelligent_decision(
                    active_trains, current_weather, current_emergencies
                )
            else:
                # Fallback to ILP only
                decision = self.ilp_engine.optimize_comprehensive_schedule(
                    active_trains, current_emergencies
                )
            
            # Apply decisions
            await self._apply_optimization_decisions(decision, active_trains)
            
            optimization_time = time.time() - start_time
            
            # Log optimization event
            self.event_log.append({
                'timestamp': self.env.now,
                'type': 'reoptimization',
                'duration': optimization_time,
                'trains_affected': len(active_trains),
                'solution_quality': decision.get('status', 'unknown')
            })
            
            logger.info(f"Re-optimization completed in {optimization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Re-optimization failed: {e}")
    
    async def _apply_optimization_decisions(self, decision, trains):
        """Apply optimization decisions to running simulation"""
        if decision.get('status') not in ['optimal', 'feasible', 'heuristic_fallback']:
            logger.warning("Invalid optimization decision - not applying")
            return
        
        # Extract recommendations
        recommendations = decision.get('recommendations', [])
        
        for recommendation in recommendations:
            await self._execute_recommendation(recommendation)
        
        # Update train schedules if provided
        train_schedules = decision.get('trains', {})
        for train_id, schedule in train_schedules.items():
            if train_id in self.train_positions:
                await self._update_train_schedule(train_id, schedule)
    
    async def _execute_recommendation(self, recommendation):
        """Execute a specific recommendation"""
        rec_type = recommendation.get('type')
        priority = recommendation.get('priority', 'medium')
        
        if rec_type == 'delay_warning':
            train_id = recommendation.get('train_id')
            if train_id and train_id in self.train_positions:
                # Increase monitoring for this train
                self.train_positions[train_id]['monitoring_level'] = 'high'
        
        elif rec_type == 'speed_restriction':
            train_id = recommendation.get('train_id')
            if train_id and train_id in self.train_positions:
                # Apply speed restriction
                current_speed = self.train_positions[train_id]['speed']
                restricted_speed = current_speed * 0.8  # 20% reduction
                self.train_positions[train_id]['speed'] = restricted_speed
                
                logger.info(f"Applied speed restriction to train {train_id}")
        
        elif rec_type == 'bottleneck_alert':
            # Handle bottleneck situation
            await self._handle_bottleneck_alert(recommendation)
        
        elif rec_type == 'emergency_response':
            # Handle emergency response
            await self._handle_emergency_response(recommendation)
    
    async def _simulate_train(self, train):
        """Simulate individual train movement"""
        logger.debug(f"Starting simulation for train {train.train_id}")
        
        # Initial delay
        if train.current_delay_minutes > 0:
            yield self.env.timeout(train.current_delay_minutes)
            self.train_positions[train.id]['delay_minutes'] += train.current_delay_minutes
        
        # Move through segments
        segments = list(Segment.objects.filter(section=self.section).order_by('sequence'))
        
        for segment in segments:
            try:
                # Request segment access
                with self.segments[segment.id]['resource'].request() as req:
                    yield req
                    
                    # Update train position
                    self.train_positions[train.id]['current_segment'] = segment.id
                    self.train_positions[train.id]['status'] = 'running'
                    self.segments[segment.id]['current_trains'].add(train.id)
                    
                    # Calculate travel time
                    travel_time = await self._calculate_travel_time(train, segment)
                    
                    # Simulate movement through segment
                    async for progress in self._simulate_segment_movement(train, segment, travel_time):
                        self.train_positions[train.id]['position_in_segment'] = progress
                        
                        # Check for disruptions during movement
                        if await self._check_disruptions(train, segment):
                            break
                        
                        yield self.env.timeout(0.1)  # Small time step
                    
                    # Exit segment
                    self.segments[segment.id]['current_trains'].discard(train.id)
                    
                    # Log segment completion
                    self._log_train_event(train, 'segment_completed', {
                        'segment_id': segment.id,
                        'travel_time': travel_time,
                        'delay': self.train_positions[train.id]['delay_minutes']
                    })
            
            except Exception as e:
                logger.error(f"Error simulating train {train.train_id} in segment {segment.name}: {e}")
                self.train_positions[train.id]['status'] = 'error'
                break
        
        # Train completed journey
        self.train_positions[train.id]['status'] = 'completed'
        logger.info(f"Train {train.train_id} completed journey with {self.train_positions[train.id]['delay_minutes']:.1f}min delay")
    
    async def _calculate_travel_time(self, train, segment):
        """Calculate travel time considering current conditions"""
        base_time = segment.length_km / train.max_speed * 60  # minutes
        
        # Apply weather factor
        if self.weather_conditions:
            weather_factor = self.weather_conditions[-1].visibility_factor
            base_time /= weather_factor
        
        # Apply segment conditions
        if self.segments[segment.id]['blocked']:
            base_time *= 2  # Double time if blocked
        
        # Apply current speed
        current_speed = self.train_positions[train.id]['speed']
        speed_factor = train.max_speed / current_speed if current_speed > 0 else 2
        base_time *= speed_factor
        
        return max(1, base_time)  # Minimum 1 minute
    
    async def _simulate_segment_movement(self, train, segment, travel_time):
        """Simulate movement through a segment with progress updates"""
        steps = int(travel_time * 10)  # 0.1 minute steps
        
        for step in range(steps):
            progress = step / steps
            yield progress
            
            # Random delay possibility
            if np.random.random() < 0.01:  # 1% chance per step
                additional_delay = np.random.exponential(2)  # 2 minute average
                self.train_positions[train.id]['delay_minutes'] += additional_delay
                logger.debug(f"Train {train.train_id} experienced {additional_delay:.1f}min delay")
        
        yield 1.0  # Complete
    
    async def _disruption_generator(self):
        """Generate random disruptions"""
        while not self.should_stop:
            try:
                # Check for disruption
                if np.random.random() < self.config['disruption_probability']:
                    await self._generate_disruption()
                
                yield self.env.timeout(1)  # Check every minute
                
            except Exception as e:
                logger.error(f"Disruption generator error: {e}")
                yield self.env.timeout(5)
    
    async def _generate_disruption(self):
        """Generate a random disruption"""
        disruption_types = ['signal_failure', 'track_obstruction', 'equipment_failure', 'weather_delay']
        disruption_type = np.random.choice(disruption_types)
        
        # Select random segment
        segment_ids = list(self.segments.keys())
        if not segment_ids:
            return
        
        affected_segment_id = np.random.choice(segment_ids)
        
        # Create disruption
        duration = np.random.exponential(15)  # 15 minute average
        
        disruption = {
            'type': disruption_type,
            'affected_segment': affected_segment_id,
            'start_time': self.env.now,
            'duration': duration,
            'severity': np.random.choice(['minor', 'moderate', 'major'], p=[0.6, 0.3, 0.1])
        }
        
        self.active_disruptions.append(disruption)
        
        # Apply disruption effects
        if disruption['severity'] == 'major':
            self.segments[affected_segment_id]['blocked'] = True
            self.segments[affected_segment_id]['disruption_end_time'] = self.env.now + duration
        
        logger.warning(f"Disruption: {disruption_type} in segment {affected_segment_id} for {duration:.1f}min")
        
        # Schedule disruption end
        self.env.process(self._end_disruption(disruption))
    
    async def _end_disruption(self, disruption):
        """End a disruption after its duration"""
        yield self.env.timeout(disruption['duration'])
        
        affected_segment_id = disruption['affected_segment']
        self.segments[affected_segment_id]['blocked'] = False
        self.segments[affected_segment_id]['disruption_end_time'] = None
        
        # Remove from active disruptions
        if disruption in self.active_disruptions:
            self.active_disruptions.remove(disruption)
        
        logger.info(f"Disruption ended in segment {affected_segment_id}")
    
    async def _weather_simulator(self):
        """Simulate weather changes"""
        while not self.should_stop:
            try:
                if np.random.random() < self.config['weather_change_probability']:
                    await self._change_weather()
                
                yield self.env.timeout(1)  # Check every minute
                
            except Exception as e:
                logger.error(f"Weather simulator error: {e}")
                yield self.env.timeout(5)
    
    async def _change_weather(self):
        """Change weather conditions"""
        conditions = ['clear', 'light_rain', 'heavy_rain', 'fog', 'storm']
        visibility_factors = [1.0, 0.9, 0.7, 0.5, 0.4]
        
        condition = np.random.choice(conditions)
        visibility_factor = visibility_factors[conditions.index(condition)]
        
        weather = WeatherCondition(
            section=self.section,
            condition=condition,
            visibility_factor=visibility_factor,
            timestamp=timezone.now(),
            temperature=np.random.normal(25, 10),  # 25°C ± 10°C
            wind_speed=np.random.exponential(10),  # 10 km/h average
            precipitation_mm=np.random.exponential(5) if 'rain' in condition else 0
        )
        
        self.weather_conditions.append(weather)
        
        # Update section weather impact
        self.section.weather_impact_factor = visibility_factor
        
        logger.info(f"Weather changed to {condition} (visibility: {visibility_factor:.1f})")
    
    async def _emergency_event_generator(self):
        """Generate emergency events"""
        while not self.should_stop:
            try:
                if np.random.random() < self.config['emergency_probability']:
                    await self._generate_emergency_event()
                
                yield self.env.timeout(1)  # Check every minute
                
            except Exception as e:
                logger.error(f"Emergency generator error: {e}")
                yield self.env.timeout(5)
    
    async def _generate_emergency_event(self):
        """Generate an emergency event"""
        event_types = ['medical_emergency', 'security_alert', 'technical_failure', 'passenger_incident']
        event_type = np.random.choice(event_types)
        
        severity = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.08, 0.02])
        duration = np.random.exponential(20) if severity >= 3 else np.random.exponential(10)
        
        emergency = EmergencyEvent(
            section=self.section,
            event_type=event_type,
            severity=severity,
            description=f"Emergency: {event_type} (severity {severity})",
            start_time=timezone.now(),
            estimated_duration_minutes=duration,
            status='active'
        )
        
        self.emergency_events.append(emergency)
        
        logger.warning(f"Emergency event: {event_type} (severity {severity}) for {duration:.1f}min")
        
        # Schedule event resolution
        self.env.process(self._resolve_emergency_event(emergency))
    
    async def _resolve_emergency_event(self, emergency):
        """Resolve an emergency event"""
        yield self.env.timeout(emergency.estimated_duration_minutes)
        
        emergency.status = 'resolved'
        emergency.end_time = timezone.now()
        
        logger.info(f"Emergency event resolved: {emergency.event_type}")
    
    async def _kpi_monitor(self):
        """Monitor and update KPIs"""
        while not self.should_stop:
            try:
                await self._calculate_current_kpis()
                yield self.env.timeout(5)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"KPI monitor error: {e}")
                yield self.env.timeout(10)
    
    async def _calculate_current_kpis(self):
        """Calculate current KPIs"""
        active_trains = [
            train_id for train_id, pos in self.train_positions.items()
            if pos['status'] in ['running', 'pending', 'delayed']
        ]
        
        completed_trains = [
            train_id for train_id, pos in self.train_positions.items()
            if pos['status'] == 'completed'
        ]
        
        # Calculate throughput
        elapsed_hours = self.env.now / 60 if self.env.now > 0 else 0.1
        throughput = len(completed_trains) / elapsed_hours if elapsed_hours > 0 else 0
        
        # Calculate average delay
        total_delay = sum([
            pos['delay_minutes'] for pos in self.train_positions.values()
        ])
        avg_delay = total_delay / len(self.train_positions) if self.train_positions else 0
        
        # Calculate punctuality (trains with <= 5 min delay)
        on_time_trains = len([
            pos for pos in self.train_positions.values()
            if pos['delay_minutes'] <= 5
        ])
        punctuality = (on_time_trains / len(self.train_positions) * 100) if self.train_positions else 100
        
        # Calculate utilization
        utilization = len(active_trains) / self.section.capacity * 100 if self.section.capacity > 0 else 0
        
        kpis = {
            'timestamp': self.env.now,
            'throughput_per_hour': throughput,
            'average_delay_minutes': avg_delay,
            'punctuality_percent': punctuality,
            'utilization_percent': utilization,
            'active_trains': len(active_trains),
            'completed_trains': len(completed_trains),
            'active_disruptions': len(self.active_disruptions),
            'active_emergencies': len([e for e in self.emergency_events if e.status == 'active'])
        }
        
        self.kpi_tracker.add_record(kpis)
    
    async def _train_movement_monitor(self):
        """Monitor train movements for anomalies"""
        while not self.should_stop:
            try:
                for train_id, position in self.train_positions.items():
                    if position['status'] == 'running':
                        await self._check_train_anomalies(train_id, position)
                
                yield self.env.timeout(1)  # Check every minute
                
            except Exception as e:
                logger.error(f"Train movement monitor error: {e}")
                yield self.env.timeout(5)
    
    async def _check_train_anomalies(self, train_id, position):
        """Check for train movement anomalies"""
        # Check for excessive delay
        if position['delay_minutes'] > 30:
            logger.warning(f"Train {train_id} has excessive delay: {position['delay_minutes']:.1f}min")
            # Could trigger immediate re-optimization
        
        # Check for stopped trains
        if position['speed'] < 5 and position['status'] == 'running':
            logger.warning(f"Train {train_id} appears to be stopped")
        
        # Check for trains in same segment too long
        if position['current_segment']:
            segment_data = self.segments[position['current_segment']]
            if len(segment_data['current_trains']) > segment_data['segment'].max_occupancy:
                logger.warning(f"Segment {position['current_segment']} overcapacity")
    
    def _log_train_event(self, train, event_type, data=None):
        """Log a train event"""
        event = {
            'timestamp': self.env.now,
            'train_id': train.train_id,
            'event_type': event_type,
            'data': data or {}
        }
        
        self.event_log.append(event)
    
    async def _run_simpy_process(self, process):
        """Run a SimPy process in async context"""
        try:
            await asyncio.sleep(0)  # Allow async context
            self.env.run(until=process)
        except Exception as e:
            logger.error(f"SimPy process error: {e}")
    
    def _generate_simulation_report(self):
        """Generate comprehensive simulation report"""
        total_trains = len(self.train_positions)
        completed_trains = len([
            pos for pos in self.train_positions.values()
            if pos['status'] == 'completed'
        ])
        
        completion_rate = (completed_trains / total_trains * 100) if total_trains > 0 else 0
        
        avg_delay = np.mean([
            pos['delay_minutes'] for pos in self.train_positions.values()
        ]) if self.train_positions else 0
        
        total_disruptions = len(self.event_log) - len([
            e for e in self.event_log if e['event_type'] == 'segment_completed'
        ])
        
        report = {
            'simulation_summary': {
                'total_trains': total_trains,
                'completed_trains': completed_trains,
                'completion_rate_percent': completion_rate,
                'average_delay_minutes': avg_delay,
                'total_disruptions': total_disruptions,
                'total_emergencies': len(self.emergency_events),
                'simulation_duration_minutes': self.simulation_time
            },
            'performance_metrics': self.kpi_tracker.get_summary(),
            'disruption_analysis': self._analyze_disruptions(),
            'optimization_events': [
                e for e in self.event_log if e['event_type'] == 'reoptimization'
            ],
            'train_details': {
                train_id: {
                    'final_delay': pos['delay_minutes'],
                    'status': pos['status'],
                    'completion_time': pos.get('completion_time', None)
                }
                for train_id, pos in self.train_positions.items()
            }
        }
        
        return report
    
    def _analyze_disruptions(self):
        """Analyze disruption patterns and impact"""
        if not self.active_disruptions:
            return {'total': 0, 'by_type': {}, 'avg_duration': 0}
        
        disruption_types = {}
        total_duration = 0
        
        for disruption in self.active_disruptions:
            dtype = disruption['type']
            disruption_types[dtype] = disruption_types.get(dtype, 0) + 1
            total_duration += disruption['duration']
        
        return {
            'total': len(self.active_disruptions),
            'by_type': disruption_types,
            'avg_duration': total_duration / len(self.active_disruptions),
            'severity_distribution': {
                severity: len([d for d in self.active_disruptions if d['severity'] == severity])
                for severity in ['minor', 'moderate', 'major']
            }
        }
    
    async def _cleanup_simulation(self):
        """Clean up simulation resources"""
        logger.info("Cleaning up simulation...")
        
        # Save final state
        await self._save_simulation_state()
        
        # Clear resources
        self.segments.clear()
        self.signals.clear()
        self.platforms.clear()
        self.train_positions.clear()
        
        logger.info("Simulation cleanup completed")
    
    async def _save_simulation_state(self):
        """Save simulation state to database"""
        try:
            # Save KPI records
            for kpi_data in self.kpi_tracker.get_all_records():
                KPIRecord.objects.create(
                    section=self.section,
                    timestamp=timezone.now(),
                    metric_name='simulation_kpi',
                    value=json.dumps(kpi_data)
                )
            
            # Save emergency events
            for emergency in self.emergency_events:
                if hasattr(emergency, 'save'):
                    emergency.save()
            
            logger.info("Simulation state saved to database")
            
        except Exception as e:
            logger.error(f"Error saving simulation state: {e}")
    
    # Placeholder methods for unimplemented functionality
    async def _handle_immediate_decisions(self):
        pass
    
    async def _update_kpis(self):
        pass
    
    async def _send_real_time_updates(self):
        pass
    
    async def _update_train_schedule(self, train_id, schedule):
        pass
    
    async def _handle_bottleneck_alert(self, recommendation):
        pass
    
    async def _handle_emergency_response(self, recommendation):
        pass
    
    async def _check_disruptions(self, train, segment):
        return False


class KPITracker:
    """Track KPIs during simulation"""
    
    def __init__(self):
        self.records = []
    
    def add_record(self, kpi_data):
        """Add a KPI record"""
        self.records.append(kpi_data.copy())
    
    def get_all_records(self):
        """Get all KPI records"""
        return self.records
    
    def get_summary(self):
        """Get KPI summary"""
        if not self.records:
            return {}
        
        latest = self.records[-1]
        
        # Calculate trends
        throughput_trend = 'stable'
        if len(self.records) >= 2:
            if latest['throughput_per_hour'] > self.records[-2]['throughput_per_hour']:
                throughput_trend = 'increasing'
            elif latest['throughput_per_hour'] < self.records[-2]['throughput_per_hour']:
                throughput_trend = 'decreasing'
        
        return {
            'current_metrics': latest,
            'trends': {
                'throughput': throughput_trend
            },
            'peak_values': {
                'max_throughput': max([r['throughput_per_hour'] for r in self.records]),
                'max_utilization': max([r['utilization_percent'] for r in self.records]),
                'worst_delay': max([r['average_delay_minutes'] for r in self.records])
            }
        }