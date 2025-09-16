import asyncio
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import simpy
import numpy as np
from django.utils import timezone
from django.db import transaction
from django.core.cache import cache
from core.models import (
    Train, Segment, Signal, Platform, TrainEvent, ResourceLock, 
    KPIRecord, EmergencyEvent, WeatherCondition, Section
)
import logging

logger = logging.getLogger(__name__)

class RealTimeRailwaySimulator:
    """
    Advanced railway simulator with real-time capabilities, 
    realistic scenarios, and AI-powered decision making
    """
    
    def __init__(self, section, simulation_speed=1.0, real_time_mode=False):
        self.section = section
        self.simulation_speed = simulation_speed  # 1.0 = real time, 2.0 = 2x speed
        self.real_time_mode = real_time_mode
        
        # Simulation environment
        self.env = simpy.Environment()
        self.current_time = timezone.now()
        
        # Resources and state
        self.segments = {}
        self.signals = {}
        self.platforms = {}
        self.trains = {}
        self.active_events = []
        
        # Real-time tracking
        self.train_positions = {}  # train_id -> current position
        self.train_speeds = {}     # train_id -> current speed
        self.signal_states = {}    # signal_id -> current state
        self.platform_occupancy = {}  # platform_id -> occupying train
        
        # Performance tracking
        self.kpi_history = []
        self.event_history = []
        
        # Simulation control
        self.is_running = False
        self.simulation_thread = None
        self.update_callbacks = []  # For WebSocket updates
        
        # Emergency and weather simulation
        self.weather_generator = WeatherGenerator(section)
        self.emergency_generator = EmergencyGenerator(section)
        
        logger.info(f"Initialized RealTimeRailwaySimulator for {section.name}")
    
    def initialize(self):
        """Initialize simulation resources and state"""
        logger.info("Initializing simulation resources...")
        
        # Initialize segments
        segments = Segment.objects.filter(section=self.section)
        for segment in segments:
            self.segments[segment.id] = simpy.Resource(
                self.env, capacity=segment.max_occupancy
            )
            logger.debug(f"Initialized segment {segment.name}")
        
        # Initialize signals
        signals = Signal.objects.filter(segment__section=self.section)
        for signal in signals:
            self.signals[signal.id] = {
                'resource': simpy.Resource(self.env, capacity=1),
                'state': signal.current_state,
                'controlled_by': None
            }
            self.signal_states[signal.id] = signal.current_state
        
        # Initialize platforms
        platforms = Platform.objects.filter(segment__section=self.section)
        for platform in platforms:
            self.platforms[platform.id] = simpy.Resource(self.env, capacity=1)
            self.platform_occupancy[platform.id] = None
        
        # Start background processes
        if self.real_time_mode:
            self.env.process(self.weather_simulation())
            self.env.process(self.emergency_simulation())
            self.env.process(self.kpi_updater())
            self.env.process(self.real_time_sync())
        
        logger.info("Simulation initialization complete")
    
    def add_train(self, train, entry_time=None, route_segments=None):
        """Add a train to the simulation with realistic behavior"""
        if entry_time is None:
            entry_time = 0
        
        if route_segments is None:
            route_segments = list(Segment.objects.filter(
                section=self.section
            ).order_by('sequence'))
        
        train_data = {
            'train': train,
            'entry_time': entry_time,
            'route_segments': route_segments,
            'current_segment_index': 0,
            'position_in_segment': 0.0,
            'current_speed': 0,
            'status': 'approaching',
            'delays': [],
            'events': [],
            'fuel_level': train.fuel_level_percent,
            'passenger_load': train.passenger_count
        }
        
        self.trains[train.id] = train_data
        self.train_positions[train.id] = 0.0
        self.train_speeds[train.id] = 0
        
        # Start train process
        self.env.process(self.run_train(train_data))
        logger.info(f"Added train {train.train_id} to simulation")
        
        return train_data
    
    def run_train(self, train_data):
        """Simulate realistic train movement with AI decision points"""
        train = train_data['train']
        route = train_data['route_segments']
        
        try:
            # Wait for entry time
            yield self.env.timeout(train_data['entry_time'])
            
            # Update train status
            train.current_status = 'running'
            train.save()
            
            self.record_event(train, 'enter_section', self.env.now)
            train_data['status'] = 'in_section'
            
            # Process each segment in route
            for segment_index, segment in enumerate(route):
                train_data['current_segment_index'] = segment_index
                
                # Request segment resource
                with self.segments[segment.id].request() as segment_req:
                    yield segment_req
                    
                    # Update current segment
                    train.current_segment = segment
                    train.save()
                    
                    # Record segment entry
                    self.record_event(train, 'enter_segment', self.env.now, 
                                    segment=segment)
                    
                    # Simulate realistic movement through segment
                    yield from self.simulate_segment_traversal(train_data, segment)
                    
                    # Record segment clearance
                    self.record_event(train, 'clear_segment', self.env.now, 
                                    segment=segment)
            
            # Train completed route
            train.current_status = 'completed'
            train.actual_departure = timezone.now()
            train.save()
            
            self.record_event(train, 'clear_section', self.env.now)
            self.calculate_performance_metrics(train_data)
            
            logger.info(f"Train {train.train_id} completed journey")
            
        except Exception as e:
            logger.error(f"Error in train simulation for {train.train_id}: {e}")
            train.current_status = 'emergency'
            train.save()
    
    def simulate_segment_traversal(self, train_data, segment):
        """Simulate realistic movement through a segment with dynamic factors"""
        train = train_data['train']
        
        # Calculate base traversal time
        base_speed = min(train.max_speed, segment.effective_max_speed)
        distance_km = segment.length_km
        
        # Apply dynamic factors
        weather_factor = self.section.weather_impact_factor
        traffic_factor = self.calculate_traffic_factor(segment)
        
        effective_speed = base_speed * weather_factor * traffic_factor
        base_time_hours = distance_km / max(effective_speed, 10)  # Minimum 10 km/h
        
        # Convert to simulation time
        traversal_time = base_time_hours * 60  # Convert to minutes
        
        # Add realistic variations and events
        segments_count = max(1, int(distance_km / 5))  # 5km segments for granular simulation
        time_per_subsegment = traversal_time / segments_count
        
        for i in range(segments_count):
            # Check for signals in this subsegment
            if i < len(segment.signals.all()):
                signal = segment.signals.all()[i]
                yield from self.handle_signal_interaction(train_data, signal)
            
            # Simulate speed variations
            speed_variation = random.uniform(0.8, 1.2)
            actual_time = time_per_subsegment * speed_variation
            
            # Update position and speed
            progress = (i + 1) / segments_count
            train_data['position_in_segment'] = progress * distance_km
            current_speed = int(effective_speed * speed_variation)
            train_data['current_speed'] = current_speed
            
            # Update real-time tracking
            self.train_speeds[train.id] = current_speed
            self.train_positions[train.id] = (
                segment.sequence * 10 + progress * 10  # Simplified position
            )
            
            # Check for random events
            if random.random() < 0.02:  # 2% chance per subsegment
                yield from self.handle_random_event(train_data, segment)
            
            # Wait for subsegment traversal
            yield self.env.timeout(actual_time)
            
            # Send real-time updates
            self.broadcast_train_update(train_data)
    
    def handle_signal_interaction(self, train_data, signal):
        """Handle realistic signal interactions"""
        train = train_data['train']
        
        # Check signal state
        current_state = self.signal_states.get(signal.id, 'green')
        
        if current_state == 'red':
            # Train must stop
            self.record_event(train, 'stop', self.env.now, signal=signal)
            train_data['current_speed'] = 0
            
            # Wait for signal to clear (realistic timing)
            wait_time = random.uniform(2, 8)  # 2-8 minutes
            yield self.env.timeout(wait_time)
            
            # Update signal state (simplified logic)
            self.signal_states[signal.id] = 'green'
            signal.current_state = 'green'
            signal.save()
            
        elif current_state == 'yellow':
            # Reduce speed
            train_data['current_speed'] = int(train_data['current_speed'] * 0.6)
            yield self.env.timeout(1)  # Brief caution period
        
        # Record signal passage
        self.record_event(train, 'pass_signal', self.env.now, signal=signal)
    
    def handle_random_event(self, train_data, segment):
        """Handle random operational events"""
        train = train_data['train']
        
        event_types = ['minor_delay', 'signal_check', 'passenger_delay', 'technical_check']
        event_type = random.choice(event_types)
        
        if event_type == 'minor_delay':
            delay_time = random.uniform(1, 3)
            self.record_event(train, 'stop', self.env.now, segment=segment, 
                            details={'reason': 'Minor operational delay'})
            yield self.env.timeout(delay_time)
            
        elif event_type == 'signal_check':
            delay_time = random.uniform(0.5, 1.5)
            yield self.env.timeout(delay_time)
            
        elif event_type == 'passenger_delay' and train.train_type in ['local', 'express']:
            delay_time = random.uniform(0.5, 2)
            yield self.env.timeout(delay_time)
        
        elif event_type == 'technical_check':
            delay_time = random.uniform(2, 5)
            self.record_event(train, 'maintenance_halt', self.env.now, 
                            segment=segment)
            yield self.env.timeout(delay_time)
    
    def calculate_traffic_factor(self, segment):
        """Calculate traffic congestion factor for speed adjustment"""
        current_trains = 0
        for train_id, train_data in self.trains.items():
            if (train_data.get('current_segment_index', -1) >= 0 and 
                train_data['route_segments'][train_data['current_segment_index']].id == segment.id):
                current_trains += 1
        
        congestion_factor = 1.0 - (current_trains * 0.1)  # 10% reduction per extra train
        return max(0.3, congestion_factor)  # Minimum 30% speed
    
    def weather_simulation(self):
        """Simulate dynamic weather changes"""
        while True:
            # Update weather every 30 minutes (simulation time)
            yield self.env.timeout(30)
            
            # Realistic weather progression
            current_weather = self.section.current_weather
            weather_transitions = {
                'clear': ['clear', 'rain', 'fog'],
                'rain': ['rain', 'clear', 'storm'],
                'fog': ['fog', 'clear'],
                'storm': ['storm', 'rain', 'clear']
            }
            
            if random.random() < 0.3:  # 30% chance of weather change
                new_weather = random.choice(weather_transitions[current_weather])
                self.section.current_weather = new_weather
                self.section.save()
                
                # Record weather event
                WeatherCondition.objects.create(
                    section=self.section,
                    temperature=random.uniform(15, 35),
                    humidity_percent=random.uniform(30, 90),
                    wind_speed_kmh=random.uniform(0, 30),
                    weather_type=new_weather,
                    visibility_km=random.uniform(0.5, 10) if new_weather == 'fog' else 10
                )
                
                logger.info(f"Weather changed to {new_weather} in {self.section.name}")
    
    def emergency_simulation(self):
        """Simulate realistic emergency events"""
        while True:
            # Check for emergencies every hour
            yield self.env.timeout(60)
            
            if random.random() < 0.05:  # 5% chance per hour
                event_types = ['equipment_failure', 'medical', 'security', 'power_outage']
                event_type = random.choice(event_types)
                severity = random.choice(['low', 'medium', 'high'])
                
                # Create emergency event
                emergency = EmergencyEvent.objects.create(
                    event_type=event_type,
                    section=self.section,
                    severity=severity,
                    description=f"Simulated {event_type} event"
                )
                
                # Affect operations based on severity
                if severity == 'high':
                    # Stop all trains temporarily
                    for train_data in self.trains.values():
                        if train_data['status'] == 'in_section':
                            train_data['current_speed'] = 0
                    
                    # Wait for resolution
                    resolution_time = random.uniform(10, 30)  # 10-30 minutes
                    yield self.env.timeout(resolution_time)
                    
                    emergency.resolved_at = timezone.now()
                    emergency.save()
                
                logger.warning(f"Emergency event: {event_type} ({severity}) in {self.section.name}")
    
    def kpi_updater(self):
        """Update KPIs periodically"""
        while True:
            yield self.env.timeout(15)  # Update every 15 minutes
            
            # Calculate current KPIs
            active_trains = len([t for t in self.trains.values() 
                               if t['status'] == 'in_section'])
            
            avg_speed = np.mean([self.train_speeds.get(tid, 0) 
                               for tid in self.train_speeds]) if self.train_speeds else 0
            
            avg_delay = self.calculate_average_delay()
            punctuality = self.calculate_punctuality()
            utilization = (active_trains / self.section.capacity) * 100 if self.section.capacity > 0 else 0
            
            # Update section metrics
            self.section.current_throughput = active_trains * 4  # Approximate trains/hour
            self.section.avg_delay_minutes = avg_delay
            self.section.utilization_percent = min(100, utilization)
            self.section.save()
            
            # Create KPI record
            KPIRecord.objects.create(
                section=self.section,
                throughput_trains=int(self.section.current_throughput),
                avg_delay_minutes=avg_delay,
                punctuality_percent=punctuality,
                utilization_percent=utilization,
                avg_speed_kmh=avg_speed,
                fuel_efficiency=random.uniform(0.8, 1.2),  # Placeholder
                passenger_satisfaction=random.uniform(6, 9),
                safety_incidents=0,
                co2_emissions_kg=active_trains * random.uniform(50, 100),
                noise_level_db=random.uniform(60, 80)
            )
    
    def real_time_sync(self):
        """Synchronize with real-world time if in real-time mode"""
        if not self.real_time_mode:
            return
        
        while True:
            # Update every second
            yield self.env.timeout(1/60)  # 1 second in simulation time
            
            # Sync with real time
            real_time_delta = (timezone.now() - self.current_time).total_seconds()
            sim_time_delta = real_time_delta * self.simulation_speed
            
            if sim_time_delta > 0:
                self.current_time = timezone.now()
    
    def calculate_average_delay(self):
        """Calculate average delay across all trains"""
        delays = []
        for train_data in self.trains.values():
            if train_data.get('delays'):
                delays.extend(train_data['delays'])
        
        return np.mean(delays) if delays else 0
    
    def calculate_punctuality(self):
        """Calculate punctuality percentage"""
        on_time_trains = 0
        total_trains = 0
        
        for train_data in self.trains.values():
            if train_data['status'] in ['completed', 'in_section']:
                total_trains += 1
                avg_delay = np.mean(train_data.get('delays', [0]))
                if avg_delay <= 5:  # On time if delay <= 5 minutes
                    on_time_trains += 1
        
        return (on_time_trains / total_trains) * 100 if total_trains > 0 else 100
    
    def record_event(self, train, event_type, timestamp, **kwargs):
        """Record a simulation event with realistic details"""
        # Convert simulation time to real timestamp
        real_timestamp = timezone.now()
        
        # Get current position and speed
        position = self.train_positions.get(train.id, 0.0)
        speed = self.train_speeds.get(train.id, 0)
        
        # Calculate delay
        expected_time = timestamp  # Simplified
        actual_time = self.env.now
        delay = max(0, actual_time - expected_time)
        
        # Create event record
        TrainEvent.objects.create(
            train=train,
            event_type=event_type,
            timestamp=real_timestamp,
            segment=kwargs.get('segment'),
            signal=kwargs.get('signal'),
            platform=kwargs.get('platform'),
            details=kwargs.get('details', {}),
            speed_at_event=speed,
            position_km=position,
            delay_minutes=delay
        )
        
        # Update train's delay history
        if train.id in self.trains:
            self.trains[train.id]['delays'].append(delay)
        
        # Add to event history
        event_data = {
            'train_id': train.train_id,
            'event_type': event_type,
            'timestamp': real_timestamp,
            'position': position,
            'speed': speed,
            'delay': delay
        }
        self.event_history.append(event_data)
        
        # Broadcast real-time update
        self.broadcast_event_update(event_data)
        
        logger.debug(f"Event recorded: {train.train_id} - {event_type}")
    
    def calculate_performance_metrics(self, train_data):
        """Calculate comprehensive performance metrics for completed train"""
        train = train_data['train']
        
        # Calculate total journey time
        total_time = self.env.now - train_data['entry_time']
        
        # Calculate average speed
        total_distance = sum(seg.length_km for seg in train_data['route_segments'])
        avg_speed = total_distance / (total_time / 60) if total_time > 0 else 0
        
        # Update train record
        train.current_speed = 0
        train.current_position_km = total_distance
        train.save()
        
        # Log performance
        logger.info(f"Train {train.train_id} performance: "
                   f"Time: {total_time:.1f}min, Speed: {avg_speed:.1f}km/h")
    
    def broadcast_train_update(self, train_data):
        """Broadcast real-time train updates to connected clients"""
        update_data = {
            'type': 'train_update',
            'train_id': train_data['train'].train_id,
            'position': self.train_positions.get(train_data['train'].id, 0),
            'speed': self.train_speeds.get(train_data['train'].id, 0),
            'status': train_data['status'],
            'current_segment': train_data.get('current_segment_index', 0)
        }
        
        # Cache for WebSocket retrieval
        cache.set(f"train_update_{train_data['train'].id}", update_data, timeout=60)
        
        # Call registered callbacks
        for callback in self.update_callbacks:
            try:
                callback(update_data)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    def broadcast_event_update(self, event_data):
        """Broadcast event updates"""
        cache.set(f"event_update_{event_data['train_id']}", event_data, timeout=300)
    
    def start_simulation(self):
        """Start the simulation in a background thread"""
        if self.is_running:
            logger.warning("Simulation already running")
            return
        
        self.is_running = True
        
        def run_simulation():
            try:
                logger.info("Starting simulation thread")
                self.env.run(until=simpy.core.Infinity)
            except Exception as e:
                logger.error(f"Simulation error: {e}")
            finally:
                self.is_running = False
        
        self.simulation_thread = threading.Thread(target=run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("Real-time simulation started")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
        logger.info("Simulation stopped")
    
    def get_real_time_status(self):
        """Get current real-time status of the simulation"""
        return {
            'section': self.section.name,
            'is_running': self.is_running,
            'active_trains': len([t for t in self.trains.values() 
                                if t['status'] == 'in_section']),
            'current_time': self.env.now,
            'weather': self.section.current_weather,
            'throughput': self.section.current_throughput,
            'avg_delay': self.section.avg_delay_minutes,
            'utilization': self.section.utilization_percent,
            'train_positions': self.train_positions,
            'train_speeds': self.train_speeds,
            'signal_states': self.signal_states,
            'platform_occupancy': self.platform_occupancy
        }

class WeatherGenerator:
    """Generate realistic weather patterns"""
    def __init__(self, section):
        self.section = section
        self.weather_patterns = {
            'clear': {'duration_hours': (2, 8), 'next': ['clear', 'rain', 'fog']},
            'rain': {'duration_hours': (1, 4), 'next': ['rain', 'clear', 'storm']},
            'fog': {'duration_hours': (0.5, 3), 'next': ['fog', 'clear']},
            'storm': {'duration_hours': (0.5, 2), 'next': ['rain', 'clear']}
        }

class EmergencyGenerator:
    """Generate realistic emergency scenarios"""
    def __init__(self, section):
        self.section = section
        self.emergency_patterns = {
            'equipment_failure': {'probability': 0.02, 'duration_minutes': (10, 45)},
            'medical': {'probability': 0.01, 'duration_minutes': (5, 20)},
            'security': {'probability': 0.005, 'duration_minutes': (15, 60)},
            'power_outage': {'probability': 0.008, 'duration_minutes': (20, 120)}
        }
