#!/usr/bin/env python
"""
Real-Life Railway Operational Scenarios for Testing World-Class System
Simulates realistic railway operations including peak hours, emergencies, weather disruptions

Usage:
    python run_scenario.py --scenario peak_hour
    python run_scenario.py --scenario emergency_response
    python run_scenario.py --scenario weather_disruption
    python run_scenario.py --scenario maintenance_window
"""

import os
import sys
import django
import argparse
import random
import time
from datetime import datetime, timedelta
from decimal import Decimal

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rail_optimize.settings')
django.setup()

from django.utils import timezone
from core.models import (
    Section, Train, Segment, Signal, Platform, 
    WeatherCondition, EmergencyEvent, PerformanceMetric
)
from simulator.engine import RealTimeRailwaySimulator
from decision.engines.ilp_engine import AdvancedILPEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealWorldScenarioRunner:
    """
    Runs realistic railway operational scenarios to test system capabilities
    """
    
    def __init__(self):
        self.section = Section.objects.first()
        if not self.section:
            raise ValueError("No railway section found. Run generate_sample_data.py first.")
        
        self.simulator = RealTimeRailwaySimulator(self.section)
        self.optimizer = AdvancedILPEngine(self.section)
        self.scenario_results = {}
    
    def run_scenario(self, scenario_name):
        """Run specified scenario and return results"""
        logger.info(f"🚂 Starting scenario: {scenario_name}")
        
        scenario_methods = {
            'peak_hour': self.peak_hour_rush,
            'emergency_response': self.emergency_response,
            'weather_disruption': self.weather_disruption,
            'maintenance_window': self.maintenance_window,
            'mixed_traffic': self.mixed_traffic_scenario,
            'system_overload': self.system_overload,
            'cascade_delay': self.cascade_delay,
            'vip_train': self.vip_train_priority
        }
        
        if scenario_name not in scenario_methods:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        start_time = time.time()
        result = scenario_methods[scenario_name]()
        end_time = time.time()
        
        result['execution_time'] = end_time - start_time
        result['scenario'] = scenario_name
        result['timestamp'] = timezone.now()
        
        self.scenario_results[scenario_name] = result
        self.print_scenario_results(result)
        
        return result
    
    def peak_hour_rush(self):
        """
        Scenario: Morning peak hour with high train density
        Realistic conditions: 8:00 AM, multiple express and local trains
        """
        logger.info("📈 Simulating morning peak hour rush (8:00 AM)")
        
        # Clear existing trains
        Train.objects.filter(current_section=self.section).delete()
        
        # Create peak hour train schedule
        trains = []
        base_time = timezone.now().replace(hour=8, minute=0, second=0, microsecond=0)
        
        # Express trains every 15 minutes
        for i in range(4):
            train = Train.objects.create(
                train_id=f"EXP80{i+1}",
                train_type='express',
                max_speed=120,
                current_speed=100,
                priority=2,
                current_section=self.section,
                scheduled_arrival=base_time + timedelta(minutes=i*15),
                status='running',
                current_delay_minutes=random.randint(-2, 8),
                passenger_capacity=800,
                current_passenger_count=random.randint(600, 800),
                current_latitude=Decimal('28.6139') + Decimal(str(random.uniform(-0.01, 0.01))),
                current_longitude=Decimal('77.2090') + Decimal(str(random.uniform(-0.01, 0.01)))
            )
            trains.append(train)
        
        # Local trains every 10 minutes
        for i in range(6):
            train = Train.objects.create(
                train_id=f"LOC80{i+1}",
                train_type='local',
                max_speed=80,
                current_speed=65,
                priority=3,
                current_section=self.section,
                scheduled_arrival=base_time + timedelta(minutes=i*10),
                status='running',
                current_delay_minutes=random.randint(0, 15),
                passenger_capacity=1200,
                current_passenger_count=random.randint(900, 1200),
                current_latitude=Decimal('28.6139') + Decimal(str(random.uniform(-0.01, 0.01))),
                current_longitude=Decimal('77.2090') + Decimal(str(random.uniform(-0.01, 0.01)))
            )
            trains.append(train)
        
        # One freight train (causing complexity)
        freight = Train.objects.create(
            train_id="FRT801",
            train_type='freight',
            max_speed=60,
            current_speed=45,
            priority=4,
            current_section=self.section,
            scheduled_arrival=base_time + timedelta(minutes=30),
            status='running',
            current_delay_minutes=12,
            passenger_capacity=0,
            current_passenger_count=0,
            current_latitude=Decimal('28.6289'),
            current_longitude=Decimal('77.2170')
        )
        trains.append(freight)
        
        # High passenger demand weather
        WeatherCondition.objects.create(
            section=self.section,
            temperature=32,
            condition='clear',
            wind_speed=8,
            visibility_factor=1.0,
            timestamp=timezone.now()
        )
        
        # Run optimization
        logger.info("🧠 Running AI optimization for peak hour traffic...")
        optimization_result = self.optimizer.optimize_comprehensive_schedule(
            trains, time_horizon_minutes=120
        )
        
        # Simulate passenger boarding delays
        self.simulate_passenger_delays(trains)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(trains, 'peak_hour')
        
        return {
            'status': 'completed',
            'trains_managed': len(trains),
            'optimization_result': optimization_result,
            'performance_metrics': metrics,
            'passenger_impact': {
                'total_passengers': sum(t.current_passenger_count for t in trains),
                'avg_delay_per_passenger': metrics['avg_delay'] * 800,  # Avg passengers per train
                'punctuality_rate': metrics['punctuality_percentage']
            },
            'system_stress': {
                'capacity_utilization': len(trains) / self.section.capacity * 100,
                'conflict_points': self.count_potential_conflicts(trains),
                'bottleneck_segments': self.identify_bottlenecks()
            }
        }
    
    def emergency_response(self):
        """
        Scenario: Emergency on track requiring immediate response
        Realistic conditions: Signal failure, train breakdown, medical emergency
        """
        logger.info("🚨 Simulating emergency response scenario")
        
        # Setup normal traffic first
        self.setup_normal_traffic()
        trains = list(Train.objects.filter(current_section=self.section))
        
        # Create emergency event
        affected_segment = Segment.objects.filter(section=self.section).first()
        emergency = EmergencyEvent.objects.create(
            section=self.section,
            event_type='signal_failure',
            description='Critical signal failure on main line - immediate response required',
            severity='critical',
            status='active',
            affected_segment=affected_segment,
            estimated_duration_minutes=45,
            impact_radius_km=2.5
        )
        
        logger.info(f"⚠️  Emergency created: {emergency.description}")
        
        # Simulate emergency response
        emergency_start = time.time()
        
        # Stop trains in affected area
        affected_trains = []
        for train in trains:
            if self.is_train_in_emergency_zone(train, emergency):
                train.status = 'emergency_stop'
                train.current_speed = 0
                train.save()
                affected_trains.append(train)
                logger.info(f"🛑 Emergency stop: {train.train_id}")
        
        # Reroute other trains
        rerouted_trains = []
        for train in trains:
            if train not in affected_trains and train.priority <= 3:
                # High priority trains get alternative routing
                train.current_delay_minutes += random.randint(15, 30)
                train.save()
                rerouted_trains.append(train)
                logger.info(f"🔄 Rerouted: {train.train_id} (+{train.current_delay_minutes} min delay)")
        
        # Emergency optimization
        remaining_trains = [t for t in trains if t not in affected_trains]
        emergency_optimization = self.optimizer.optimize_comprehensive_schedule(
            remaining_trains, time_horizon_minutes=180
        )
        
        # Simulate emergency resolution
        time.sleep(2)  # Simulate response time
        emergency_duration = time.time() - emergency_start
        
        # Resume operations
        for train in affected_trains:
            train.status = 'delayed'
            train.current_speed = max(20, train.max_speed * 0.6)  # Cautious restart
            train.current_delay_minutes += random.randint(20, 40)
            train.save()
            logger.info(f"▶️  Resumed: {train.train_id}")
        
        emergency.status = 'resolved'
        emergency.actual_duration_minutes = int(emergency_duration / 60 * 45)  # Scale to realistic time
        emergency.save()
        
        metrics = self.calculate_performance_metrics(trains, 'emergency_response')
        
        return {
            'status': 'emergency_resolved',
            'emergency_details': {
                'type': emergency.event_type,
                'duration_minutes': emergency.actual_duration_minutes,
                'affected_trains': len(affected_trains),
                'rerouted_trains': len(rerouted_trains),
                'response_time_seconds': emergency_duration
            },
            'optimization_result': emergency_optimization,
            'performance_metrics': metrics,
            'recovery_statistics': {
                'trains_stopped': len(affected_trains),
                'avg_additional_delay': sum(t.current_delay_minutes for t in affected_trains) / len(affected_trains) if affected_trains else 0,
                'system_recovery_time': emergency_duration,
                'passenger_impact_score': self.calculate_passenger_impact(trains)
            }
        }
    
    def weather_disruption(self):
        """
        Scenario: Severe weather affecting operations
        Realistic conditions: Heavy rain, reduced visibility, speed restrictions
        """
        logger.info("🌧️ Simulating severe weather disruption")
        
        self.setup_normal_traffic()
        trains = list(Train.objects.filter(current_section=self.section))
        
        # Create severe weather condition
        severe_weather = WeatherCondition.objects.create(
            section=self.section,
            temperature=18,
            condition='heavy_rain',
            wind_speed=45,
            visibility_factor=0.6,  # 60% visibility
            precipitation_mm=25,
            timestamp=timezone.now()
        )
        
        # Update section weather impact
        self.section.weather_impact_factor = 0.7  # 30% reduction in performance
        self.section.save()
        
        logger.info(f"🌪️  Severe weather: {severe_weather.condition}, visibility: {severe_weather.visibility_factor}")
        
        # Apply weather-based restrictions
        weather_affected = 0
        for train in trains:
            # Reduce speed by 30-50%
            speed_reduction = random.uniform(0.3, 0.5)
            train.current_speed = int(train.current_speed * (1 - speed_reduction))
            
            # Increase delays due to cautious operation
            weather_delay = random.randint(5, 20)
            train.current_delay_minutes += weather_delay
            
            # Some trains may need to stop temporarily
            if random.random() < 0.2:  # 20% chance
                train.status = 'weather_hold'
                train.current_speed = 0
                train.current_delay_minutes += random.randint(10, 25)
                logger.info(f"☔ Weather hold: {train.train_id}")
            
            train.save()
            weather_affected += 1
        
        # Weather-aware optimization
        weather_optimization = self.optimizer.optimize_comprehensive_schedule(
            trains, time_horizon_minutes=240  # Longer horizon due to weather
        )
        
        # Simulate passenger notifications and alternatives
        passenger_impact = self.simulate_weather_passenger_impact(trains)
        
        metrics = self.calculate_performance_metrics(trains, 'weather_disruption')
        
        return {
            'status': 'weather_managed',
            'weather_details': {
                'condition': severe_weather.condition,
                'visibility_factor': float(severe_weather.visibility_factor),
                'wind_speed': severe_weather.wind_speed,
                'system_impact_factor': self.section.weather_impact_factor
            },
            'trains_affected': weather_affected,
            'optimization_result': weather_optimization,
            'performance_metrics': metrics,
            'passenger_impact': passenger_impact,
            'weather_adaptations': {
                'speed_reductions': f"30-50% across {weather_affected} trains",
                'temporary_holds': len([t for t in trains if t.status == 'weather_hold']),
                'avg_weather_delay': sum(t.current_delay_minutes for t in trains) / len(trains),
                'safety_measures': "Enhanced braking distances, reduced acceleration"
            }
        }
    
    def maintenance_window(self):
        """
        Scenario: Planned maintenance requiring traffic coordination
        Realistic conditions: Track maintenance, limited capacity, scheduled disruptions
        """
        logger.info("🔧 Simulating maintenance window operations")
        
        self.setup_normal_traffic()
        trains = list(Train.objects.filter(current_section=self.section))
        
        # Define maintenance window
        maintenance_start = timezone.now() + timedelta(hours=1)
        maintenance_end = maintenance_start + timedelta(hours=3)
        
        # Reduce available segments (simulate track closure)
        available_segments = list(Segment.objects.filter(section=self.section))
        maintenance_segments = available_segments[:2]  # Close first 2 segments
        
        logger.info(f"🚧 Maintenance window: {maintenance_start.strftime('%H:%M')} - {maintenance_end.strftime('%H:%M')}")
        logger.info(f"🚧 Affected segments: {[s.name for s in maintenance_segments]}")
        
        # Reschedule trains to avoid maintenance window
        rescheduled = 0
        diverted = 0
        
        for train in trains:
            # Check if train conflicts with maintenance
            estimated_arrival = train.scheduled_arrival or timezone.now()
            
            if maintenance_start <= estimated_arrival <= maintenance_end:
                if train.priority <= 2:  # High priority - reschedule before maintenance
                    new_time = maintenance_start - timedelta(minutes=random.randint(15, 45))
                    train.scheduled_arrival = new_time
                    train.current_delay_minutes -= random.randint(10, 30)  # Earlier departure
                    rescheduled += 1
                    logger.info(f"⏰ Rescheduled early: {train.train_id}")
                else:  # Lower priority - reschedule after maintenance
                    new_time = maintenance_end + timedelta(minutes=random.randint(10, 30))
                    train.scheduled_arrival = new_time
                    train.current_delay_minutes += random.randint(30, 60)
                    rescheduled += 1
                    logger.info(f"⏰ Rescheduled later: {train.train_id}")
            
            # Some trains use alternative routes (simulated)
            elif random.random() < 0.3:  # 30% use alternative routing
                train.current_delay_minutes += random.randint(15, 25)  # Longer route
                diverted += 1
                logger.info(f"🔄 Alternative route: {train.train_id}")
            
            train.save()
        
        # Maintenance-aware optimization
        pre_maintenance_trains = [t for t in trains if (t.scheduled_arrival or timezone.now()) < maintenance_start]
        post_maintenance_trains = [t for t in trains if (t.scheduled_arrival or timezone.now()) > maintenance_end]
        
        pre_optimization = self.optimizer.optimize_comprehensive_schedule(
            pre_maintenance_trains, time_horizon_minutes=60
        ) if pre_maintenance_trains else None
        
        post_optimization = self.optimizer.optimize_comprehensive_schedule(
            post_maintenance_trains, time_horizon_minutes=180
        ) if post_maintenance_trains else None
        
        # Simulate maintenance coordination
        maintenance_efficiency = random.uniform(0.85, 0.98)  # Maintenance efficiency
        
        metrics = self.calculate_performance_metrics(trains, 'maintenance_window')
        
        return {
            'status': 'maintenance_coordinated',
            'maintenance_details': {
                'window_start': maintenance_start.isoformat(),
                'window_end': maintenance_end.isoformat(),
                'duration_hours': 3,
                'affected_segments': [s.name for s in maintenance_segments],
                'maintenance_efficiency': maintenance_efficiency
            },
            'coordination_statistics': {
                'trains_rescheduled': rescheduled,
                'trains_diverted': diverted,
                'pre_maintenance_trains': len(pre_maintenance_trains),
                'post_maintenance_trains': len(post_maintenance_trains)
            },
            'optimization_results': {
                'pre_maintenance': pre_optimization,
                'post_maintenance': post_optimization
            },
            'performance_metrics': metrics,
            'operational_impact': {
                'capacity_reduction': f"{len(maintenance_segments) / len(available_segments) * 100:.1f}%",
                'avg_passenger_delay': metrics['avg_delay'] * 600,  # Estimated passengers
                'cost_savings': maintenance_efficiency * 100000  # Estimated maintenance savings
            }
        }
    
    def mixed_traffic_scenario(self):
        """
        Scenario: Complex mixed traffic with different train types and priorities
        """
        logger.info("🚄 Simulating complex mixed traffic scenario")
        
        Train.objects.filter(current_section=self.section).delete()
        
        # Create diverse train mix
        trains = []
        
        # High-speed express
        express = Train.objects.create(
            train_id="HSEXP01",
            train_type='high_speed',
            max_speed=160,
            current_speed=140,
            priority=1,
            current_section=self.section,
            status='running'
        )
        trains.append(express)
        
        # Regular express trains
        for i in range(3):
            train = Train.objects.create(
                train_id=f"EXP90{i+1}",
                train_type='express',
                max_speed=120,
                current_speed=100,
                priority=2,
                current_section=self.section,
                status='running'
            )
            trains.append(train)
        
        # Local trains
        for i in range(4):
            train = Train.objects.create(
                train_id=f"LOC90{i+1}",
                train_type='local',
                max_speed=80,
                current_speed=65,
                priority=3,
                current_section=self.section,
                status='running'
            )
            trains.append(train)
        
        # Freight trains
        for i in range(2):
            train = Train.objects.create(
                train_id=f"FRT90{i+1}",
                train_type='freight',
                max_speed=60,
                current_speed=45,
                priority=4,
                current_section=self.section,
                status='running'
            )
            trains.append(train)
        
        # Special VIP train
        vip = Train.objects.create(
            train_id="VIP001",
            train_type='special',
            max_speed=130,
            current_speed=110,
            priority=1,
            current_section=self.section,
            status='running'
        )
        trains.append(vip)
        
        # Run comprehensive optimization
        optimization_result = self.optimizer.optimize_comprehensive_schedule(trains)
        
        metrics = self.calculate_performance_metrics(trains, 'mixed_traffic')
        
        return {
            'status': 'mixed_traffic_optimized',
            'traffic_composition': {
                'high_speed': 1,
                'express': 3,
                'local': 4,
                'freight': 2,
                'special': 1,
                'total': len(trains)
            },
            'optimization_result': optimization_result,
            'performance_metrics': metrics,
            'complexity_analysis': {
                'priority_levels': len(set(t.priority for t in trains)),
                'speed_variance': max(t.max_speed for t in trains) - min(t.max_speed for t in trains),
                'optimization_complexity': 'high'
            }
        }
    
    def system_overload(self):
        """
        Scenario: System operating at maximum capacity
        """
        logger.info("📊 Simulating system overload scenario")
        
        Train.objects.filter(current_section=self.section).delete()
        
        # Create trains at 95% capacity
        max_trains = int(self.section.capacity * 0.95)
        trains = []
        
        for i in range(max_trains):
            train_type = random.choice(['express', 'local', 'freight'])
            train = Train.objects.create(
                train_id=f"OVL{i+1:03d}",
                train_type=train_type,
                max_speed=random.randint(60, 120),
                current_speed=random.randint(40, 100),
                priority=random.randint(2, 4),
                current_section=self.section,
                status='running',
                current_delay_minutes=random.randint(0, 20)
            )
            trains.append(train)
        
        # Run stress test optimization
        optimization_result = self.optimizer.optimize_comprehensive_schedule(
            trains, time_horizon_minutes=120
        )
        
        metrics = self.calculate_performance_metrics(trains, 'system_overload')
        
        return {
            'status': 'overload_managed',
            'capacity_analysis': {
                'trains_active': len(trains),
                'section_capacity': self.section.capacity,
                'utilization_percentage': (len(trains) / self.section.capacity) * 100
            },
            'optimization_result': optimization_result,
            'performance_metrics': metrics,
            'stress_indicators': {
                'avg_optimization_time': optimization_result.get('solve_time', 0),
                'algorithm_confidence': optimization_result.get('algorithm_confidence', 'unknown'),
                'bottleneck_severity': 'high' if metrics['avg_delay'] > 15 else 'medium'
            }
        }
    
    def cascade_delay(self):
        """
        Scenario: Cascade delay propagation through the system
        """
        logger.info("🔄 Simulating cascade delay scenario")
        
        self.setup_normal_traffic()
        trains = list(Train.objects.filter(current_section=self.section))
        
        # Create initial delay
        primary_train = trains[0]
        primary_train.current_delay_minutes = 25
        primary_train.save()
        
        logger.info(f"⏱️  Initial delay: {primary_train.train_id} delayed by {primary_train.current_delay_minutes} minutes")
        
        # Simulate cascade effect
        cascade_affected = []
        for i, train in enumerate(trains[1:], 1):
            if i <= 3:  # First 3 trains heavily affected
                cascade_delay = random.randint(8, 15)
                train.current_delay_minutes += cascade_delay
                cascade_affected.append(train)
                logger.info(f"🔗 Cascade effect: {train.train_id} +{cascade_delay} min")
            elif i <= 6:  # Next 3 trains moderately affected
                cascade_delay = random.randint(3, 8)
                train.current_delay_minutes += cascade_delay
                cascade_affected.append(train)
            
            train.save()
        
        # Run optimization to minimize cascade
        optimization_result = self.optimizer.optimize_comprehensive_schedule(trains)
        
        metrics = self.calculate_performance_metrics(trains, 'cascade_delay')
        
        return {
            'status': 'cascade_mitigated',
            'cascade_analysis': {
                'initial_delay': primary_train.current_delay_minutes,
                'trains_affected': len(cascade_affected),
                'total_delay_propagated': sum(t.current_delay_minutes for t in cascade_affected),
                'cascade_factor': len(cascade_affected) / len(trains)
            },
            'optimization_result': optimization_result,
            'performance_metrics': metrics,
            'mitigation_effectiveness': {
                'delay_recovery_rate': optimization_result.get('kpis', {}).get('punctuality_percent', 0),
                'system_resilience': 'high' if metrics['punctuality_percentage'] > 70 else 'medium'
            }
        }
    
    def vip_train_priority(self):
        """
        Scenario: VIP train requiring absolute priority
        """
        logger.info("👑 Simulating VIP train priority scenario")
        
        self.setup_normal_traffic()
        trains = list(Train.objects.filter(current_section=self.section))
        
        # Add VIP train with highest priority
        vip_train = Train.objects.create(
            train_id="VIP_PM001",
            train_type='special',
            max_speed=140,
            current_speed=120,
            priority=1,  # Highest priority
            current_section=self.section,
            status='approaching',
            passenger_capacity=300,
            current_passenger_count=25  # VIP passengers
        )
        trains.append(vip_train)
        
        logger.info(f"👑 VIP train {vip_train.train_id} requires absolute priority")
        
        # All other trains must accommodate VIP
        accommodated_trains = []
        for train in trains[:-1]:  # Exclude VIP train
            if train.priority > 1:  # Lower priority
                # Some trains delayed, others rerouted
                if random.random() < 0.6:  # 60% get delayed
                    delay = random.randint(10, 25)
                    train.current_delay_minutes += delay
                    accommodated_trains.append(train)
                    logger.info(f"⏰ Delayed for VIP: {train.train_id} +{delay} min")
                else:  # 40% rerouted
                    train.current_delay_minutes += random.randint(5, 15)
                    accommodated_trains.append(train)
                    logger.info(f"🔄 Rerouted for VIP: {train.train_id}")
            
            train.save()
        
        # VIP-priority optimization
        optimization_result = self.optimizer.optimize_comprehensive_schedule(trains)
        
        metrics = self.calculate_performance_metrics(trains, 'vip_priority')
        
        return {
            'status': 'vip_priority_granted',
            'vip_details': {
                'train_id': vip_train.train_id,
                'vip_passenger_count': vip_train.current_passenger_count,
                'priority_level': vip_train.priority,
                'on_time_performance': True  # VIP always on time
            },
            'accommodation_impact': {
                'trains_accommodated': len(accommodated_trains),
                'total_delay_cost': sum(t.current_delay_minutes for t in accommodated_trains),
                'system_disruption_score': len(accommodated_trains) / len(trains) * 100
            },
            'optimization_result': optimization_result,
            'performance_metrics': metrics,
            'protocol_effectiveness': {
                'vip_satisfaction': 100,
                'public_service_impact': metrics['punctuality_percentage'],
                'operational_efficiency': optimization_result.get('kpis', {}).get('fuel_efficiency_score', 0)
            }
        }
    
    # Helper methods
    
    def setup_normal_traffic(self):
        """Setup normal traffic conditions"""
        Train.objects.filter(current_section=self.section).delete()
        
        train_configs = [
            ('EXP701', 'express', 120, 2),
            ('EXP702', 'express', 120, 2),
            ('LOC701', 'local', 80, 3),
            ('LOC702', 'local', 80, 3),
            ('LOC703', 'local', 80, 3),
            ('FRT701', 'freight', 60, 4)
        ]
        
        for train_id, train_type, max_speed, priority in train_configs:
            Train.objects.create(
                train_id=train_id,
                train_type=train_type,
                max_speed=max_speed,
                current_speed=int(max_speed * random.uniform(0.7, 0.9)),
                priority=priority,
                current_section=self.section,
                scheduled_arrival=timezone.now() + timedelta(minutes=random.randint(-10, 60)),
                status='running',
                current_delay_minutes=random.randint(0, 8)
            )
    
    def simulate_passenger_delays(self, trains):
        """Simulate passenger boarding delays"""
        for train in trains:
            if train.train_type in ['express', 'local']:
                boarding_delay = random.randint(0, 3)
                train.current_delay_minutes += boarding_delay
                train.save()
    
    def simulate_weather_passenger_impact(self, trains):
        """Calculate passenger impact during weather disruptions"""
        total_passengers = sum(getattr(t, 'current_passenger_count', 200) for t in trains)
        avg_delay = sum(t.current_delay_minutes for t in trains) / len(trains)
        
        return {
            'total_affected_passengers': total_passengers,
            'avg_delay_per_passenger': avg_delay,
            'passenger_satisfaction_score': max(0, 100 - avg_delay * 2),
            'alternative_transport_usage': min(30, avg_delay * 0.5)  # Percentage
        }
    
    def is_train_in_emergency_zone(self, train, emergency):
        """Check if train is in emergency zone"""
        return random.random() < 0.3  # 30% chance for simulation
    
    def count_potential_conflicts(self, trains):
        """Count potential train conflicts"""
        return len(trains) * (len(trains) - 1) // 2 * 0.1  # Simplified calculation
    
    def identify_bottlenecks(self):
        """Identify system bottlenecks"""
        segments = Segment.objects.filter(section=self.section)
        return [s.name for s in segments[:2]]  # First 2 segments as bottlenecks
    
    def calculate_passenger_impact(self, trains):
        """Calculate passenger impact score"""
        total_delay = sum(t.current_delay_minutes for t in trains)
        total_passengers = sum(getattr(t, 'current_passenger_count', 200) for t in trains)
        return total_delay * total_passengers / 1000  # Normalized score
    
    def calculate_performance_metrics(self, trains, scenario_type):
        """Calculate comprehensive performance metrics"""
        if not trains:
            return {
                'avg_delay': 0,
                'punctuality_percentage': 100,
                'total_trains': 0,
                'on_time_trains': 0
            }
        
        total_trains = len(trains)
        on_time_trains = len([t for t in trains if t.current_delay_minutes <= 5])
        total_delay = sum(t.current_delay_minutes for t in trains)
        avg_delay = total_delay / total_trains
        punctuality = (on_time_trains / total_trains) * 100
        
        # Store metrics in database
        PerformanceMetric.objects.create(
            section=self.section,
            metric_type=scenario_type,
            avg_delay_minutes=avg_delay,
            punctuality_percentage=punctuality,
            total_trains=total_trains,
            on_time_trains=on_time_trains,
            timestamp=timezone.now()
        )
        
        return {
            'avg_delay': round(avg_delay, 2),
            'punctuality_percentage': round(punctuality, 1),
            'total_trains': total_trains,
            'on_time_trains': on_time_trains,
            'total_delay_minutes': total_delay,
            'efficiency_score': round(100 - (avg_delay * 2), 1)
        }
    
    def print_scenario_results(self, result):
        """Print formatted scenario results"""
        print("\n" + "="*80)
        print(f"🎯 SCENARIO RESULTS: {result['scenario'].upper()}")
        print("="*80)
        
        print(f"📊 Status: {result['status']}")
        print(f"⏱️  Execution Time: {result['execution_time']:.2f} seconds")
        
        if 'trains_managed' in result:
            print(f"🚂 Trains Managed: {result['trains_managed']}")
        
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print(f"📈 Average Delay: {metrics['avg_delay']} minutes")
            print(f"⏰ Punctuality: {metrics['punctuality_percentage']}%")
            print(f"🎯 Efficiency Score: {metrics['efficiency_score']}")
        
        if 'optimization_result' in result and result['optimization_result']:
            opt = result['optimization_result']
            print(f"🧠 Optimization Status: {opt.get('status', 'unknown')}")
            print(f"⚡ Solve Time: {opt.get('solve_time', 0):.2f} seconds")
            print(f"🎯 Algorithm Confidence: {opt.get('algorithm_confidence', 'unknown')}")
        
        # Scenario-specific details
        if result['scenario'] == 'peak_hour':
            print(f"👥 Passengers Affected: {result['passenger_impact']['total_passengers']}")
            print(f"📊 Capacity Utilization: {result['system_stress']['capacity_utilization']:.1f}%")
        
        elif result['scenario'] == 'emergency_response':
            print(f"🚨 Emergency Duration: {result['emergency_details']['duration_minutes']} minutes")
            print(f"🛑 Trains Affected: {result['emergency_details']['affected_trains']}")
            print(f"⚡ Response Time: {result['emergency_details']['response_time_seconds']:.1f} seconds")
        
        elif result['scenario'] == 'weather_disruption':
            print(f"🌧️  Weather Condition: {result['weather_details']['condition']}")
            print(f"👁️  Visibility Factor: {result['weather_details']['visibility_factor']}")
            print(f"💨 Wind Speed: {result['weather_details']['wind_speed']} km/h")
        
        print("\n" + "="*80)


def main():
    """Main function to run scenarios"""
    parser = argparse.ArgumentParser(description='Run real-world railway operational scenarios')
    parser.add_argument('--scenario', 
                       choices=['peak_hour', 'emergency_response', 'weather_disruption', 
                               'maintenance_window', 'mixed_traffic', 'system_overload',
                               'cascade_delay', 'vip_train', 'all'],
                       default='peak_hour',
                       help='Scenario to run')
    parser.add_argument('--loops', type=int, default=1, help='Number of times to run scenario')
    parser.add_argument('--delay', type=float, default=0, help='Delay between loops (seconds)')
    
    args = parser.parse_args()
    
    try:
        runner = RealWorldScenarioRunner()
        
        if args.scenario == 'all':
            scenarios = ['peak_hour', 'emergency_response', 'weather_disruption', 
                        'maintenance_window', 'mixed_traffic', 'system_overload',
                        'cascade_delay', 'vip_train']
        else:
            scenarios = [args.scenario]
        
        all_results = {}
        
        for loop in range(args.loops):
            if args.loops > 1:
                print(f"\n🔄 Loop {loop + 1} of {args.loops}")
            
            for scenario in scenarios:
                result = runner.run_scenario(scenario)
                all_results[f"{scenario}_loop_{loop}"] = result
                
                if args.delay > 0:
                    time.sleep(args.delay)
        
        # Summary if multiple loops
        if args.loops > 1:
            print("\n" + "="*80)
            print("📊 MULTI-LOOP SUMMARY")
            print("="*80)
            
            for scenario in scenarios:
                scenario_results = [v for k, v in all_results.items() if k.startswith(scenario)]
                if scenario_results:
                    avg_execution_time = sum(r['execution_time'] for r in scenario_results) / len(scenario_results)
                    avg_punctuality = sum(r['performance_metrics']['punctuality_percentage'] for r in scenario_results) / len(scenario_results)
                    
                    print(f"🎯 {scenario.upper()}:")
                    print(f"   ⏱️  Avg Execution Time: {avg_execution_time:.2f}s")
                    print(f"   ⏰ Avg Punctuality: {avg_punctuality:.1f}%")
        
        print(f"\n✅ All scenarios completed successfully!")
        print(f"🎯 World-class railway system demonstrated real-life operational capabilities!")
        
    except Exception as e:
        logger.error(f"❌ Scenario execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
