#!/usr/bin/env python
"""
🚂 ULTIMATE SIH DEMO DATABASE GENERATOR 🚂
Creates impressive Indian railway data for hackathon presentation
Includes major routes, realistic trains, live scenarios
"""

import os
import sys
import random
from datetime import datetime, timedelta
from decimal import Decimal

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rail_optimize.settings')
import django
django.setup()

from django.utils import timezone
from core.models import *

class UltimateDemoDataGenerator:
    """Generate world-class demo data for SIH presentation"""
    
    def __init__(self):
        self.created_objects = {
            'sections': [],
            'trains': [],
            'events': [],
            'decisions': []
        }
    
    def clear_existing_data(self):
        """Clean slate for demo"""
        print("🧹 Clearing existing data...")
        models_to_clear = [
            KPIRecord, TrainEvent, Decision, ResourceLock, 
            Train, Platform, Signal, Segment, 
            WeatherCondition, EmergencyEvent, PredictionModel, Section
        ]
        
        for model in models_to_clear:
            count = model.objects.count()
            if count > 0:
                model.objects.all().delete()
                print(f"   Cleared {count} {model.__name__} records")
    
    def create_major_indian_routes(self):
        """Create major Indian railway corridors"""
        print("🚆 Creating Major Indian Railway Corridors...")
        
        routes = [
            {
                'name': 'Delhi - Mumbai Western Railway',
                'length_km': 1384,
                'capacity': 50,
                'description': 'Busiest railway corridor in India connecting capital to financial hub',
                'segments': [
                    ('New Delhi', 0, 45, 'Major junction with 16 platforms'),
                    ('Gurgaon', 32, 80, 'Industrial hub with heavy passenger traffic'),
                    ('Rewari', 82, 100, 'Historical railway junction'),
                    ('Jaipur', 308, 110, 'Pink city major station'),
                    ('Ajmer', 445, 100, 'Religious tourism hub'),
                    ('Jodhpur', 584, 120, 'Desert region major junction'),
                    ('Ahmedabad', 845, 130, 'Gujarat commercial center'),
                    ('Vadodara', 986, 140, 'Industrial city'),
                    ('Surat', 1150, 120, 'Diamond city'),
                    ('Mumbai Central', 1384, 60, 'Financial capital terminus')
                ]
            },
            {
                'name': 'Chennai - Bangalore Golden Quadrilateral',
                'length_km': 362,
                'capacity': 35,
                'description': 'High-speed corridor connecting IT hubs',
                'segments': [
                    ('Chennai Central', 0, 50, 'South India major hub'),
                    ('Tambaram', 27, 80, 'Suburban junction'),
                    ('Chengalpattu', 56, 100, 'Regional center'),
                    ('Villupuram', 164, 110, 'Important junction'),
                    ('Salem', 278, 120, 'Industrial center'),
                    ('Bangalore City', 362, 70, 'Silicon Valley of India')
                ]
            },
            {
                'name': 'Kolkata - Howrah Circular Railway',
                'length_km': 185,
                'capacity': 40,
                'description': 'Eastern India busy suburban network',
                'segments': [
                    ('Howrah Junction', 0, 40, 'Historic major terminus'),
                    ('Sealdah', 8, 45, 'Major suburban hub'),
                    ('Dum Dum', 22, 60, 'Airport connection'),
                    ('Barrackpore', 35, 80, 'Industrial suburb'),
                    ('Naihati', 48, 90, 'Railway workshop town'),
                    ('Bandel', 65, 85, 'River port town')
                ]
            }
        ]
        
        for route_data in routes:
            section = Section.objects.create(
                name=route_data['name'],
                length_km=route_data['length_km'],
                capacity=route_data['capacity'],
                description=route_data['description'],
                current_weather=random.choice(['clear', 'rain', 'fog']),
                visibility_km=random.uniform(5, 15),
                temperature_celsius=random.uniform(15, 35),
                is_active=True,
                current_throughput=random.uniform(15, 30),
                avg_delay_minutes=random.uniform(2, 15),
                utilization_percent=random.uniform(60, 85)
            )
            
            self.created_objects['sections'].append(section)
            print(f"   ✓ Created route: {section.name}")
            
            # Create segments with realistic data
            for i, (name, km, max_speed, desc) in enumerate(route_data['segments']):
                segment = Segment.objects.create(
                    section=section,
                    name=name,
                    length_km=route_data['segments'][i+1][1] - km if i < len(route_data['segments'])-1 else 50,
                    sequence=i+1,
                    max_speed=max_speed,
                    has_siding=random.choice([True, False]),
                    gradient_percent=random.uniform(-2, 3),
                    curve_radius_m=random.randint(800, 2000) if random.random() > 0.3 else None,
                    electrified=True,
                    is_blocked=False,
                    current_occupancy=random.randint(0, 2),
                    max_occupancy=random.randint(2, 4),
                    platform_count=random.randint(2, 8),
                    signal_count=random.randint(2, 6)
                )
                
                # Create signals for each segment
                for j in range(random.randint(2, 4)):
                    Signal.objects.create(
                        segment=segment,
                        name=f"{name} Signal {j+1}",
                        position_km=random.uniform(0, segment.length_km),
                        signal_type=random.choice(['main', 'distant', 'junction']),
                        current_state=random.choice(['green', 'yellow', 'red']),
                        auto_control=True
                    )
                
                # Create platforms for major stations
                if 'Central' in name or 'Junction' in name or i == 0 or i == len(route_data['segments'])-1:
                    for k in range(random.randint(3, 8)):
                        Platform.objects.create(
                            segment=segment,
                            name=f"Platform {k+1}",
                            length_m=random.randint(300, 500),
                            can_cross=random.choice([True, False]),
                            is_occupied=random.choice([True, False])
                        )
    
    def create_realistic_train_fleet(self):
        """Create diverse Indian train fleet"""
        print("🚄 Creating Realistic Indian Train Fleet...")
        
        train_types = [
            # Express trains
            ('Rajdhani Express', 'express', 1, 'Red and yellow livery', 160, 24),
            ('Shatabdi Express', 'express', 1, 'White and blue AC train', 150, 16),
            ('Duronto Express', 'express', 1, 'Non-stop superfast', 130, 22),
            ('Garib Rath', 'express', 2, 'AC budget express', 130, 20),
            ('Jan Shatabdi', 'express', 2, 'Daytime express', 110, 14),
            
            # Mail/Express
            ('Mail Express', 'express', 2, 'Long distance mail', 110, 20),
            ('Superfast Express', 'express', 2, 'Premium express service', 120, 18),
            ('Express Train', 'express', 3, 'Regular express', 100, 16),
            
            # Local trains
            ('EMU Local', 'local', 4, 'Electric suburban', 80, 12),
            ('MEMU Local', 'local', 4, 'Mainline suburban', 90, 16),
            ('Passenger Train', 'local', 5, 'All stations local', 70, 20),
            
            # Freight
            ('Container Freight', 'freight', 3, 'Container cargo', 75, 50),
            ('Coal Freight', 'freight', 4, 'Coal transportation', 60, 58),
            ('Steel Freight', 'freight', 3, 'Iron and steel', 80, 45),
            
            # Special
            ('Vande Bharat', 'special', 1, 'Indigenous semi-high speed', 180, 16),
            ('Tejas Express', 'special', 1, 'Premium service', 130, 20),
        ]
        
        sections = list(Section.objects.all())
        
        for section in sections:
            section_trains = []
            
            # Create trains for each type
            for i, (name_prefix, train_type, priority, description, max_speed, length) in enumerate(train_types):
                # Create multiple trains of each type
                for j in range(random.randint(1, 3)):
                    train_number = f"{random.randint(10000, 99999)}"
                    
                    # Generate realistic timing
                    base_time = timezone.now().replace(minute=0, second=0, microsecond=0)
                    departure_offset = random.randint(-2, 8)  # -2 to +8 hours from now
                    journey_duration = random.randint(180, 720)  # 3-12 hours
                    
                    scheduled_departure = base_time + timedelta(hours=departure_offset)
                    scheduled_arrival = scheduled_departure + timedelta(minutes=journey_duration)
                    
                    # Add some delay reality
                    delay_minutes = random.randint(0, 45) if random.random() > 0.3 else 0
                    actual_departure = scheduled_departure + timedelta(minutes=delay_minutes) if delay_minutes else None
                    
                    train = Train.objects.create(
                        train_id=train_number,
                        name=f"{train_number} {name_prefix}",
                        train_type=train_type,
                        priority=priority,
                        length_m=length * 20,  # Convert coaches to meters
                        max_speed=max_speed,
                        scheduled_departure=scheduled_departure,
                        scheduled_arrival=scheduled_arrival,
                        actual_departure=actual_departure,
                        current_status=random.choice(['scheduled', 'running', 'delayed']) if departure_offset <= 0 else 'scheduled',
                        current_segment=random.choice(list(section.segments.all())) if random.random() > 0.5 else None,
                        current_speed=random.randint(40, max_speed) if departure_offset <= 0 else 0,
                        current_position_km=random.uniform(0, 100),
                        passenger_count=random.randint(200, 1500) if train_type != 'freight' else 0,
                        cargo_weight_tons=random.randint(1000, 4000) if train_type == 'freight' else 0,
                        fuel_level_percent=random.uniform(30, 100),
                        driver_name=f"Driver {random.choice(['Rajesh', 'Suresh', 'Amit', 'Pradeep', 'Vinod'])} {random.choice(['Kumar', 'Singh', 'Sharma', 'Gupta', 'Yadav'])}",
                        crew_count=random.randint(3, 6)
                    )
                    
                    section_trains.append(train)
                    self.created_objects['trains'].append(train)
            
            print(f"   ✓ Created {len(section_trains)} trains for {section.name}")
    
    def create_live_events_and_scenarios(self):
        """Create realistic operational events"""
        print("⚡ Creating Live Operational Events...")
        
        trains = list(Train.objects.all())
        segments = list(Segment.objects.all())
        
        # Create various types of events
        event_scenarios = [
            ('enter_segment', 'Train entering segment', 'normal'),
            ('pass_signal', 'Signal cleared', 'normal'),
            ('stop', 'Scheduled stop', 'normal'),
            ('depart', 'Departure from station', 'normal'),
            ('speed_restriction', 'Speed restriction due to weather', 'caution'),
            ('emergency_stop', 'Emergency brake application', 'emergency'),
            ('weather_delay', 'Monsoon affecting schedule', 'weather'),
            ('fuel_stop', 'Unscheduled fuel stop', 'maintenance'),
        ]
        
        for train in trains[:20]:  # Create events for first 20 trains
            num_events = random.randint(2, 6)
            
            for i in range(num_events):
                event_type, description, category = random.choice(event_scenarios)
                
                # Create realistic event timing
                event_time = timezone.now() - timedelta(
                    minutes=random.randint(0, 240)  # Last 4 hours
                )
                
                event = TrainEvent.objects.create(
                    train=train,
                    event_type=event_type,
                    timestamp=event_time,
                    segment=random.choice(segments) if random.random() > 0.3 else None,
                    speed_at_event=random.randint(0, train.max_speed),
                    position_km=random.uniform(0, 50),
                    delay_minutes=random.uniform(0, 30) if 'delay' in event_type else 0,
                    details={
                        'description': description,
                        'category': category,
                        'weather_impact': random.choice([True, False]),
                        'automation_level': random.choice(['manual', 'semi-auto', 'automatic'])
                    }
                )
                
                self.created_objects['events'].append(event)
        
        print(f"   ✓ Created {len(self.created_objects['events'])} operational events")
    
    def create_ai_decisions_and_kpis(self):
        """Generate AI optimization decisions and KPIs"""
        print("🧠 Generating AI Decisions and Performance Data...")
        
        sections = list(Section.objects.all())
        trains = list(Train.objects.all())
        
        # Create optimization decisions
        decision_types = [
            ('precedence', 'Train precedence optimization', 'high'),
            ('reroute', 'Dynamic route optimization', 'medium'),
            ('speed_limit', 'Adaptive speed control', 'high'),
            ('platform_change', 'Platform reallocation', 'medium'),
            ('emergency', 'Emergency response protocol', 'very_high'),
        ]
        
        for i in range(30):  # Create 30 decisions
            decision_type, description, confidence = random.choice(decision_types)
            
            decision = Decision.objects.create(
                decision_type=decision_type,
                recommended_action=f"AI recommends: {description}",
                explanation=f"Based on current traffic analysis and {random.choice(['weather conditions', 'passenger load', 'infrastructure status', 'emergency protocols'])}",
                confidence_level=confidence,
                expected_benefit=f"Estimated {random.randint(5, 25)}% improvement in {random.choice(['throughput', 'punctuality', 'fuel efficiency'])}",
                is_implemented=random.choice([True, False]),
                algorithm_used=random.choice(['ilp', 'heuristic', 'ml']),
                computation_time_ms=random.uniform(100, 3000)
            )
            
            # Add involved trains
            decision.trains_involved.set(random.sample(trains, random.randint(1, 4)))
            self.created_objects['decisions'].append(decision)
        
        # Create comprehensive KPI records
        for section in sections:
            for hours_ago in range(24):  # 24 hours of data
                timestamp = timezone.now() - timedelta(hours=hours_ago)
                
                # Simulate realistic daily patterns
                hour = timestamp.hour
                if 6 <= hour <= 10 or 17 <= hour <= 21:  # Peak hours
                    throughput = random.randint(20, 35)
                    delay = random.uniform(5, 20)
                    punctuality = random.uniform(70, 85)
                    utilization = random.uniform(80, 95)
                else:  # Off-peak
                    throughput = random.randint(8, 20)
                    delay = random.uniform(2, 10)
                    punctuality = random.uniform(85, 95)
                    utilization = random.uniform(40, 70)
                
                KPIRecord.objects.create(
                    section=section,
                    timestamp=timestamp,
                    throughput_trains=throughput,
                    avg_delay_minutes=delay,
                    punctuality_percent=punctuality,
                    utilization_percent=utilization,
                    avg_speed_kmh=random.uniform(80, 120),
                    fuel_efficiency=random.uniform(2.5, 4.2),
                    passenger_satisfaction=random.uniform(6.5, 9.2),
                    safety_incidents=random.randint(0, 2),
                    co2_emissions_kg=random.uniform(1200, 2800),
                    noise_level_db=random.uniform(65, 85)
                )
        
        print(f"   ✓ Created {len(self.created_objects['decisions'])} AI decisions")
        print(f"   ✓ Generated {24 * len(sections)} KPI records")
    
    def create_weather_and_emergencies(self):
        """Add realistic weather and emergency scenarios"""
        print("🌦️ Creating Weather and Emergency Scenarios...")
        
        sections = list(Section.objects.all())
        
        # Weather conditions
        weather_scenarios = [
            ('clear', 25, 10, 0, 15),
            ('rain', 22, 5, 15, 12),
            ('fog', 18, 2, 0, 8),
            ('storm', 20, 3, 35, 5),
        ]
        
        for section in sections:
            for i in range(12):  # 12 hours of weather data
                weather_type, temp, visibility, precip, wind = random.choice(weather_scenarios)
                
                WeatherCondition.objects.create(
                    section=section,
                    timestamp=timezone.now() - timedelta(hours=i),
                    temperature=temp + random.uniform(-5, 5),
                    humidity_percent=random.uniform(40, 90),
                    wind_speed_kmh=wind + random.uniform(0, 10),
                    precipitation_mm=precip + random.uniform(0, 10),
                    visibility_km=visibility + random.uniform(0, 5),
                    weather_type=weather_type
                )
        
        # Emergency events
        emergency_types = [
            ('equipment_failure', 'Signal failure at junction', 'medium'),
            ('medical', 'Medical emergency on train', 'high'),
            ('security', 'Suspicious activity reported', 'high'),
            ('power_outage', 'Traction power failure', 'critical'),
            ('natural_disaster', 'Flooding on tracks', 'critical'),
        ]
        
        for i in range(5):  # 5 emergency scenarios
            event_type, desc, severity = random.choice(emergency_types)
            
            emergency = EmergencyEvent.objects.create(
                event_type=event_type,
                section=random.choice(sections),
                severity=severity,
                description=desc,
                response_actions=f"Emergency protocol activated: {random.choice(['Traffic control notified', 'Emergency services dispatched', 'Alternative route activated', 'Passenger evacuation initiated'])}",
                timestamp=timezone.now() - timedelta(minutes=random.randint(10, 300))
            )
            
            # Resolve some emergencies
            if random.random() > 0.4:
                emergency.resolved_at = emergency.timestamp + timedelta(minutes=random.randint(30, 180))
                emergency.save()
        
        print("   ✓ Created weather conditions and emergency scenarios")
    
    def generate_ultimate_demo_data(self):
        """Generate complete demo-ready dataset"""
        print("\n" + "="*70)
        print("🚂 GENERATING ULTIMATE SIH DEMO DATABASE 🚂")
        print("="*70)
        
        self.clear_existing_data()
        self.create_major_indian_routes()
        self.create_realistic_train_fleet()
        self.create_live_events_and_scenarios()
        self.create_ai_decisions_and_kpis()
        self.create_weather_and_emergencies()
        
        # Summary
        print("\n" + "="*70)
        print("🎉 ULTIMATE DEMO DATABASE READY! 🎉")
        print("="*70)
        print(f"📍 Sections (Routes): {len(self.created_objects['sections'])}")
        print(f"🚄 Trains: {len(self.created_objects['trains'])}")
        print(f"⚡ Events: {len(self.created_objects['events'])}")
        print(f"🧠 AI Decisions: {len(self.created_objects['decisions'])}")
        print(f"📊 KPI Records: {KPIRecord.objects.count()}")
        print(f"🌦️ Weather Records: {WeatherCondition.objects.count()}")
        print(f"🚨 Emergency Events: {EmergencyEvent.objects.count()}")
        print(f"🛤️ Total Segments: {Segment.objects.count()}")
        print(f"🚥 Signals: {Signal.objects.count()}")
        print(f"🏗️ Platforms: {Platform.objects.count()}")
        
        print("\n✅ Your system is now HACKATHON READY!")
        print("🎯 Features to highlight:")
        print("   • Major Indian railway corridors")
        print("   • Realistic train operations")
        print("   • AI-powered optimization")
        print("   • Live weather integration")
        print("   • Emergency response systems")
        print("   • Real-time performance tracking")
        
        return True

if __name__ == '__main__':
    generator = UltimateDemoDataGenerator()
    generator.generate_ultimate_demo_data()
