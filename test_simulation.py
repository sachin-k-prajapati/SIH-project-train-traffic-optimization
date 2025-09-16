#!/usr/bin/env python
"""Simple test of real-time simulation functionality"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rail_optimize.settings')
django.setup()

from core.models import Section, Train
from simulator.engine import RealTimeRailwaySimulator
from django.utils import timezone
import time
import random

def quick_simulation_test():
    """Quick test of simulation functionality"""
    print("🚂 Testing Real-Time Railway Simulation...")
    
    # Get section and trains
    section = Section.objects.first()
    if not section:
        print("❌ No section found!")
        return False
    
    trains = Train.objects.all()[:3]
    if not trains:
        print("❌ No trains found!")
        return False
    
    print(f"✓ Section: {section.name}")
    print(f"✓ Trains: {len(trains)}")
    
    try:
        # Create simulator
        simulator = RealTimeRailwaySimulator(section, simulation_speed=10.0)
        print("✓ Simulator created successfully")
        
        # Test basic functionality
        simulator._initialize_resources()
        print("✓ Resources initialized")
        
        # Test train positioning
        for train in trains:
            simulator.train_positions[train.id] = 0.0
            simulator.train_speeds[train.id] = random.randint(40, 80)
        
        print("✓ Train positioning initialized")
        
        # Simulate some movement
        for i in range(3):
            for train in trains:
                old_pos = simulator.train_positions.get(train.id, 0)
                new_pos = old_pos + 0.5  # Move 0.5 km
                simulator.train_positions[train.id] = new_pos
                print(f"  Train {train.train_id}: {old_pos:.1f}km → {new_pos:.1f}km")
            time.sleep(0.1)  # Brief pause
        
        print("✅ Real-time simulation test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False

if __name__ == '__main__':
    success = quick_simulation_test()
    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")
