#!/usr/bin/env python
"""Test script for railway optimization engines"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rail_optimize.settings')
django.setup()

from core.models import Section, Train
from decision.engines.heuristic_engine import HeuristicEngine
from decision.engines.ilp_engine import AdvancedILPEngine
from django.utils import timezone

def test_engines():
    """Test both heuristic and ILP engines"""
    print("Testing Railway Optimization Engines...")
    
    # Get sample data
    section = Section.objects.first()
    if not section:
        print("ERROR: No section found! Please generate sample data first.")
        return
    
    trains = list(Train.objects.all()[:3])
    if not trains:
        print("ERROR: No trains found! Please generate sample data first.")
        return
    
    print(f"Testing with section: {section.name}")
    print(f"Number of trains: {len(trains)}")
    
    # Test Heuristic Engine
    print("\n=== Testing Heuristic Engine ===")
    try:
        heuristic_engine = HeuristicEngine(section)
        heuristic_results = heuristic_engine.decide_precedence(trains, timezone.now())
        print(f"✓ Heuristic engine successful: {len(heuristic_results)} recommendations")
        for i, rec in enumerate(heuristic_results[:3]):
            print(f"  {i+1}. {rec.get('action', 'N/A')}")
    except Exception as e:
        print(f"✗ Heuristic engine error: {e}")
    
    # Test ILP Engine
    print("\n=== Testing ILP Engine ===")
    try:
        ilp_engine = AdvancedILPEngine(section)
        ilp_results = ilp_engine.optimize_comprehensive_schedule(trains)
        print(f"✓ ILP engine successful: {ilp_results.get('status', 'unknown')}")
        if 'recommendations' in ilp_results:
            print(f"  Generated {len(ilp_results['recommendations'])} ILP recommendations")
    except Exception as e:
        print(f"✗ ILP engine error: {e}")
    
    print("\nEngine testing completed!")

if __name__ == '__main__':
    test_engines()
