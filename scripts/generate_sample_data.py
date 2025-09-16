import os
import sys
import random
from datetime import datetime, timedelta

# Make sure project root is on sys.path so the 'rail_optimize' package can be imported
# when this script is executed directly from inside the package directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(package_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

def _ensure_django():
    import django
    # Ensure we run from the project root so imports/resolver behave the same
    try:
        os.chdir(project_root)
    except Exception:
        pass
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rail_optimize.settings')
    django.setup()

_ensure_django_called = False

def _maybe_setup_django():
    global _ensure_django_called
    if not _ensure_django_called:
        _ensure_django()
        _ensure_django_called = True

# Delay importing models until Django is configured
def _import_models():
    from core.models import Section, Segment, Signal, Platform, Train, TrainEvent, Decision, KPIRecord
    from django.utils import timezone
    return Section, Segment, Signal, Platform, Train, TrainEvent, Decision, KPIRecord, timezone

def create_sample_data():
    _maybe_setup_django()
    Section, Segment, Signal, Platform, Train, TrainEvent, Decision, KPIRecord, timezone = _import_models()

    # Create section
    section, created = Section.objects.get_or_create(
        name="Mumbai Central - Vadodara",
        defaults={
            'length_km': 400,
            'capacity': 20,
            'description': "Main Western Railway corridor"
        }
    )
    
    # Create segments
    segments_data = [
        {'name': 'Mumbai Central - Borivali', 'length_km': 30, 'sequence': 1, 'max_speed': 100},
        {'name': 'Borivali - Virar', 'length_km': 25, 'sequence': 2, 'max_speed': 100},
        {'name': 'Virar - Vapi', 'length_km': 80, 'sequence': 3, 'max_speed': 110},
        {'name': 'Vapi - Surat', 'length_km': 100, 'sequence': 4, 'max_speed': 110},
        {'name': 'Surat - Bharuch', 'length_km': 60, 'sequence': 5, 'max_speed': 100},
        {'name': 'Bharuch - Vadodara', 'length_km': 45, 'sequence': 6, 'max_speed': 100},
    ]
    
    segments = []
    for seg_data in segments_data:
        segment, created = Segment.objects.get_or_create(
            section=section,
            name=seg_data['name'],
            defaults={
                'length_km': seg_data['length_km'],
                'sequence': seg_data['sequence'],
                'max_speed': seg_data['max_speed'],
                'has_siding': random.choice([True, False])
            }
        )
        segments.append(segment)
    
    # Create signals for each segment
    for segment in segments:
        for i in range(1, 4):  # 3 signals per segment
            Signal.objects.get_or_create(
                segment=segment,
                name=f"Signal {i}",
                defaults={
                    'position_km': segment.length_km * (i / 4),
                    'signal_type': random.choice(['main', 'distant', 'junction'])
                }
            )
    
    # Create platforms
    platform_names = ['Platform 1', 'Platform 2', 'Platform 3', 'Platform 4']
    for segment in segments[:3]:  # Platforms only in first 3 segments
        for platform_name in platform_names[:2]:  # 2 platforms per segment
            Platform.objects.get_or_create(
                segment=segment,
                name=f"{segment.name} - {platform_name}",
                defaults={
                    'length_m': random.randint(300, 600),
                    'can_cross': random.choice([True, False])
                }
            )
    
    # Create sample trains
    trains_data = [
        {'train_id': '12951', 'name': 'Mumbai Central - Delhi Rajdhani', 'train_type': 'express', 'priority': 1},
        {'train_id': '12953', 'name': 'Mumbai Central - Ahmedabad Shatabdi', 'train_type': 'express', 'priority': 1},
        {'train_id': '19031', 'name': 'Mumbai Central - Firozpur Janata', 'train_type': 'express', 'priority': 2},
        {'train_id': '12933', 'name': 'Valsad - Virar Local', 'train_type': 'local', 'priority': 3},
        {'train_id': '50103', 'name': 'Freight Train 50103', 'train_type': 'freight', 'priority': 4},
        {'train_id': '22955', 'name': 'Mumbai Central - Jaipur Superfast', 'train_type': 'express', 'priority': 2},
        {'train_id': '59443', 'name': 'Mumbai Central - Surat Passenger', 'train_type': 'local', 'priority': 3},
    ]
    
    trains = []
    for train_data in trains_data:
        train, created = Train.objects.get_or_create(
            train_id=train_data['train_id'],
            defaults={
                'name': train_data['name'],
                'train_type': train_data['train_type'],
                'priority': train_data['priority'],
                'length_m': random.randint(300, 500),
                'max_speed': 100 if train_data['train_type'] == 'express' else 80,
                'scheduled_arrival': timezone.now() + timedelta(hours=random.randint(1, 6))
            }
        )
        trains.append(train)
    
    # Create some sample events
    for train in trains[:3]:  # Events for first 3 trains
        for i in range(3):  # 3 events per train
            TrainEvent.objects.create(
                train=train,
                event_type=random.choice(['enter_segment', 'pass_signal', 'stop', 'depart']),
                segment=random.choice(segments),
                timestamp=timezone.now() - timedelta(minutes=random.randint(10, 60))
            )
    
    # Create sample decisions
    for i in range(5):
        decision = Decision.objects.create(
            decision_type=random.choice(['precedence', 'crossing', 'hold', 'reroute']),
            recommended_action=f"Sample decision {i+1}",
            explanation="This is a sample decision for demonstration purposes.",
            is_implemented=random.choice([True, False]),
            manual_override=random.choice([True, False]) if i == 2 else False
        )
        decision.trains_involved.set(random.sample(list(trains), min(2, len(trains))))
    
    # Create sample KPI records
    for i in range(24):  # 24 hours of data
        KPIRecord.objects.create(
            section=section,
            timestamp=timezone.now() - timedelta(hours=23-i),
            throughput_trains=random.randint(10, 25),
            avg_delay_minutes=random.uniform(0, 30),
            punctuality_percent=random.uniform(70, 98),
            utilization_percent=random.uniform(60, 95)
        )
    
    print("Sample data created successfully!")
    print(f"Section: {section.name}")
    print(f"Segments: {Segment.objects.filter(section=section).count()}")
    print(f"Trains: {Train.objects.count()}")
    print(f"Events: {TrainEvent.objects.count()}")
    print(f"Decisions: {Decision.objects.count()}")
    print(f"KPI Records: {KPIRecord.objects.count()}")

if __name__ == '__main__':
    create_sample_data()

def run():
    """Entry point for manage.py runscript or other runners."""
    create_sample_data()