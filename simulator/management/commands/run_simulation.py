# management/commands/run_simulation.py
from django.core.management.base import BaseCommand
from simulator.engine import RailwaySimulator
from core.models import Section, Train
from django.utils import timezone
from datetime import timedelta

class Command(BaseCommand):
    help = 'Run railway simulation'
    
    def add_arguments(self, parser):
        parser.add_argument('--hours', type=int, default=6, help='Simulation duration in hours')
        parser.add_argument('--section', type=int, help='Section ID to simulate')
    
    def handle(self, *args, **options):
        hours = options['hours']
        section_id = options['section']
        
        if section_id:
            section = Section.objects.get(id=section_id)
        else:
            section = Section.objects.first()
        
        if not section:
            self.stdout.write(self.style.ERROR('No sections found. Create sample data first.'))
            return
        
        simulator = RailwaySimulator(section, simulation_time=hours * 60)
        simulator.initialize()
        
        # Add some trains with staggered entry times
        trains = Train.objects.all()[:5]
        for i, train in enumerate(trains):
            entry_time = i * 30  # 30 minutes apart
            scheduled_arrival = timezone.now() + timedelta(minutes=entry_time + 120)  # 2 hours journey
            simulator.add_train(train, entry_time, scheduled_arrival)
        
        self.stdout.write(self.style.SUCCESS(
            f"Starting simulation for {section.name} for {hours} hours with {len(trains)} trains..."
        ))
        
        results = simulator.run()
        
        self.stdout.write(self.style.SUCCESS(
            f"Simulation completed!\n"
            f"Throughput: {results['throughput']:.2f} trains/hour\n"
            f"Average delay: {results['avg_delay']:.2f} minutes\n"
            f"Punctuality: {results['punctuality']:.2f}%"
        ))
        
        # Display individual train results
        self.stdout.write("\nTrain results:")
        for train_data in results['trains']:
            self.stdout.write(
                f"{train_data['train'].train_id}: "
                f"Entry: {train_data['entry_time']}min, "
                f"Exit: {train_data['exit_time']}min, "
                f"Delay: {train_data['delay']:.1f}min"
            )