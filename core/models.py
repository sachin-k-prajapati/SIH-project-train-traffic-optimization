from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import json
from datetime import timedelta

class Section(models.Model):
    name = models.CharField(max_length=200)
    length_km = models.FloatField()
    capacity = models.IntegerField(help_text="Maximum concurrent trains")
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Real-world operational data
    current_weather = models.CharField(max_length=50, default='clear', 
                                     choices=[('clear', 'Clear'), ('rain', 'Rain'), 
                                            ('fog', 'Fog'), ('storm', 'Storm')])
    visibility_km = models.FloatField(default=10.0)
    temperature_celsius = models.FloatField(default=25.0)
    
    # Operational status
    is_active = models.BooleanField(default=True)
    maintenance_window_start = models.DateTimeField(null=True, blank=True)
    maintenance_window_end = models.DateTimeField(null=True, blank=True)
    
    # Performance metrics
    current_throughput = models.FloatField(default=0, help_text="Trains per hour")
    avg_delay_minutes = models.FloatField(default=0)
    utilization_percent = models.FloatField(default=0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    
    def __str__(self):
        return self.name
    
    @property
    def is_under_maintenance(self):
        now = timezone.now()
        return (self.maintenance_window_start and self.maintenance_window_end and
                self.maintenance_window_start <= now <= self.maintenance_window_end)
    
    @property
    def weather_impact_factor(self):
        """Returns speed reduction factor based on weather (0.0 to 1.0)"""
        weather_factors = {
            'clear': 1.0,
            'rain': 0.85,
            'fog': 0.6,
            'storm': 0.4
        }
        return weather_factors.get(self.current_weather, 1.0)

class Segment(models.Model):
    section = models.ForeignKey(Section, on_delete=models.CASCADE, related_name='segments')
    name = models.CharField(max_length=200)
    length_km = models.FloatField()
    sequence = models.IntegerField()
    max_speed = models.IntegerField(help_text="Maximum speed in km/h")
    has_siding = models.BooleanField(default=False)
    
    # Real-world constraints
    gradient_percent = models.FloatField(default=0.0, help_text="Track gradient")
    curve_radius_m = models.IntegerField(null=True, blank=True, help_text="Minimum curve radius")
    electrified = models.BooleanField(default=True)
    
    # Dynamic status
    is_blocked = models.BooleanField(default=False)
    block_reason = models.CharField(max_length=200, blank=True)
    current_occupancy = models.IntegerField(default=0)
    max_occupancy = models.IntegerField(default=1)
    
    # Infrastructure details
    platform_count = models.IntegerField(default=0)
    signal_count = models.IntegerField(default=2)
    
    class Meta:
        ordering = ['section', 'sequence']
    
    def __str__(self):
        return f"{self.section.name} - {self.name}"
    
    @property
    def occupancy_percent(self):
        if self.max_occupancy == 0:
            return 0
        return (self.current_occupancy / self.max_occupancy) * 100
    
    @property
    def effective_max_speed(self):
        """Calculate effective max speed considering weather and gradient"""
        base_speed = self.max_speed
        weather_factor = self.section.weather_impact_factor
        gradient_factor = max(0.7, 1.0 - abs(self.gradient_percent) / 100)
        return int(base_speed * weather_factor * gradient_factor)

class Signal(models.Model):
    SIGNAL_STATES = [
        ('green', 'Green - Proceed'),
        ('yellow', 'Yellow - Caution'),
        ('red', 'Red - Stop'),
        ('flashing_yellow', 'Flashing Yellow - Prepare to Stop'),
        ('maintenance', 'Under Maintenance')
    ]
    
    segment = models.ForeignKey(Segment, on_delete=models.CASCADE, related_name='signals')
    name = models.CharField(max_length=50)
    position_km = models.FloatField()
    signal_type = models.CharField(max_length=20, choices=[
        ('main', 'Main Signal'),
        ('distant', 'Distant Signal'),
        ('junction', 'Junction Signal'),
        ('starter', 'Starter Signal'),
        ('advanced_starter', 'Advanced Starter')
    ])
    
    # Dynamic signal state
    current_state = models.CharField(max_length=20, choices=SIGNAL_STATES, default='green')
    last_state_change = models.DateTimeField(auto_now=True)
    
    # Operational data
    controlled_by_train = models.ForeignKey('Train', on_delete=models.SET_NULL, null=True, blank=True)
    auto_control = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.segment} - {self.name} ({self.current_state})"

class Platform(models.Model):
    segment = models.ForeignKey(Segment, on_delete=models.CASCADE, related_name='platforms')
    name = models.CharField(max_length=100)
    length_m = models.IntegerField()
    can_cross = models.BooleanField(default=False)
    
    # Dynamic occupancy
    is_occupied = models.BooleanField(default=False)
    occupied_by_train = models.ForeignKey('Train', on_delete=models.SET_NULL, null=True, blank=True)
    occupation_start = models.DateTimeField(null=True, blank=True)
    expected_departure = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return self.name
    
    @property
    def occupation_duration(self):
        if self.occupation_start:
            return timezone.now() - self.occupation_start
        return None

class Train(models.Model):
    TRAIN_TYPES = [
        ('express', 'Express'),
        ('local', 'Local'),
        ('freight', 'Freight'),
        ('special', 'Special'),
        ('maintenance', 'Maintenance'),
    ]
    
    TRAIN_STATUS = [
        ('scheduled', 'Scheduled'),
        ('running', 'Running'),
        ('delayed', 'Delayed'),
        ('stopped', 'Stopped'),
        ('cancelled', 'Cancelled'),
        ('completed', 'Completed'),
        ('emergency', 'Emergency Stop')
    ]
    
    train_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    train_type = models.CharField(max_length=20, choices=TRAIN_TYPES)
    priority = models.IntegerField(default=3, validators=[MinValueValidator(1), MaxValueValidator(5)])  # 1=highest, 5=lowest
    length_m = models.IntegerField()
    max_speed = models.IntegerField()
    
    # Scheduling
    scheduled_arrival = models.DateTimeField(null=True, blank=True)
    scheduled_departure = models.DateTimeField(null=True, blank=True)
    actual_arrival = models.DateTimeField(null=True, blank=True)
    actual_departure = models.DateTimeField(null=True, blank=True)
    
    # Real-time status
    current_status = models.CharField(max_length=20, choices=TRAIN_STATUS, default='scheduled')
    current_segment = models.ForeignKey(Segment, on_delete=models.SET_NULL, null=True, blank=True)
    current_speed = models.IntegerField(default=0, help_text="Current speed in km/h")
    current_position_km = models.FloatField(default=0.0)
    
    # Operational data
    passenger_count = models.IntegerField(default=0)
    cargo_weight_tons = models.FloatField(default=0.0)
    fuel_level_percent = models.FloatField(default=100.0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    
    # Driver and crew info
    driver_name = models.CharField(max_length=100, blank=True)
    crew_count = models.IntegerField(default=2)
    
    def __str__(self):
        return f"{self.train_id} - {self.name}"
    
    @property
    def current_delay_minutes(self):
        if self.scheduled_arrival and self.actual_arrival:
            return (self.actual_arrival - self.scheduled_arrival).total_seconds() / 60
        elif self.scheduled_arrival and self.current_status == 'running':
            expected_now = timezone.now()
            return max(0, (expected_now - self.scheduled_arrival).total_seconds() / 60)
        return 0
    
    @property
    def estimated_completion_time(self):
        if self.current_segment and self.current_speed > 0:
            remaining_distance = self.current_segment.length_km - self.current_position_km
            time_hours = remaining_distance / max(self.current_speed, 1)
            return timezone.now() + timedelta(hours=time_hours)
        return None

class TrainEvent(models.Model):
    EVENT_TYPES = [
        ('enter_section', 'Enter Section'),
        ('enter_segment', 'Enter Segment'),
        ('pass_signal', 'Pass Signal'),
        ('stop', 'Stop'),
        ('depart', 'Depart'),
        ('clear_segment', 'Clear Segment'),
        ('clear_section', 'Clear Section'),
        ('emergency_stop', 'Emergency Stop'),
        ('speed_restriction', 'Speed Restriction'),
        ('maintenance_halt', 'Maintenance Halt'),
        ('weather_delay', 'Weather Delay'),
        ('fuel_stop', 'Fuel Stop'),
        ('crew_change', 'Crew Change')
    ]
    
    train = models.ForeignKey(Train, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    segment = models.ForeignKey(Segment, on_delete=models.SET_NULL, null=True, blank=True)
    signal = models.ForeignKey(Signal, on_delete=models.SET_NULL, null=True, blank=True)
    platform = models.ForeignKey(Platform, on_delete=models.SET_NULL, null=True, blank=True)
    details = models.JSONField(default=dict)
    
    # Event-specific data
    speed_at_event = models.IntegerField(default=0)
    position_km = models.FloatField(default=0.0)
    delay_minutes = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.train} - {self.event_type} at {self.timestamp}"

class ResourceLock(models.Model):
    resource_type = models.CharField(max_length=20, choices=[
        ('segment', 'Segment'),
        ('signal', 'Signal'),
        ('platform', 'Platform')
    ])
    resource_id = models.IntegerField()
    train = models.ForeignKey(Train, on_delete=models.CASCADE)
    locked_at = models.DateTimeField(auto_now_add=True)
    released_at = models.DateTimeField(null=True, blank=True)
    lock_reason = models.CharField(max_length=200, default='Normal operation')
    
    def __str__(self):
        return f"{self.train} locked {self.resource_type} {self.resource_id}"
    
    @property
    def lock_duration(self):
        end_time = self.released_at or timezone.now()
        return end_time - self.locked_at

class Decision(models.Model):
    DECISION_TYPES = [
        ('precedence', 'Train Precedence'),
        ('crossing', 'Crossing Priority'),
        ('hold', 'Hold Train'),
        ('reroute', 'Reroute Train'),
        ('speed_limit', 'Speed Restriction'),
        ('emergency', 'Emergency Action'),
        ('maintenance', 'Maintenance Window'),
        ('platform_change', 'Platform Change')
    ]
    
    CONFIDENCE_LEVELS = [
        ('very_high', 'Very High (>90%)'),
        ('high', 'High (80-90%)'),
        ('medium', 'Medium (60-80%)'),
        ('low', 'Low (40-60%)'),
        ('very_low', 'Very Low (<40%)')
    ]
    
    timestamp = models.DateTimeField(auto_now_add=True)
    decision_type = models.CharField(max_length=30, choices=DECISION_TYPES)
    trains_involved = models.ManyToManyField(Train)
    
    # Decision details
    recommended_action = models.TextField()
    explanation = models.TextField()
    confidence_level = models.CharField(max_length=20, choices=CONFIDENCE_LEVELS, default='medium')
    expected_benefit = models.TextField(blank=True, help_text="Expected time savings, throughput improvement, etc.")
    
    # Implementation
    is_implemented = models.BooleanField(default=False)
    implemented_at = models.DateTimeField(null=True, blank=True)
    manual_override = models.BooleanField(default=False)
    override_reason = models.TextField(blank=True)
    
    # Algorithm source
    algorithm_used = models.CharField(max_length=50, default='heuristic', 
                                    choices=[('heuristic', 'Heuristic'), ('ilp', 'ILP'), ('ml', 'Machine Learning')])
    computation_time_ms = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.decision_type} decision at {self.timestamp}"

class KPIRecord(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    section = models.ForeignKey(Section, on_delete=models.CASCADE)
    
    # Performance metrics
    throughput_trains = models.IntegerField()  # trains per hour
    avg_delay_minutes = models.FloatField()
    punctuality_percent = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    utilization_percent = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    
    # Additional KPIs
    avg_speed_kmh = models.FloatField(default=0.0)
    fuel_efficiency = models.FloatField(default=0.0, help_text="Fuel consumed per km")
    passenger_satisfaction = models.FloatField(default=0.0, validators=[MinValueValidator(0), MaxValueValidator(10)])
    safety_incidents = models.IntegerField(default=0)
    
    # Environmental metrics
    co2_emissions_kg = models.FloatField(default=0.0)
    noise_level_db = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"KPI for {self.section} at {self.timestamp}"

# New models for advanced features

class WeatherCondition(models.Model):
    section = models.ForeignKey(Section, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    temperature = models.FloatField()
    humidity_percent = models.FloatField()
    wind_speed_kmh = models.FloatField()
    precipitation_mm = models.FloatField(default=0.0)
    visibility_km = models.FloatField()
    weather_type = models.CharField(max_length=50)
    
    class Meta:
        ordering = ['-timestamp']

class EmergencyEvent(models.Model):
    EVENT_TYPES = [
        ('accident', 'Accident'),
        ('fire', 'Fire'),
        ('medical', 'Medical Emergency'),
        ('security', 'Security Threat'),
        ('equipment_failure', 'Equipment Failure'),
        ('natural_disaster', 'Natural Disaster'),
        ('power_outage', 'Power Outage')
    ]
    
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    section = models.ForeignKey(Section, on_delete=models.CASCADE)
    segment = models.ForeignKey(Segment, on_delete=models.CASCADE, null=True, blank=True)
    affected_trains = models.ManyToManyField(Train, blank=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    severity = models.CharField(max_length=20, choices=[
        ('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('critical', 'Critical')
    ])
    
    description = models.TextField()
    response_actions = models.TextField(blank=True)
    
    @property
    def is_active(self):
        return self.resolved_at is None
    
    def __str__(self):
        return f"{self.event_type} - {self.severity} at {self.section}"

class PredictionModel(models.Model):
    """Store ML model predictions for various metrics"""
    section = models.ForeignKey(Section, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    prediction_type = models.CharField(max_length=50, choices=[
        ('delay', 'Delay Prediction'),
        ('throughput', 'Throughput Prediction'),
        ('congestion', 'Congestion Prediction'),
        ('maintenance', 'Maintenance Prediction')
    ])
    
    predicted_value = models.FloatField()
    confidence_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    prediction_horizon_hours = models.IntegerField(default=1)
    model_version = models.CharField(max_length=50, default='v1.0')
    
    class Meta:
        ordering = ['-timestamp']
