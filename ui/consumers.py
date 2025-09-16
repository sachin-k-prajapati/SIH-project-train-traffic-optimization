import json
import asyncio
from datetime import datetime, timedelta
from django.utils import timezone
from django.db import models
from core.models import Section, Train, Segment, WeatherCondition, EmergencyEvent
from simulator.engine import RealTimeRailwaySimulator
from decision.engines.ilp_engine import AdvancedILPEngine
import logging
import random

# Simulated WebSocket consumer for development
# In production, would use Django Channels
class AsyncWebsocketConsumer:
    """Placeholder for Django Channels AsyncWebsocketConsumer"""
    def __init__(self, *args, **kwargs):
        self.scope = {}
        self.channel_layer = None
        self.channel_name = "dashboard"
    
    async def accept(self):
        pass
    
    async def send(self, text_data):
        pass

def database_sync_to_async(func):
    """Placeholder for Django Channels database_sync_to_async"""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

logger = logging.getLogger(__name__)

class EnhancedDashboardConsumer(AsyncWebsocketConsumer):
    """
    Enhanced WebSocket consumer for real-time railway control dashboard
    Provides live train tracking, network visualization, and real-time updates
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_id = None
        self.simulator = None
        self.optimizer = None
        self.update_task = None
        self.train_positions = {}
        self.station_status = {}
        
    async def connect(self):
        """Accept WebSocket connection and start real-time updates"""
        self.section_id = self.scope['url_route']['kwargs'].get('section_id', 1)
        
        # Initialize train positions and station status
        await self.initialize_network_state()
        
        await self.accept()
        logger.info(f"WebSocket connected for section {self.section_id}")
        
        # Start real-time update task
        self.update_task = asyncio.create_task(self.send_periodic_updates())
        
    async def disconnect(self, close_code):
        """Clean up when WebSocket disconnects"""
        if self.update_task:
            self.update_task.cancel()
        logger.info(f"WebSocket disconnected for section {self.section_id}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'get_train_positions':
                await self.send_train_positions()
            elif message_type == 'get_station_status':
                station_id = data.get('station_id')
                await self.send_station_status(station_id)
            elif message_type == 'optimize_routes':
                await self.handle_route_optimization()
            elif message_type == 'update_train':
                await self.handle_train_update(data)
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from WebSocket")
    
    async def initialize_network_state(self):
        """Initialize the network state with stations and trains"""
        # Station network (matches your design)
        self.stations = {
            'A': {'name': 'Mumbai Central', 'x': 100, 'y': 200, 'platforms': 3},
            'B': {'name': 'Dadar Junction', 'x': 300, 'y': 200, 'platforms': 4},
            'C': {'name': 'Kurla', 'x': 500, 'y': 150, 'platforms': 2},
            'D': {'name': 'Thane', 'x': 700, 'y': 150, 'platforms': 3},
            'E': {'name': 'Ghatkopar', 'x': 500, 'y': 250, 'platforms': 2},
            'F': {'name': 'Mulund', 'x': 700, 'y': 250, 'platforms': 2},
        }
        
        # Initialize train positions (12-18 trains as requested)
        self.train_positions = {
            f'T{str(i).zfill(3)}': {
                'id': f'T{str(i).zfill(3)}',
                'name': f'Train {str(i).zfill(3)}',
                'x': random.randint(100, 700),
                'y': random.randint(150, 250),
                'status': random.choice(['moving', 'stopped', 'boarding']),
                'route': self.generate_random_route(),
                'speed': random.randint(40, 120),
                'delay': random.randint(0, 15),
                'destination': random.choice(list(self.stations.keys()))
            }
            for i in range(1, 16)  # 15 trains
        }
        
        # Initialize station status
        for station_id, station_data in self.stations.items():
            self.station_status[station_id] = {
                'platforms': [
                    {
                        'number': i + 1,
                        'status': random.choice(['free', 'occupied', 'maintenance']),
                        'train_id': f'T{str(random.randint(1, 15)).zfill(3)}' if random.random() > 0.6 else None,
                        'direction': random.choice(['northbound', 'southbound', 'eastbound', 'westbound'])
                    }
                    for i in range(station_data['platforms'])
                ],
                'incoming_trains': self.generate_incoming_trains(station_id)
            }
    
    def generate_random_route(self):
        """Generate a random route between stations"""
        stations = list(self.stations.keys())
        start = random.choice(stations)
        end = random.choice([s for s in stations if s != start])
        return f"{start}-{end}"
    
    def generate_incoming_trains(self, station_id):
        """Generate incoming train schedule for a station"""
        trains = []
        for i in range(random.randint(2, 5)):
            eta_minutes = random.randint(5, 60)
            trains.append({
                'train_id': f'T{str(random.randint(1, 15)).zfill(3)}',
                'eta': (datetime.now() + timedelta(minutes=eta_minutes)).strftime('%H:%M'),
                'from_station': random.choice([k for k in self.stations.keys() if k != station_id]),
                'status': random.choice(['ontime', 'delayed']) if random.random() > 0.3 else 'ontime',
                'delay': random.randint(1, 10) if random.random() > 0.7 else 0
            })
        return sorted(trains, key=lambda x: x['eta'])
    
    async def send_periodic_updates(self):
        """Send periodic updates every 5 seconds"""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                # Update train positions
                await self.update_train_positions()
                
                # Send all updates
                await self.send_network_update()
                await self.send_kpi_update()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
    
    async def update_train_positions(self):
        """Update train positions for animation"""
        for train_id, train_data in self.train_positions.items():
            if train_data['status'] == 'moving':
                # Simple movement simulation
                train_data['x'] += random.randint(-20, 20)
                train_data['y'] += random.randint(-10, 10)
                
                # Keep trains within bounds
                train_data['x'] = max(100, min(700, train_data['x']))
                train_data['y'] = max(150, min(250, train_data['y']))
                
                # Occasionally change status
                if random.random() > 0.95:
                    train_data['status'] = random.choice(['moving', 'stopped', 'boarding'])
    
    async def send_network_update(self):
        """Send network state update"""
        update_data = {
            'type': 'network_update',
            'timestamp': datetime.now().isoformat(),
            'trains': list(self.train_positions.values()),
            'station_status': self.station_status
        }
        
        await self.send(text_data=json.dumps(update_data))
    
    async def send_kpi_update(self):
        """Send KPI updates"""
        kpi_data = {
            'type': 'kpi_update',
            'timestamp': datetime.now().isoformat(),
            'kpis': {
                'active_trains': len([t for t in self.train_positions.values() if t['status'] != 'stopped']),
                'avg_delay': sum(t['delay'] for t in self.train_positions.values()) / len(self.train_positions),
                'throughput': random.randint(25, 40),
                'punctuality': random.randint(75, 95),
                'alerts': random.randint(0, 3)
            }
        }
        
        await self.send(text_data=json.dumps(kpi_data))
    
    async def send_train_positions(self):
        """Send current train positions"""
        position_data = {
            'type': 'train_positions',
            'timestamp': datetime.now().isoformat(),
            'trains': list(self.train_positions.values())
        }
        
        await self.send(text_data=json.dumps(position_data))
    
    async def send_station_status(self, station_id):
        """Send detailed station status"""
        if station_id in self.station_status:
            status_data = {
                'type': 'station_status',
                'timestamp': datetime.now().isoformat(),
                'station_id': station_id,
                'station_name': self.stations[station_id]['name'],
                'status': self.station_status[station_id]
            }
            
            await self.send(text_data=json.dumps(status_data))
    
    async def handle_route_optimization(self):
        """Handle route optimization request"""
        # Simulate optimization calculation
        await asyncio.sleep(1)  # Simulate processing time
        
        optimization_result = {
            'type': 'optimization_result',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'improved_routes': [
                {
                    'train_id': 'T001',
                    'original_route': 'A-B-C',
                    'optimized_route': 'A-E-F',
                    'time_saved': 12
                },
                {
                    'train_id': 'T005',
                    'original_route': 'B-C-D',
                    'optimized_route': 'B-D',
                    'time_saved': 8
                }
            ],
            'total_delay_reduction': 20,
            'efficiency_improvement': 15.5
        }
        
        await self.send(text_data=json.dumps(optimization_result))
    
    async def handle_train_update(self, data):
        """Handle manual train updates"""
        train_id = data.get('train_id')
        if train_id in self.train_positions:
            updates = data.get('updates', {})
            self.train_positions[train_id].update(updates)
            
            # Send confirmation
            response = {
                'type': 'train_update_response',
                'timestamp': datetime.now().isoformat(),
                'train_id': train_id,
                'status': 'updated',
                'new_data': self.train_positions[train_id]
            }
            
            await self.send(text_data=json.dumps(response))
    """
    WebSocket consumer for real-time dashboard updates
    Provides live train tracking, KPI updates, and optimization results
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_id = None
        self.simulator = None
        self.optimizer = None
        self.update_task = None
        
    async def connect(self):
        """Accept WebSocket connection and start real-time updates"""
        self.section_id = self.scope['url_route']['kwargs'].get('section_id', 1)
        
        # Join dashboard group
        self.dashboard_group = f'dashboard_{self.section_id}'
        await self.channel_layer.group_add(
            self.dashboard_group,
            self.channel_name
        )
        
        await self.accept()
        
        # Initialize simulator and optimizer
        await self.initialize_systems()
        
        # Start periodic updates
        self.update_task = asyncio.create_task(self.periodic_updates())
        
        # Send initial data
        await self.send_initial_data()
        
        logger.info(f"Dashboard WebSocket connected for section {self.section_id}")
    
    async def disconnect(self, close_code):
        """Clean up on disconnect"""
        # Cancel update task
        if self.update_task:
            self.update_task.cancel()
        
        # Leave dashboard group
        await self.channel_layer.group_discard(
            self.dashboard_group,
            self.channel_name
        )
        
        logger.info(f"Dashboard WebSocket disconnected (code: {close_code})")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'request_update':
                await self.send_full_update()
            elif message_type == 'run_optimization':
                await self.run_optimization()
            elif message_type == 'emergency_stop':
                await self.handle_emergency_stop()
            elif message_type == 'update_weather':
                await self.update_weather(data.get('weather'))
            elif message_type == 'train_command':
                await self.handle_train_command(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    @database_sync_to_async
    def initialize_systems(self):
        """Initialize simulation and optimization systems"""
        try:
            section = Section.objects.get(id=self.section_id)
            self.simulator = RealTimeRailwaySimulator(section)
            self.optimizer = AdvancedILPEngine(section)
            return True
        except Section.DoesNotExist:
            logger.error(f"Section {self.section_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error initializing systems: {e}")
            return False
    
    async def periodic_updates(self):
        """Send periodic updates to connected clients"""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                await self.send_kpi_update()
                await self.send_train_updates()
                await self.check_alerts()
                
                # Run optimization every 30 seconds
                if datetime.now().second % 30 == 0:
                    await self.run_background_optimization()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def send_initial_data(self):
        """Send initial dashboard data"""
        initial_data = await self.get_initial_data()
        await self.send(text_data=json.dumps({
            'type': 'initial_data',
            'data': initial_data
        }))
    
    @database_sync_to_async
    def get_initial_data(self):
        """Get initial data for dashboard"""
        try:
            section = Section.objects.get(id=self.section_id)
            active_trains = Train.objects.filter(
                current_section=section,
                status__in=['running', 'delayed', 'approaching']
            )
            
            # Calculate KPIs
            total_trains = active_trains.count()
            delayed_trains = active_trains.filter(
                current_delay_minutes__gt=5
            ).count()
            
            punctuality = ((total_trains - delayed_trains) / total_trains * 100) if total_trains > 0 else 100
            avg_delay = active_trains.aggregate(
                avg_delay=models.Avg('current_delay_minutes')
            )['avg_delay'] or 0
            
            # Get weather data
            weather = WeatherCondition.objects.filter(
                section=section,
                timestamp__gte=timezone.now() - timedelta(hours=1)
            ).order_by('-timestamp').first()
            
            return {
                'section': {
                    'id': section.id,
                    'name': section.name,
                    'capacity': section.capacity
                },
                'kpis': {
                    'active_trains': total_trains,
                    'punctuality': round(punctuality, 1),
                    'avg_delay': round(avg_delay, 1),
                    'throughput': round(total_trains * 0.8, 1),  # Simulated
                    'efficiency': round(85 + (punctuality - 90) * 2, 1),  # Calculated
                    'utilization': round(total_trains / section.capacity * 100, 1)
                },
                'trains': [
                    {
                        'id': train.id,
                        'train_id': train.train_id,
                        'type': train.train_type,
                        'status': self.get_train_status(train),
                        'current_speed': train.current_speed,
                        'delay_minutes': train.current_delay_minutes,
                        'latitude': float(train.current_latitude) if train.current_latitude else 0,
                        'longitude': float(train.current_longitude) if train.current_longitude else 0,
                        'next_station': self.get_next_station(train)
                    }
                    for train in active_trains
                ],
                'weather': {
                    'temperature': weather.temperature if weather else 25,
                    'condition': weather.condition if weather else 'clear',
                    'impact_factor': float(weather.visibility_factor) if weather else 1.0
                } if weather else None,
                'alerts': self.get_recent_alerts()
            }
            
        except Exception as e:
            logger.error(f"Error getting initial data: {e}")
            return {}
    
    def get_train_status(self, train):
        """Determine train status based on delay"""
        if train.current_delay_minutes > 5:
            return 'delayed'
        elif train.current_delay_minutes < -2:
            return 'early'
        else:
            return 'on-time'
    
    def get_next_station(self, train):
        """Get next station for train"""
        # Simplified - would normally calculate based on current position
        return "Next Station"
    
    def get_recent_alerts(self):
        """Get recent alerts and emergencies"""
        # Placeholder for alert system
        return []
    
    async def send_kpi_update(self):
        """Send updated KPIs to clients"""
        kpis = await self.calculate_current_kpis()
        await self.send(text_data=json.dumps({
            'type': 'kpi_update',
            'kpis': kpis
        }))
    
    @database_sync_to_async
    def calculate_current_kpis(self):
        """Calculate current KPIs"""
        try:
            section = Section.objects.get(id=self.section_id)
            active_trains = Train.objects.filter(
                current_section=section,
                status__in=['running', 'delayed', 'approaching']
            )
            
            total_trains = active_trains.count()
            if total_trains == 0:
                return {
                    'active_trains': 0,
                    'punctuality': 100,
                    'avg_delay': 0,
                    'throughput': 0,
                    'efficiency': 100,
                    'utilization': 0
                }
            
            on_time_trains = active_trains.filter(
                current_delay_minutes__lte=5
            ).count()
            
            punctuality = (on_time_trains / total_trains * 100)
            avg_delay = active_trains.aggregate(
                avg_delay=models.Avg('current_delay_minutes')
            )['avg_delay'] or 0
            
            # Simulated throughput (trains per hour)
            current_hour = timezone.now().hour
            throughput = total_trains * (0.8 + 0.4 * abs(12 - current_hour) / 12)
            
            # Efficiency based on punctuality and speed
            avg_speed = active_trains.aggregate(
                avg_speed=models.Avg('current_speed')
            )['avg_speed'] or 60
            
            efficiency = min(100, (punctuality * 0.6) + (avg_speed / 100 * 40))
            
            utilization = min(100, total_trains / section.capacity * 100)
            
            return {
                'active_trains': total_trains,
                'punctuality': round(punctuality, 1),
                'avg_delay': round(avg_delay, 1),
                'throughput': round(throughput, 1),
                'efficiency': round(efficiency, 1),
                'utilization': round(utilization, 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating KPIs: {e}")
            return {}
    
    async def send_train_updates(self):
        """Send updated train positions and status"""
        trains = await self.get_train_updates()
        for train in trains:
            await self.send(text_data=json.dumps({
                'type': 'train_update',
                'train': train
            }))
    
    @database_sync_to_async
    def get_train_updates(self):
        """Get current train positions and status"""
        try:
            section = Section.objects.get(id=self.section_id)
            active_trains = Train.objects.filter(
                current_section=section,
                status__in=['running', 'delayed', 'approaching']
            )
            
            return [
                {
                    'id': train.id,
                    'train_id': train.train_id,
                    'type': train.train_type,
                    'status': self.get_train_status(train),
                    'current_speed': train.current_speed,
                    'delay_minutes': train.current_delay_minutes,
                    'latitude': float(train.current_latitude) if train.current_latitude else 0,
                    'longitude': float(train.current_longitude) if train.current_longitude else 0,
                    'next_station': self.get_next_station(train),
                    'fuel_level': getattr(train, 'fuel_level', 85),  # Simulated
                    'passenger_count': getattr(train, 'passenger_count', 150)  # Simulated
                }
                for train in active_trains
            ]
            
        except Exception as e:
            logger.error(f"Error getting train updates: {e}")
            return []
    
    async def check_alerts(self):
        """Check for new alerts and send to clients"""
        alerts = await self.get_new_alerts()
        for alert in alerts:
            await self.send(text_data=json.dumps({
                'type': 'alert',
                'alert': alert
            }))
    
    @database_sync_to_async
    def get_new_alerts(self):
        """Get new alerts based on current conditions"""
        alerts = []
        
        try:
            section = Section.objects.get(id=self.section_id)
            active_trains = Train.objects.filter(
                current_section=section,
                status__in=['running', 'delayed', 'approaching']
            )
            
            # Check for high delays
            heavily_delayed = active_trains.filter(
                current_delay_minutes__gt=15
            )
            
            for train in heavily_delayed:
                alerts.append({
                    'message': f'Train {train.train_id} severely delayed',
                    'action': f'Current delay: {train.current_delay_minutes} minutes',
                    'priority': 'high',
                    'train_id': train.train_id,
                    'timestamp': timezone.now().isoformat()
                })
            
            # Check capacity utilization
            utilization = len(active_trains) / section.capacity * 100
            if utilization > 85:
                alerts.append({
                    'message': 'Section approaching capacity limit',
                    'action': f'Current utilization: {utilization:.1f}%',
                    'priority': 'medium',
                    'timestamp': timezone.now().isoformat()
                })
            
            # Check for emergency events
            recent_emergencies = EmergencyEvent.objects.filter(
                section=section,
                created_at__gte=timezone.now() - timedelta(minutes=5),
                status='active'
            )
            
            for emergency in recent_emergencies:
                alerts.append({
                    'message': f'Emergency: {emergency.event_type}',
                    'action': emergency.description,
                    'priority': 'high',
                    'timestamp': emergency.created_at.isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
        
        return alerts
    
    async def run_optimization(self):
        """Run optimization and send results"""
        try:
            result = await self.run_optimization_sync()
            await self.send(text_data=json.dumps({
                'type': 'optimization_result',
                'result': result
            }))
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            await self.send(text_data=json.dumps({
                'type': 'optimization_error',
                'error': str(e)
            }))
    
    @database_sync_to_async
    def run_optimization_sync(self):
        """Run optimization synchronously"""
        try:
            section = Section.objects.get(id=self.section_id)
            active_trains = list(Train.objects.filter(
                current_section=section,
                status__in=['running', 'delayed', 'approaching']
            ))
            
            if not active_trains:
                return {
                    'status': 'no_trains',
                    'message': 'No active trains to optimize'
                }
            
            # Run optimization
            optimizer = AdvancedILPEngine(section)
            result = optimizer.optimize_comprehensive_schedule(active_trains)
            
            return {
                'status': result.get('status', 'unknown'),
                'solve_time': result.get('solve_time', 0),
                'objective_value': result.get('objective_value', 0),
                'kpis': result.get('kpis', {}),
                'recommendations': result.get('recommendations', []),
                'algorithm_confidence': result.get('algorithm_confidence', 'medium'),
                'trains_optimized': len(active_trains)
            }
            
        except Exception as e:
            logger.error(f"Error in optimization sync: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def run_background_optimization(self):
        """Run optimization in background and cache results"""
        try:
            result = await self.run_optimization_sync()
            # Could store results in cache for quick access
            logger.info(f"Background optimization completed: {result.get('status')}")
        except Exception as e:
            logger.error(f"Background optimization failed: {e}")
    
    async def handle_emergency_stop(self):
        """Handle emergency stop command"""
        try:
            affected_trains = await self.emergency_stop_trains()
            await self.send(text_data=json.dumps({
                'type': 'emergency_response',
                'action': 'stop_all',
                'affected_trains': affected_trains,
                'timestamp': timezone.now().isoformat()
            }))
            
            # Broadcast to all clients in group
            await self.channel_layer.group_send(
                self.dashboard_group,
                {
                    'type': 'emergency_alert',
                    'message': 'EMERGENCY STOP ACTIVATED',
                    'action': 'All trains have been signaled to stop',
                    'priority': 'critical'
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling emergency stop: {e}")
    
    @database_sync_to_async
    def emergency_stop_trains(self):
        """Set all trains in section to emergency stop"""
        try:
            section = Section.objects.get(id=self.section_id)
            active_trains = Train.objects.filter(
                current_section=section,
                status__in=['running', 'delayed', 'approaching']
            )
            
            # Update train status
            train_ids = []
            for train in active_trains:
                train.status = 'emergency_stop'
                train.current_speed = 0
                train.save()
                train_ids.append(train.train_id)
            
            # Create emergency event
            EmergencyEvent.objects.create(
                section=section,
                event_type='emergency_stop',
                description='Emergency stop activated via dashboard',
                severity='critical',
                status='active'
            )
            
            return train_ids
            
        except Exception as e:
            logger.error(f"Error stopping trains: {e}")
            return []
    
    async def update_weather(self, weather_data):
        """Update weather conditions"""
        try:
            await self.update_weather_sync(weather_data)
            
            # Broadcast weather update
            await self.channel_layer.group_send(
                self.dashboard_group,
                {
                    'type': 'weather_update',
                    'weather': weather_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating weather: {e}")
    
    @database_sync_to_async
    def update_weather_sync(self, weather_data):
        """Update weather conditions in database"""
        try:
            section = Section.objects.get(id=self.section_id)
            
            WeatherCondition.objects.create(
                section=section,
                temperature=weather_data.get('temperature', 25),
                condition=weather_data.get('condition', 'clear'),
                wind_speed=weather_data.get('wind_speed', 5),
                visibility_factor=weather_data.get('impact_factor', 1.0),
                timestamp=timezone.now()
            )
            
            # Update section weather impact
            section.weather_impact_factor = weather_data.get('impact_factor', 1.0)
            section.save()
            
        except Exception as e:
            logger.error(f"Error updating weather sync: {e}")
    
    async def handle_train_command(self, command_data):
        """Handle train-specific commands"""
        try:
            train_id = command_data.get('train_id')
            command = command_data.get('command')
            params = command_data.get('params', {})
            
            result = await self.execute_train_command(train_id, command, params)
            
            await self.send(text_data=json.dumps({
                'type': 'train_command_result',
                'train_id': train_id,
                'command': command,
                'result': result
            }))
            
        except Exception as e:
            logger.error(f"Error handling train command: {e}")
    
    @database_sync_to_async
    def execute_train_command(self, train_id, command, params):
        """Execute command for specific train"""
        try:
            train = Train.objects.get(train_id=train_id)
            
            if command == 'set_speed':
                new_speed = params.get('speed', train.current_speed)
                train.current_speed = min(new_speed, train.max_speed)
                train.save()
                return {'status': 'success', 'new_speed': train.current_speed}
            
            elif command == 'set_priority':
                new_priority = params.get('priority', train.priority)
                train.priority = max(1, min(5, new_priority))
                train.save()
                return {'status': 'success', 'new_priority': train.priority}
            
            elif command == 'delay_train':
                delay_minutes = params.get('delay', 0)
                train.current_delay_minutes += delay_minutes
                train.save()
                return {'status': 'success', 'total_delay': train.current_delay_minutes}
            
            else:
                return {'status': 'error', 'message': f'Unknown command: {command}'}
                
        except Train.DoesNotExist:
            return {'status': 'error', 'message': f'Train {train_id} not found'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def send_full_update(self):
        """Send complete dashboard update"""
        try:
            data = await self.get_initial_data()
            await self.send(text_data=json.dumps({
                'type': 'full_update',
                'data': data
            }))
        except Exception as e:
            logger.error(f"Error sending full update: {e}")
    
    # Group message handlers
    async def emergency_alert(self, event):
        """Handle emergency alert from group"""
        await self.send(text_data=json.dumps({
            'type': 'alert',
            'alert': {
                'message': event['message'],
                'action': event['action'],
                'priority': event['priority'],
                'timestamp': timezone.now().isoformat()
            }
        }))
    
    async def weather_update(self, event):
        """Handle weather update from group"""
        await self.send(text_data=json.dumps({
            'type': 'weather_update',
            'weather': event['weather']
        }))

# WebSocket routing configuration for Django Channels
# Add to your Django routing.py file:

# from django.urls import re_path
# from . import consumers

# websocket_urlpatterns = [
#     re_path(r'ws/dashboard/(?P<section_id>\d+)/$', consumers.DashboardConsumer.as_asgi()),
#     re_path(r'ws/dashboard/$', consumers.DashboardConsumer.as_asgi()),
# ]
