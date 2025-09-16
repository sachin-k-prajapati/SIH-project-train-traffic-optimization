from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from core.models import Train, TrainEvent, Decision, KPIRecord, Section
from django.utils import timezone
from datetime import timedelta
from api.serializers import (
    TrainSerializer, TrainEventSerializer, 
    DecisionSerializer, KPISerializer, SectionSerializer
)
from decision.engines.heuristic_engine import HeuristicEngine

@api_view(['POST'])
def train_events(request):
    """Ingest train position and status events"""
    serializer = TrainEventSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def disruptions(request):
    """Ingest incident events"""
    # Implementation would handle disruption events
    return Response({"status": "disruption recorded"}, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def recommendations(request):
    """Return ranked decisions"""
    section_id = request.GET.get('section_id')
    if not section_id:
        return Response({"error": "section_id required"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        section = Section.objects.get(id=section_id)
        active_trains = Train.objects.filter(
            trainevent__event_type='enter_section',
            trainevent__timestamp__gte=timezone.now() - timedelta(hours=1)
        ).distinct()
        
        engine = HeuristicEngine(section)
        decisions = engine.decide_precedence(active_trains, timezone.now())
        
        return Response({
            'section': section.name,
            'active_trains': TrainSerializer(active_trains, many=True).data,
            'recommendations': decisions
        })
    except Section.DoesNotExist:
        return Response({"error": "Section not found"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
def decision_override(request):
    """Log manual override of a decision"""
    decision_id = request.data.get('decision_id')
    reason = request.data.get('reason')
    
    try:
        decision = Decision.objects.get(id=decision_id)
        decision.manual_override = True
        decision.override_reason = reason
        decision.save()
        
        return Response({"status": "override recorded"})
    except Decision.DoesNotExist:
        return Response({"error": "Decision not found"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def kpis(request):
    """Return section KPIs"""
    section_id = request.GET.get('section_id')
    hours = int(request.GET.get('hours', 24))
    
    kpis = KPIRecord.objects.filter(
        section_id=section_id,
        timestamp__gte=timezone.now() - timedelta(hours=hours)
    )
    
    serializer = KPISerializer(kpis, many=True)
    return Response(serializer.data)

class SectionList(generics.ListAPIView):
    queryset = Section.objects.all()
    serializer_class = SectionSerializer

class TrainList(generics.ListAPIView):
    queryset = Train.objects.all()
    serializer_class = TrainSerializer

class DecisionList(generics.ListAPIView):
    queryset = Decision.objects.all()
    serializer_class = DecisionSerializer