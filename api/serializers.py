from rest_framework import serializers
from core.models import Train, TrainEvent, Decision, KPIRecord, Section

class TrainSerializer(serializers.ModelSerializer):
    class Meta:
        model = Train
        fields = '__all__'

class TrainEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainEvent
        fields = '__all__'

class DecisionSerializer(serializers.ModelSerializer):
    trains_involved = TrainSerializer(many=True, read_only=True)
    
    class Meta:
        model = Decision
        fields = '__all__'

class KPISerializer(serializers.ModelSerializer):
    class Meta:
        model = KPIRecord
        fields = '__all__'

class SectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Section
        fields = '__all__'