from rest_framework import serializers

# 분석 요청 DTO
class ColumnSerializer(serializers.Serializer):
    columnName = serializers.CharField(max_length=200)

class MetadataSerializer(serializers.Serializer):
    columns = ColumnSerializer(many=True)
    targetColumns = ColumnSerializer(many=True)

class AnalysisRequestSerializer(serializers.Serializer):
    file = serializers.FileField()
    metadata = MetadataSerializer()

