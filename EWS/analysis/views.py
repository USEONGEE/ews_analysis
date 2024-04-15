
from django.shortcuts import render
from .models import FileUpload
from django.http import HttpResponse, Http404, HttpResponseRedirect, JsonResponse
from django.core.exceptions import BadRequest
from django.core import serializers
from django.shortcuts import render, get_object_or_404
import pandas as pd
from pycaret.regression import *
import json
from time import sleep
import requests
from threading import Thread
from .model.main import *
import redis
import pandas as pd
from datetime import datetime
# api_view 임포트
from rest_framework.decorators import api_view as api_viuw
from .serializers import AnalysisRequestSerializer

# 분석 구 버전
@api_viuw(['POST'])
def analysis(request):
  # Validation
  serializer = AnalysisRequestSerializer(data=request.data)
  if not serializer.is_valid():
    return JsonResponse(serializer.errors, status=400)
  metadata = serializer.validated_data
  
  # 파일 확장자 일기
  file = request.FILES['file']
  file_name = file.name
  extension = file_name.split('.')[-1].lower()
  print(metadata)
  
  if metadata['all'] == True : 
  # 파일 확장자에 따른 처리
    if extension == 'csv':
      df = pd.read_csv(file)
    elif extension in ['xls', 'xlsx']:
      df = pd.read_excel(file)
  else:
    columns = [item['columnName'] for item in metadata['columns']]
    if extension == 'csv':
      df = pd.read_csv(file, usecols=columns)
    elif extension in ['xls', 'xlsx']:
      df = pd.read_excel(file, usecols=columns)
  
  oca = OCA(df, metadata['targetColumns'][0]['columnName'])
  result = oca.run()
  
  return JsonResponse(result, status=200)

# 분석 최신 버전
@api_viuw(['POST'])
def analysisV2(request):
  if 'file' not in request.FILES:
    print("파일이 업로드되지 않았습니다.")
    return HttpResponse("파일이 업로드되지 않았습니다.", status=400)
  
  try:  
    metadata_str = request.POST.get('metadata', '{}')  # 기본값으로 빈 JSON 문자열 설정
    metadata = json.loads(metadata_str)
  except json.JSONDecodeError:
    print("metadata 형식이 잘못되었습니다.")
    return HttpResponse("metadata 형식이 잘못되었습니다.", status=400)
  
  
  # 파일 확장자 일기
  file = request.FILES['file']
  file_name = file.name
  extension = file_name.split('.')[-1].lower()
  print(metadata)
  

  columns = [item['columnName'] for item in metadata['columns']]
  if extension == 'csv':
    df = pd.read_csv(file, usecols=columns)
  elif extension in ['xls', 'xlsx']:
    df = pd.read_excel(file, usecols=columns)

  oca = OCA(df, metadata['targetColumns'][0]['columnName'])
  thread = Thread(target=analysisV2_callback, args=(oca, metadata['callbackUrl'], metadata['redisKey']))
  thread.start()

  # 클라이언트에게 즉시 응답 반환
  return JsonResponse({"message": "분석을 시작했습니다."}, status=202)

def analysisV2_callback(oca, callback_url, redis_key):
  r = redis.Redis(host='localhost', port=6379, db=0)
  try :
    redis_result = r.hgetall(redis_key)
    print(redis_result)
    token = redis_result[b'token'].decode('utf-8')
    print(token)
    
    if not token:
      print("Token not found")
      return

    # oca.run() 실행 및 결과 처리
    analysis_result = oca.run()  # oca.run()이 동기적으로 실행된다고 가정
    print(analysis_result.keys())
    payload = {
    "token": token,
    "body": analysis_result
    }
    # Convert the payload to a JSON string
    
    # 결과를 callback URL로 POST 요청
    print(callback_url)
    response = requests.post(callback_url, json=payload)
    print(f"Callback POST status: {response.status_code}")
  finally :
    r.close()
    r.connection_pool.disconnect()  
