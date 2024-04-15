
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

# 분석 구 버전
@api_viuw(['POST'])
def analysis(request):
  # Validation
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
  # Validation
  if request.method.lower() != "post" :
    return HttpResponse("POST 요청만 가능합니다.", status=405)
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

# df의 column별 타입 체크
@api_viuw(['POST'])
def column_type_check(request) :

  if request.method.lower() != "post" :
    return HttpResponse("POST 요청만 가능합니다.", status=405)
  if 'file' not in request.FILES:
    print("파일이 업로드되지 않았습니다.")
    return HttpResponse("파일이 업로드되지 않았습니다.", status=400) 
  
  file = request.FILES['file']
  file_name = file.name
  extension = file_name.split('.')[-1].lower()
  

  if extension == 'csv':
    df = pd.read_csv(file)
  elif extension in ['xls', 'xlsx']:
    df = pd.read_excel(file)
  
  callback_url = request.POST['callbackUrl']
  redis_key = request.POST['redisKey']


  thread = Thread(target=column_type_check_callback, args=(df, callback_url, redis_key))
  thread.start()
  
  return JsonResponse({"message": "Type Check를 시작했습니다."}, status=202) 
  
  
def detect_column_types(df):
    dtos = []  # 'dtos' 리스트 초기화
    
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        
        # 모든 값에 대해 데이터 타입 추론
        inferred_types = {infer_data_type(value) for value in unique_values}
        
        # 최종 데이터 타입 결정
        if "datetime" in inferred_types:
            final_type = "datetime"
        elif "float" in inferred_types:
            final_type = "float"
        elif "int" in inferred_types:
            final_type = "int"
        else:
            final_type = "str"
        
        # 각 열에 대한 정보를 'dtos' 리스트에 딕셔너리 형태로 추가
        dtos.append({
            "columnName": column,
            "dataType": final_type
        })
    return dtos# 'dtos'를 포함한 딕셔너리 반환

def infer_data_type(value):
    try:
        int_value = int(value)
        float_value = float(value)
        if float_value - int_value == 0:
            return 'int'
        else:
            return 'float'
    except ValueError:
        try:
            datetime.strptime(str(value), '%Y%m%d')
            return 'datetime'
        except ValueError:
            return 'str'

def column_type_check_callback(df, callback_url, redis_key) :
  print(callback_url)
  r = redis.Redis(host='localhost', port=6379, db=0)
  try :
    
    redis_result = r.hgetall(redis_key)
    token = redis_result[b'token'].decode('utf-8')
    if not token:
      print("Token not found")
      return
    result = detect_column_types(df)
    # oca.run() 실행 및 결과 처리
    print(result)

    payload = {
    "token": token,
    "dtos": result
    }
    # Convert the payload to a JSON string
    
    # 결과를 callback URL로 POST 요청
    response = requests.post(callback_url, json=payload)
    print(f"Callback POST status: {response.status_code}")
  finally :
    r.close()
    r.connection_pool.disconnect()  
   
        
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

