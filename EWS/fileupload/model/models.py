## pycaret
import pycaret.regression as reg
import pycaret.classification as cls
import pandas as pd
## LSTM

from .dashboard import *
import statsmodels.formula.api as smf

# visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from explainerdashboard import ExplainerDashboard, RegressionExplainer, ClassifierExplainer, ExplainerHub
import zipfile
import os
import shutil



# def lstm_train_test(df1, df2, bank_lists, target, timestep=1):
#     '''
#     df1 : DataFrame, (essential df1)
#     df2 : DataFrame, (essential df2)
#     bank_lists : list (e.g. ["BANK #1"]
#     target : str
#     timestep : int
#     '''

#     ## feature or label
#     date = select_bank(df1, bank_lists, "delete")['date']  # 날짜 추출
#     df = select_bank(df2, bank_lists, "delete")  # 실제로 돌릴 데이터
#     all_col = df.columns.to_list()
#     feature = all_col.copy()
#     # feature.remove(target)  # target 제외한 변수로 돌리는 과정

#     ## MinMaxScaler 정규화
#     scaler_target = MinMaxScaler(feature_range=(0, 1))
#     scaler_feature = MinMaxScaler(feature_range=(0, 1))
#     df[target] = scaler_target.fit_transform(df[target].values.reshape(-1, 1))
#     df[feature] = scaler_feature.fit_transform(df[feature])

#     num_feature = len(feature)
#     target_list = [target]

#     train_set, test_set = train_test_split(df, test_size=0.4, shuffle=False, random_state=42)
#     val_set, test_set = train_test_split(test_set, test_size=0.5, shuffle=False, random_state=42)
#     print(type(train_set))
#     print(train_set.to_numpy().shape)

#     x_train, y_train = datasetcreation(train_set, feature, target_list, timestep=timestep)
#     x_val, y_val = datasetcreation(val_set, feature, target_list, timestep=timestep)
#     x_test, y_test = datasetcreation(test_set, feature, target_list, timestep=timestep)

#     print(f'# of train : {len(x_train)} \n# of val : {len(x_val)} \n# of test : {len(x_test)}')
#     print(x_train, y_train)
#     print(x_train.shape, y_train.shape)
#     return x_train, x_val, x_test, y_train, y_val, y_test, scaler_feature, scaler_target, date



## analysis

def ml_create_model(base, models):
    list_models = []
    for model in models:
        list_models.append(base.create_model(model, cross_validation=True) )
    
    return list_models

def auto_ml_reg(df, target):
    
    reg_model = reg.setup(df, target = target, session_id=42, normalize_method='minmax', fold=5, )
    
    
    reg_avail_model = reg_model.models().index.to_list()
    
    try:
        reg_avail_model.remove('lightgbm')
    except ValueError:
        pass
 

    return reg_model, reg_avail_model


def make_reg_dashboards(base, models):
    regression_models = {'lr' : 'Linear Regression',
    'lasso' : 'Lasso Regression',
    'ridge' : 'Ridge Regression',
    'en' : 'Elastic Net',
    'lar' : 'Least Angle Regression',
    'llar' :   'Lasso Least Angle Regression',
    'omp' : 'Orthogonal Matching Pursuit',
    'br' : 'Bayesian Ridge',
    'ard' :  'Automatic Relevance Determination',
    'par' : 'Passive Aggressive Regressor',
    'ransac' : 'Random Sample Consensus',
    'tr' : 'TheilSen Regressor',
    'huber' : 'Huber Regressor',
    'kr' : 'Kernel Ridge',
    'svm' : 'Support Vector Regression',
    'knn' :  'K Neighbors Regressor',
    'dt' : 'Decision Tree Regressor',
    'rf' : 'Random Forest Regressor',
    'et' :  'Extra Trees Regressor',
    'ada' :  'AdaBoost Regressor',
    'gbr' : 'Gradient Boosting Regressor',
    'mlp' : 'MLP Regressor',
    'xgboost' : 'Extreme Gradient Boosting',
    'lightgbm' :  'Light Gradient Boosting Machine',
    'catboost' : 'CatBoost Regressor',
    'dummy' : 'Dummy variable'}

    reg_description = {
    'lr': 'Linear Regression은 선형 모델을 사용하여 종속 변수와 독립 변수 간의 관계를 모델링하는 알고리즘입니다. 최소제곱법을 사용하여 선형 함수의 계수를 추정합니다.',
    'lasso': 'Lasso Regression은 L1 정규화를 사용하여 선형 모델의 계수를 추정하는 알고리즘입니다. L1 정규화는 일부 계수를 정확히 0으로 만들어 특성 선택 효과를 제공합니다.',
    'ridge': 'Ridge Regression은 L2 정규화를 사용하여 선형 모델의 계수를 추정하는 알고리즘입니다. L2 정규화는 계수의 크기를 제한하여 과적합을 방지합니다.',
    'en': 'Elastic Net은 L1과 L2 정규화를 결합한 선형 모델입니다. Lasso와 Ridge의 장점을 결합하여 특성 선택과 계수 축소를 동시에 수행합니다.',
    'lar': 'Least Angle Regression은 Forward Stepwise Regression의 변형으로, 각 단계에서 상관관계가 가장 높은 변수를 선택하여 모델을 구축합니다.',
    'llar': 'Lasso Least Angle Regression은 Least Angle Regression에 L1 정규화를 적용한 알고리즘입니다. Lasso 정규화를 사용하여 일부 계수를 0으로 만듭니다.',
    'omp': 'Orthogonal Matching Pursuit은 Forward Stepwise Regression의 변형으로, 잔차와 가장 상관관계가 높은 변수를 선택하여 모델을 구축합니다.',
    'br': 'Bayesian Ridge Regression은 베이지안 추론을 사용하여 선형 모델의 계수를 추정하는 알고리즘입니다. 사전 확률 분포를 도입하여 regularization을 수행합니다.',
    'ard': 'Automatic Relevance Determination은 Sparse Bayesian Learning의 한 종류로, 관련성이 낮은 입력 변수를 자동으로 식별하고 제거하는 기술입니다.',
    'par': 'Passive Aggressive Regressor는 온라인 학습 알고리즘의 일종으로, 각 학습 단계에서 손실 함수를 최소화하도록 모델을 업데이트합니다.',
    'ransac': 'Random Sample Consensus는 이상치에 강건한 회귀 알고리즘입니다. 데이터에서 무작위로 샘플을 선택하여 모델을 학습하고, 콘센서스를 최대화하는 모델을 선택합니다.',
    'tr': 'TheilSen Regressor는 중앙값 기반의 비모수 회귀 알고리즘입니다. 데이터의 중앙값을 사용하여 이상치에 강건한 회귀선을 추정합니다.',
    'huber': 'Huber Regressor는 Huber 손실 함수를 사용하여 이상치에 강건한 선형 회귀를 수행하는 알고리즘입니다.',
    'kr': 'Kernel Ridge Regression은 비선형 커널 함수를 사용하여 데이터를 고차원 공간에 매핑한 후 Ridge Regression을 적용하는 알고리즘입니다.',
    'svm': 'Support Vector Regression은 Support Vector Machine을 회귀 문제에 적용한 알고리즘입니다. 마진을 최대화하는 초평면을 찾아 회귀선을 추정합니다.',
    'knn': 'K Neighbors Regressor는 k-최근접 이웃 알고리즘을 사용하여 회귀를 수행합니다. 새로운 데이터 포인트의 값을 예측할 때, 주변의 k개 이웃 데이터 포인트들의 값을 평균하여 예측합니다.',
    'dt': 'Decision Tree Regressor는 트리 구조를 사용하여 회귀 모델을 학습하는 알고리즘입니다. 각 노드에서 특성에 대한 조건을 평가하여 데이터를 분할하고, 리프 노드에 도달할 때까지 재귀적으로 분할을 반복합니다.',
    'rf': 'Random Forest Regressor는 다수의 의사결정 트리를 앙상블하여 회귀를 수행하는 알고리즘입니다. 각 트리는 데이터의 무작위 부분 집합을 사용하여 학습되며, 최종 예측은 모든 트리의 예측을 평균하여 얻습니다.',
    'et': 'Extra Trees Regressor는 Random Forest와 유사한 앙상블 알고리즘으로, 의사결정 트리를 무작위로 구성하여 회귀를 수행합니다. 무작위성을 극대화하여 모델의 분산을 줄이고 일반화 성능을 향상시킵니다.',
    'ada': 'AdaBoost Regressor는 부스팅 앙상블 기법을 사용하여 회귀를 수행하는 알고리즘입니다. 약한 학습기를 순차적으로 학습시키고, 각 학습기의 예측을 가중 결합하여 최종 예측을 생성합니다.',
    'gbr': 'Gradient Boosting Regressor는 부스팅 앙상블 기법을 사용하여 회귀를 수행하는 알고리즘입니다. 약한 학습기를 순차적으로 학습시키고, 각 학습기는 이전 학습기의 잔차를 학습하여 모델을 점진적으로 개선합니다.',
    'mlp': 'Multi-layer Perceptron (MLP) Regressor는 인공 신경망을 사용하여 회귀를 수행하는 알고리즘입니다. 입력층, 은닉층, 출력층으로 구성된 신경망을 통해 데이터를 학습하고 비선형 관계를 모델링합니다.',
    'xgboost': 'Extreme Gradient Boosting (XGBoost)는 경사 부스팅 알고리즘의 효율적인 구현체로, 회귀와 분류 문제에 널리 사용됩니다. 병렬 처리, 트리 가지치기, 규제 기법 등을 통해 성능과 확장성을 높였습니다.',
    'lightgbm': 'Light Gradient Boosting Machine (LightGBM)은 경사 부스팅 알고리즘의 경량화된 버전으로, 더 빠른 학습 속도와 낮은 메모리 사용량을 제공합니다. 리프 중심 분할과 히스토그램 기반 알고리즘을 사용하여 효율성을 높였습니다.',
    'catboost': 'CatBoost Regressor는 범주형 변수를 자동으로 처리할 수 있는 경사 부스팅 알고리즘입니다. 순열 기반 특성 중요도, 대칭 트리 분할, 순서화된 부스팅 등의 기술을 사용하여 높은 예측 성능을 제공합니다.',
    'dummy': 'Dummy Regressor는 예측 시 항상 같은 값을 반환하는 더미(dummy) 회귀 모델입니다. 기준 모델(baseline model)로 사용되어 다른 모델의 성능 평가에 도움을 줍니다.'
    }
    
    X_test_df = base.X_test_transformed.copy()
    y_test = base.y_test_transformed
    X_test_df.columns = [col.replace(".", "__").replace("{", "__").replace("}", "__") for col in X_test_df.columns]
    
    list_models = ml_create_model(base, models)
    explain_models = []

    for model, model_name in zip(list_models,models):
        print(model)
        try:
            explainer = RegressionExplainer(model, X_test_df, y_test)
        except :
            pass
        
        try:
            dashboard = ExplainerDashboard(explainer, title=f'{regression_models[model_name]}', description=reg_description[model_name])
        except:
            try:
                dashboard = ExplainerDashboard(explainer, title=f'{model_name}', description='설명을 표기할 수 없습니다.')
            except:
                continue
        explain_models.append(dashboard)
    
    return explain_models


def check_target_type(df, target_column):
    unique_values = df[target_column].nunique()
    
    if unique_values == 2:
        target_type = "Binary"
    elif unique_values == 3:
        target_type = "Multi-class"
    else:
        target_type = "Multi-class (more than 3)"
        
        # 클래스 수가 3개 초과인 경우, 중간값을 기준으로 binary 변수로 변환
        median_value = df[target_column].median()
        df[target_column] = np.where(df[target_column] <= median_value, 0, 1)
        target_type = "Binary (converted)"
        unique_values = 2
    
    return target_type, unique_values, df

def auto_ml_cls(df, target):
    
    cls_model = cls.setup(df, target = target, session_id=42, normalize_method='minmax', fold=5, )
    
    
    reg_avail_model = cls_model.models().index.to_list()
    
    try:
        reg_avail_model.remove('lightgbm')
    except ValueError:
        pass
 

    return cls_model, reg_avail_model

def make_cls_dashboards(base, models):
    classificiaton_models = {'lr' : 'Logistic Regression',
    'knn' : 'K Neighbors Classifier',
    'nb' : 'Naive Bayes',
    'dt' : 'Decision Tree Classifier',
    'svm' : 'SVM - Linear Kernel',
    'rbfsvm' : 'SVM - Radial Kernel',
    'gpc' : 'Gaussian Process Classifier',
    'mlp' : 'MLP Classifier',
    'ridge' : 'Ridge Classifier',
    'rf' : 'Random Forest Classifier',
    'qda' : 'Quadratic Discriminant Analysis',
    'ada' : 'Ada Boost Classifier',
    'gbc' : 'Gradient Boosting Classifier',
    'lda' : 'Linear Discriminant Analysis',
    'et' : 'Extra Trees Classifier',
    'xgboost' : 'Extreme Gradient Boosting',
    'lightgbm' : 'Light Gradient Boosting Machine',
    'catboost' : 'CatBoost Classifier'}

    cls_description = {
    'lr': 'Logistic Regression은 선형 모델을 사용하여 이진 분류 문제를 해결하는 알고리즘입니다. 로지스틱 함수를 사용하여 입력 변수와 클래스 확률 사이의 관계를 모델링합니다.',
    'knn': 'K Neighbors Classifier는 k-최근접 이웃 알고리즘을 사용하여 분류를 수행합니다. 새로운 데이터 포인트의 클래스를 예측할 때, 주변의 k개 이웃 데이터 포인트들의 클래스를 고려하여 다수결로 결정합니다.',
    'nb': 'Naive Bayes는 베이즈 정리를 사용하여 확률 기반 분류를 수행하는 알고리즘입니다. 각 특성이 독립적이라는 가정 하에, 각 클래스에 대한 조건부 확률을 계산하여 클래스를 예측합니다.',
    'dt': 'Decision Tree Classifier는 트리 구조를 사용하여 분류 규칙을 학습하는 알고리즘입니다. 각 노드에서 특성에 대한 조건을 평가하여 데이터를 분할하고, 리프 노드에 도달할 때까지 재귀적으로 분할을 반복합니다.',
    'svm': 'Support Vector Machine (SVM) with Linear Kernel은 선형 커널 함수를 사용하여 데이터를 고차원 공간에 매핑하고, 클래스 간의 최대 마진 초평면을 찾아 분류를 수행하는 알고리즘입니다.',
    'rbfsvm': 'Support Vector Machine (SVM) with Radial Basis Function (RBF) Kernel은 RBF 커널 함수를 사용하여 데이터를 고차원 공간에 매핑하고, 클래스 간의 최대 마진 초평면을 찾아 분류를 수행하는 알고리즘입니다.',
    'gpc': 'Gaussian Process Classifier는 베이지안 접근 방식을 사용하여 분류를 수행하는 알고리즘입니다. 가우시안 프로세스를 사용하여 데이터의 사후 확률 분포를 추정하고, 이를 기반으로 클래스를 예측합니다.',
    'mlp': 'Multi-layer Perceptron (MLP) Classifier는 인공 신경망을 사용하여 분류를 수행하는 알고리즘입니다. 입력층, 은닉층, 출력층으로 구성된 신경망을 통해 데이터를 학습하고, 비선형 관계를 모델링할 수 있습니다.',
    'ridge': 'Ridge Classifier는 릿지 회귀를 사용하여 분류를 수행하는 알고리즘입니다. L2 정규화를 사용하여 모델의 복잡도를 제어하고, 과적합을 방지합니다.',
    'rf': 'Random Forest Classifier는 다수의 의사결정 트리를 앙상블하여 분류를 수행하는 알고리즘입니다. 각 트리는 데이터의 무작위 부분 집합을 사용하여 학습되며, 최종 예측은 모든 트리의 예측을 결합하여 이루어집니다. Random Forest는 높은 정확도와 일반화 능력을 제공하며, 특성 중요도를 평가할 수 있습니다.',
    'qda': 'Quadratic Discriminant Analysis (QDA)는 이차 판별 함수를 사용하여 분류를 수행하는 알고리즘입니다. 클래스별로 다른 공분산 행렬을 가정하며, 결정 경계가 비선형일 수 있습니다.',
    'ada': 'Ada Boost Classifier는 부스팅 앙상블 기법을 사용하여 분류를 수행하는 알고리즘입니다. 약한 학습기를 순차적으로 학습시키고, 각 학습기의 예측을 가중 결합하여 최종 예측을 생성합니다. 어려운 샘플에 더 높은 가중치를 부여하여 학습합니다.',
    'gbc': 'Gradient Boosting Classifier는 부스팅 앙상블 기법을 사용하여 분류를 수행하는 알고리즘입니다. 약한 학습기를 순차적으로 학습시키고, 각 학습기는 이전 학습기의 오차를 보완하는 방향으로 학습합니다. 경사 하강법을 사용하여 모델을 최적화합니다.',
    'lda': 'Linear Discriminant Analysis (LDA)는 선형 판별 함수를 사용하여 분류를 수행하는 알고리즘입니다. 클래스 간 분산을 최대화하고 클래스 내 분산을 최소화하는 방향으로 데이터를 투영하여 분류합니다.',
    'et': 'Extra Trees Classifier는 Random Forest와 유사한 앙상블 알고리즘으로, 의사결정 트리를 무작위로 구성하여 분류를 수행합니다. 무작위성을 극대화하여 다양성을 높이고, 과적합을 방지합니다.',
    'xgboost': 'Extreme Gradient Boosting (XGBoost)는 경사 부스팅 알고리즘의 효율적인 구현체로, 분류와 회귀 문제에 널리 사용됩니다. 병렬 처리, 트리 가지치기, 규제 기법 등을 통해 성능과 확장성을 높였습니다.',
    'lightgbm': 'Light Gradient Boosting Machine (LightGBM)은 경사 부스팅 알고리즘의 경량화된 버전으로, 더 빠른 학습 속도와 낮은 메모리 사용량을 제공합니다. 리프 중심 분할과 히스토그램 기반 알고리즘을 사용하여 효율성을 높였습니다.',
    'catboost': 'CatBoost Classifier는 범주형 변수를 자동으로 처리할 수 있는 경사 부스팅 알고리즘입니다. 순열 기반 특성 중요도, 대칭 트리 분할, 순서화된 부스팅 등의 기술을 사용하여 높은 정확도와 예측 품질을 제공합니다.'
    }
    
    X_test_df = base.X_test_transformed.copy()
    y_test = base.y_test_transformed
    X_test_df.columns = [col.replace(".", "__").replace("{", "__").replace("}", "__") for col in X_test_df.columns]
    
    list_models = ml_create_model(base, models)
    explain_models = []

    for model, model_name in zip(list_models,models):
        print(model)
        try:
            explainer = ClassifierExplainer(model, X_test_df, y_test)
        except :
            pass
        
        try:
            dashboard = ExplainerDashboard(explainer, title=f'{classificiaton_models[model_name]}', description=cls_description[model_name])
        except:
            try: 
                dashboard = ExplainerDashboard(explainer, title=f'{model_name}', description='설명을 표기할 수 없습니다.')
            except:
                continue
        
        explain_models.append(dashboard)
    
    return explain_models

def make_hub(db1, db2):
    final_dashboard = db1 + db2
    hub = ExplainerHub(final_dashboard, title=f"file 에 대한 분석입니다.", description='사용자가 입력하는 분석의 목적입니다.')
    hub.to_zip('hub.zip')



def create_dashboard_dict():
    original_path = os.getcwd()
    
    zip_path = './hub.zip'

# hub.zip 파일 압축 해제
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./')

# explainerhub 폴더 경로
    explainerhub_path = './explainerhub'
    dashboard_dict = {}

    # 폴더 내의 모든 파일 리스트 가져오기
    file_list = os.listdir(explainerhub_path)

    # "dashboard"로 시작하는 html 파일만 필터링
    dashboard_files = [file for file in file_list if file.startswith("dashboard") and file.endswith(".html")]

    # 각 파일을 읽어서 딕셔너리에 추가
    for file_name in dashboard_files:
        file_path = os.path.join(explainerhub_path, file_name)
        
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # <H1> 태그에서 분석 이름 추출
        start_index = html_content.find("<H1>") + 4
        end_index = html_content.find("</H1>")
        analysis_name = html_content[start_index:end_index]

        # 딕셔너리에 분석 이름과 html 내용 추가
        dashboard_dict[analysis_name] = html_content
    os.chdir(original_path)
    return dashboard_dict

def delete_files_and_folders():
    # 현재 경로 가져오기
    current_path = os.path.abspath(".")
    
    # "hub.zip" 파일 삭제
    hub_zip_path = os.path.join(current_path, "hub.zip")
    if os.path.exists(hub_zip_path):
        os.remove(hub_zip_path)
        print("hub.zip 파일이 삭제되었습니다.")
    else:
        print("hub.zip 파일이 존재하지 않습니다.")
    
    # "explainerhub" 폴더 삭제
    explainerhub_path = os.path.join(current_path, "explainerhub")
    if os.path.exists(explainerhub_path):
        shutil.rmtree(explainerhub_path)
        print("explainerhub 폴더가 삭제되었습니다.")
    else:
        print("explainerhub 폴더가 존재하지 않습니다.")
    return None



# def lstm_model(df,bank_lists, target):  # V1
#     '''
#     df : pd.DataFrame (df2)
#     bank_lists : list (e.g. ['BANK #1']
#     target : str "KV001"
#     '''
#     ## data prep
#     x_train, x_val, x_test, y_train, y_val, y_test, scalar_target = lstm_train_test(df,bank_lists,target)

#     ## lstm AI
#     AI = Sequential()
#     AI.add(LSTM(50, activation='relu', input_shape=(1,x_train.shape[2])))
#     AI.add(Dense(1))
#     AI.compile(optimizer='adam', loss='mse')
#     AI.fit(x_train, y_train,
#               validation_data=(x_val, y_val),
#               epochs=100)
#     pred = AI.predict(x_test)
#     print(pred,y_test)

#     rescaled_actual = scalar_target.inverse_transform(y_test.reshape(-1,1))
#     rescaled_pred = scalar_target.inverse_transform(pred.reshape(-1,1))
#     print(rescaled_actual, rescaled_pred)


#     # inverse_transformered_X = scalar.inverse_transform(transformed_X)

#     ## hp tuning

#     '''trace1 = go.Scatter(
#         x=rescaled_pred.index,
#         y=stock_data['Close'],
#         mode='lines',
#         name='Actual Price'
#     )

#     # Create a trace for the predicted data
#     trace2 = go.Scatter(
#         x=valid.index,
#         y=valid['Predictions'],
#         mode='lines',
#         name='Predicted Price'
#     )

#     # Define the layout
#     layout = go.Layout(
#         title='Stock Price Prediction using LSTM',
#         xaxis={'title': 'Date'},
#         yaxis={'title': 'Close Price USD'}
#     )
#     # Create a Figure and plot the graph
#     fig = go.Figure(data=[trace1, trace2], layout=layout)
#     fig.show()
#    '''
#     return None


# def lstm_model(df1, df2, bank_lists, target, timestep):  # V2
#     '''
#     df1 : pd.DataFrame (df1)
#     df2 : pd.DataFrame (df2)
#     bank_lists : list (e.g. ['BANK #1']
#     target : str (e.g. "KV001")
#     timestep : int (e.g. 1, 2,,,,)
#     '''
#     ## data prep
#     x_train, x_val, x_test, y_train, y_val, y_test, scaler_feature, scaler_target, date = lstm_train_test(df1, df2,
#                                                                                                           bank_lists,
#                                                                                                           target,
#                                                                                                           timestep=timestep)

#     ## lstm AI
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', input_shape=(timestep, x_train.shape[2])))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(x_train, y_train,
#               validation_data=(x_val, y_val),
#               verbose=0,
#               epochs=100)
#     pred = model.predict(x_test)

#     rescaled_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))
#     rescaled_pred = scaler_target.inverse_transform(pred.reshape(-1, 1))
#     # predict
#     df_pred = pd.DataFrame(rescaled_pred, columns=['pred'])
#     pred_date = date[-len(rescaled_pred):]

#     ''' hyperparameter tuning'''

#     # Visualization

#     fig = go.Figure()

#     # 첫 번째 데이터셋 추가
#     fig.add_trace(
#         go.Scatter(x=date, y=select_bank(df2, bank_lists, 'delete')[target], mode='lines+markers', name='Actual'))

#     # 두 번째 데이터셋 추가
#     fig.add_trace(go.Scatter(x=pred_date, y=df_pred['pred'], mode='lines+markers', name='predict'))

#     # 레이아웃 설정
#     fig.update_layout(title=f'{target} Time Series Forecasting with LSTM',
#                       xaxis_title='Date',
#                       yaxis_title='Value')

#     # 그래프 보여주기 & 저장
#     fig.show()
#     fig.write_html('./bank_sol/HTML/analysis/lstm_figure.html')

#     return None

# ## classification 
# ## preprocess.py의 binning()을 통해 df 전처리 후 밑의 기능 수행 가능.

# def fold_value(df, target):
#     condition = df[target].value_counts().min()
#     if condition < 10: fold = condition
#     else: fold = 10 # 기본값이 10이어서
#     return fold

# def auto_ml_cls_first(df, target, ignore, save_path,method = 'minmax' ):
#     model = cls.setup(data = df, target = target, ignore_features = ignore, normalize = True, normalize_method = method)
   
#     fold = fold_value(df,target)

#     # best = cls.compare_models(fold = fold_value)
    
#     xgb_cls = cls.create_model('xgboost',cross_validation=True, fold = fold)
#     cls_dashboard_html(model,xgb_cls,"XGB", save_path)
#     # lightgbm_cls = cls.create_model('lightgbm',cross_validation=True, fold = fold_value)
#     # cls_dashboard_html(AI,lightgbm_cls,"Lightgbm")
#     lr_cls = cls.create_model('lr',cross_validation=True, fold = fold)
#     cls_dashboard_html(model,lr_cls,"lr",save_path)
    

#     '''
#     line37 참고
#     '''
#     avail_model = model.models().index.to_list()
#     for tmp in avail_model:
#         try:
#             cls_model = cls.create_model(tmp, cross_validation=True, fold = fold)
#             cls_dashboard_html(model, cls_model, tmp, save_path)
#         except :
#             print(f"{tmp}")
#             avail_model.remove(tmp)

#     return avail_model

def quantreg(df):
   
    y_target = df.columns[0]
    x_columns = df.columns[1:]
    formula = y_target + " ~ " + ' + '.join(x_columns)

    quantiles = [0.25, 0.5, 0.75]  # 원하는 분위수 설정

    results = []
    for q in quantiles:
        model = smf.quantreg(formula, df)
        result = model.fit(q=q)
        results.append(result)

# 결과 출력
    for q, result in zip(quantiles, results):
        print(f"Quantile: {q}")
        print(result.summary())
        plt.figure(figsize=(10, 6))
        plt.scatter(df[y_target], result.fittedvalues, color='blue', alpha=0.5)
        plt.xlabel('Actual')
        plt.ylabel('Fitted')
        plt.title(f'Quantile Regression - Quantile: {q}')
        plt.plot([min(df[y_target]), max(df[y_target])], [min(df[y_target]), max(df[y_target])], color='red', linestyle='—')
        plt.grid(True)
        plt.show()
        print("=" * 80)