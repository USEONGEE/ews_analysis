import os
import re
import socket
import numpy as np
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.chdir('./Data')  # 추후 DB로 변환

def read_data(path, type):
    try:
        if type == 'csv':
            df = pd.read_csv(path,encoding='utf-8',index_col=0)

        elif type == 'excel':
            df = pd.read_excel(path,index_col=0)

        else:
            print("Wrong data type")
            '''
            이부분에 api 호출 key?
            '''
            df  = ''
    except:

        try:
            if type == 'csv':
                df = pd.read_csv(path,encoding='cp949',index_col=0)

            elif type == 'excel':
                df = pd.read_excel(path,index_col=0)

            else:
                print("Wrong data type")
                '''
                이부분에 api 호출 key?
                '''
                df = ''
        except:
            if type == 'csv':
                df = pd.read_csv(path, encoding='euc-kr',index_col=0)

            elif type == 'excel':
                df = pd.read_excel(path,index_col=0)

            else:
                print("Wrong data type")
                '''
                이부분에 api 호출 key?
                '''
                df = ''

    return df

def Id_list(df, col_id):
    id_list = df[col_id].unique()
    return id_list

def make_date(df):
    '''
    WD Raw data에서 'year.code' & 'month.code'를 병합하여 하나의 날짜로 만들어 주는 코드
    df : raw data(WD)
    '''
    df['date'] = pd.to_datetime(df['year.code'].astype(str) + df['month.code'].astype(str), format='%Y%m')
    df.drop(columns=['year.code','month.code'], inplace =True)
    return df

def transform_date(df, cols: Union[str, list]):
    '''
    cols:

    '''

    if isinstance(cols, str):
        df[cols] = pd.to_datetime(df[cols])

    elif isinstance(cols, list):
        year_pattern = r'\b(.*Y.*|.*y.*|.*year.*|.*YEAR.*|.*년.*|.*년도.*)\b'
        month_pattern = r'\b(.*M.*|.*m.*|.*monthv|vMONTH.*|.*mon.*|.*월.*)\b'
        day_pattern = r'\b(d.*|.*일.*|.*day.*)\b'
        year_matches, month_matches, day_matches = [],[],[]

        for i in cols:
            year_matches.extend(re.findall(year_pattern, i))
            month_matches.extend(re.findall(month_pattern, i))
            day_matches.extend(re.findall(day_pattern, i))

        if  not day_matches: ## day가 비었을 때
            df['date'] = pd.to_datetime(df[year_matches[0]].astype(str) + df[month_matches[0]].astype(str), format='%Y%m')
            df.drop(columns=[year_matches[0], month_matches[0]], inplace=True)

        else: ## day가 있을 때
            df['date'] = pd.to_datetime(df[year_matches[0]].astype(str) + df[month_matches[0]].astype(str) + df[day_matches[0]].astype(str), format='%Y%m%d')
            df.drop(columns=[year_matches[0], month_matches[0], day_matches[0]], inplace=True)

    else:
        print("Date Col must 'str','list'")


    return df




def missing_df(df):
    try:
        df = df.replace([np.inf, -np.inf], np.nan)  # "inf" 값을 NaN으로 대체
        df = df.dropna()
    except:
        df = df.dropna()
    return df

def order_df(df):
    '''
        1 : 은행 + 날짜 + features
        2 : 은행 + features
        3 : features
    '''
    df = make_date(df)
    feature_1 = ['date', 'bank.code', 'KV001', 'KV002', 'KV003', 'KV004', 'KV005', 'KV006', 'KV007', 'KV008', 'KV009',
                 'KV010', 'KV011', 'KV012', 'KV013', 'KV014', 'KV015', 'KV016', 'KV017', 'KV018']
    feature_2 = ['bank.code', 'KV001', 'KV002', 'KV003', 'KV004', 'KV005', 'KV006', 'KV007', 'KV008', 'KV009', 'KV010',
                 'KV011', 'KV012', 'KV013', 'KV014', 'KV015', 'KV016', 'KV017', 'KV018']
    feature_3 = ['KV001', 'KV002', 'KV003', 'KV004', 'KV005', 'KV006', 'KV007', 'KV008', 'KV009', 'KV010',
                 'KV011', 'KV012', 'KV013', 'KV014', 'KV015', 'KV016', 'KV017', 'KV018']

    df_timeseries = df[feature_1]
    df_timeseries = missing_df(df_timeseries)
    df_bank = df[feature_2]
    df_bank = missing_df(df_bank)
    df_feature = df[feature_3]
    df_feature = missing_df(df_feature)

    return df_timeseries, df_bank, df_feature

def order_df_ins(df):
    df = make_date(df)
    feature_1 = ['date', 'life.name', 'KV3001', 'KV3002', 'KV3003', 'KV3004', 'KV3005', 'KV3006', 'KV3007', 'KV3008', 'KV3009',
                 'KV3010']
    feature_2 = ['life.name','KV3001', 'KV3002', 'KV3003', 'KV3004', 'KV3005', 'KV3006', 'KV3007', 'KV3008', 'KV3009',
                 'KV3010']
    feature_3 = ['KV3001', 'KV3002', 'KV3003', 'KV3004', 'KV3005', 'KV3006', 'KV3007', 'KV3008', 'KV3009',
                 'KV3010']

    df_timeseries = df[feature_1]
    df_timeseries = missing_df(df_timeseries)
    df_bank = df[feature_2]
    df_bank = missing_df(df_bank)
    df_feature = df[feature_3]
    df_feature = missing_df(df_feature)

    return df_timeseries, df_bank, df_feature

def select_bank(df, bank_list,type):
    '''
    bank_list : list
    type : str ('delete' or 'origin'
    '''
    if type=='delete':
        df = df[df['bank.code'].isin(bank_list)]
        df = df.drop(columns = 'bank.code')

    elif type=='origin':
        df = df[df['bank.code'].isin(bank_list)]

    else:
        df = None
    return df

def select_object(df, object_col,  object_name, type: bool):
    '''
    df : dataframe
    object_name : str
    type : bool (true : 'delete' , Flase : 'origin'
    '''
    if type:
        df = df[df[object_col].isin(object_name)]
        df = df.drop(columns = object_col)

    else:
        df = df[df[object_col].isin(object_name)]
    return df

def select_ins(df, bank_list,type):
    '''
    bank_list : list
    type : str ('delete' or 'origin'
    '''
    if type=='delete':
        df = df[df['life.name'].isin(bank_list)]
        df = df.drop(columns = 'life.name')

    elif type=='origin':
        df = df[df['life.name'].isin(bank_list)]

    else:
        df = None
    return df

# def lstm_train_test(df, bank_lists, target):
#
#     '''
#     df : DataFrame, (essential df2)
#     bank_lists : list (e.g. ["BANK #1"]
#     target : str
#     '''
#
#     ## feature or label
#     df = select_bank(df, bank_lists,"delete")
#     all_col = df.columns.to_list()
#     feature = all_col.copy()
#     feature.remove(target)
#
#     ## MinMaxScaler
#     scalar_target = MinMaxScaler(feature_range=(0, 1))
#     scalar_feature = MinMaxScaler(feature_range=(0, 1))
#     df[target] = scalar_target.fit_transform(df[target].values.reshape(-1, 1))
#     df[feature] = scalar_feature.fit_transform(df[feature])
#
#     num_feature = len(all_col)
#     target = [target]
#
#     label_df = pd.DataFrame(df, columns=target)
#     feature_df = pd.DataFrame(df, columns=all_col)
#
#     feature_df = feature_df.to_numpy()
#     label_df = label_df.to_numpy()
#
#
#
#     x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.4, shuffle=False, random_state=42)
#     x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False, random_state=42)
#
#     print(f'# of train : {len(x_train)} \n# of val : {len(x_val)} \n# of test : {len(x_test)}')
#
#     return x_train.reshape(-1,1,num_feature), x_val.reshape(-1,1,num_feature), x_test.reshape(-1,1,num_feature), y_train.reshape(-1,1,1), y_val.reshape(-1,1,1), y_test.reshape(-1,1,1), scalar_target

def datasetcreation(df,x_feature, y_feature, timestep=1):
    '''
    df : pd.DataFrame ,
    x_feature : list (feature list)
    y_feature : list (target list)
    timestep : int
    '''
    DataX, DataY = [], []
    for i in range(len(df)- timestep -1):
        feature = df[x_feature][i:(i+ timestep)]
        DataX.append(feature)
        target = df[y_feature].iloc[i+ timestep]
        target = np.array(target)
        DataY.append(target)
    return np.array(DataX), np.array(DataY)


def lstm_train_test(df1, df2, bank_lists, target, timestep=1):
    '''
    df1 : DataFrame, (essential df1)
    df2 : DataFrame, (essential df2)
    bank_lists : list (e.g. ["BANK #1"]
    target : str
    timestep : int
    '''

    ## feature or label
    date = select_bank(df1, bank_lists, "delete")['date']  # 날짜 추출
    df = select_bank(df2, bank_lists, "delete")  # 실제로 돌릴 데이터
    all_col = df.columns.to_list()
    feature = all_col.copy()
    # feature.remove(target)  # target 제외한 변수로 돌리는 과정

    ## MinMaxScaler 정규화
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_feature = MinMaxScaler(feature_range=(0, 1))
    df[target] = scaler_target.fit_transform(df[target].values.reshape(-1, 1))
    df[feature] = scaler_feature.fit_transform(df[feature])

    num_feature = len(feature)
    target_list = [target]

    train_set, test_set = train_test_split(df, test_size=0.4, shuffle=False, random_state=42)
    val_set, test_set = train_test_split(test_set, test_size=0.5, shuffle=False, random_state=42)
    print(type(train_set))
    print(train_set.to_numpy().shape)

    x_train, y_train = datasetcreation(train_set, feature, target_list, timestep=timestep)
    x_val, y_val = datasetcreation(val_set, feature, target_list, timestep=timestep)
    x_test, y_test = datasetcreation(test_set, feature, target_list, timestep=timestep)

    print(f'# of train : {len(x_train)} \n# of val : {len(x_val)} \n# of test : {len(x_test)}')
    print(x_train, y_train)
    print(x_train.shape, y_train.shape)
    return x_train, x_val, x_test, y_train, y_val, y_test, scaler_feature, scaler_target, date

def lstm_train_test_ins(df1, df2, bank_lists, target, timestep=1):
    '''
    df1 : DataFrame, (essential df1)
    df2 : DataFrame, (essential df2)
    bank_lists : list (e.g. ["BANK #1"]
    target : str
    timestep : int
    '''

    ## feature or label
    date = select_ins(df1, bank_lists, "delete")['date']  # 날짜 추출
    df = select_ins(df2, bank_lists, "delete")  # 실제로 돌릴 데이터
    all_col = df.columns.to_list()
    feature = all_col.copy()
    # feature.remove(target)  # target 제외한 변수로 돌리는 과정

    ## MinMaxScaler 정규화
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_feature = MinMaxScaler(feature_range=(0, 1))
    df[target] = scaler_target.fit_transform(df[target].values.reshape(-1, 1))
    df[feature] = scaler_feature.fit_transform(df[feature])

    num_feature = len(feature)
    target_list = [target]

    train_set, test_set = train_test_split(df, test_size=0.4, shuffle=False, random_state=42)
    val_set, test_set = train_test_split(test_set, test_size=0.5, shuffle=False, random_state=42)
    print(type(train_set))
    print(train_set.to_numpy().shape)

    x_train, y_train = datasetcreation(train_set, feature, target_list, timestep=timestep)
    x_val, y_val = datasetcreation(val_set, feature, target_list, timestep=timestep)
    x_test, y_test = datasetcreation(test_set, feature, target_list, timestep=timestep)

    print(f'# of train : {len(x_train)} \n# of val : {len(x_val)} \n# of test : {len(x_test)}')
    print(x_train, y_train)
    print(x_train.shape, y_train.shape)
    return x_train, x_val, x_test, y_train, y_val, y_test, scaler_feature, scaler_target, date
def make_localhost(port):    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 1))
    local_ip_address = s.getsockname()[0] 
    return f'http://{local_ip_address}:{port}'

def binning(df, target):
    '''
    bank_name을 select_bank를 통해 가져와 간소화
    '''
    df[target] = pd.qcut(df[target], q = 4, labels = [f'range_{i}' for i in range(0,4)]) # 2개의 클래스로 분류해야하므로 은행 데이터셋일 때도 다시 고려해야함.
    
    
    return df


# 지급여력비율의 범주 분류의     기준을 금감원 권고 사항인 150%로 함.
# 따라서 보험 데이터셋에만 가능. 
def binning_ins(df, target):
    bins = [df[target].min(), 150, df[target].max()] 
    labels = ['권고 이하', '권고 이상']
    df_label = pd.cut(df[target], bins = bins, labels = labels, include_lowest = True)
    df[target] = df_label
    return df

# fold => 교차 검증 횟수
# 범주 중 가장 데이터를 적게 가지는 범주를 고려해서 각 검증 마다 그 범주의 데이터를 최소한 1개라도 사용하도록 하기 위해.
# fold가 너무 적을 경우는 우선 고려하지 않음. => 추후 개선 필요
def fold_value(df, target):
    condition = df[target].value_counts().min()
    if condition < 10: fold = condition
    else: fold = 10 # 기본값이 10이어서
    return fold

# explainerdashboard에서 한글이 깨져서 컬럼을 영어로 변경
def to_english_ins(df):
    df.columns = ['Point in Time',
                    'Number of Contracts',
                    'Insurance Amount (in thousands of KRW)',
                    'Premium Income (in thousands of KRW)',
                    'Unearned Premiums for Previous Period (in thousands of KRW)',
                    'Unearned Premiums for Next Period (in thousands of KRW)',
                    'Earned Premiums (in thousands of KRW)',
                    'Number of Original Premium Cases',
                    'Insurance Payout (in thousands of KRW)',
                    'Reserve Reversal Amount(in thousands of KRW)',
                    'Reserve Accumulation Amount (in thousands of KRW)',
                    'Loss Incurred (in thousands of KRW)',
                    'Loss Ratio (in percentage)',
                    'Payment Capacity Ratio']
    return df

# 범주화 후 컬럼명 영어로
def class_eng(df,target):
    df = binning_ins(df,target)
    df = to_english_ins(df)
    target_eng = df.columns[-1]
    date_eng = df.columns[0]
    return df, target_eng, date_eng