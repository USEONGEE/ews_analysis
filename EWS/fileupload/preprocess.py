import os
import socket

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
시나리오 1 KVALL.WD 데이터가 들어옴.
1. year.code, month.code 결합 & 삭제 ==> make_date ==> order_df
2. data 정렬 ==> order_df
    ## 1 : 은행 + 날짜 + features
    ## 2 : 은행 + features
    ## 3 : features
3. drop.na or imputation(추후) ==> order_df에서 같이 진행
4. select bank(option) ==> select_bank
5. visualization(each bank, all of bank)
6. modeling?
"""
os.chdir("D:/dragonfly/ews/ews_analysis_server/EWS/Data")  # 추후 DB로 변환


def read_data(path, type):
    if type == "csv":
        df = pd.read_csv(path)
    elif type == "excel":
        df = pd.read_excel(path)

    else:
        print("Wrong data type")
        """
        이부분에 api 호출 key?
        """
        df = ""

    return df


"""
front-end에서 데이터 읽고 넘겨줄 때, feature 선택하고 정렬까지(e.g. target 1열, 나머지 feature 2열)
%
feature type까지 고려
ID(e.g. 은행, 보험사, and anything)
target(e.g. 은행에서 'KV001', 보험사 위기감지 변수)
feature(others without timesiries, target and others without target)
"""

## AutoML preprocess


def Id_list(df, col_id):
    id_list = df[col_id].unique()
    return id_list


def prep_data(df, feature_col):
    df = df[feature_col]
    df = df.replace([np.inf, -np.inf], np.nan)  # "inf" 값을 NaN으로 대체
    df = df.dropna()
    return df


def make_date(df):

    df["date"] = pd.to_datetime(
        df["year.code"].astype(str) + df["month.code"].astype(str), format="%Y%m"
    )
    df.drop(columns=["year.code", "month.code"], inplace=True)
    return df


## 1 : 은행 + 날짜 + features
## 2 : 은행 + features
## 3 : features
def order_df(df, data_type):
    df = make_date(df)
    feature_1 = [
        "date",
        "bank.code",
        "KV001",
        "KV002",
        "KV003",
        "KV004",
        "KV005",
        "KV006",
        "KV007",
        "KV008",
        "KV009",
        "KV010",
        "KV011",
        "KV012",
        "KV013",
        "KV014",
        "KV015",
        "KV016",
        "KV017",
        "KV018",
    ]
    feature_2 = [
        "bank.code",
        "KV001",
        "KV002",
        "KV003",
        "KV004",
        "KV005",
        "KV006",
        "KV007",
        "KV008",
        "KV009",
        "KV010",
        "KV011",
        "KV012",
        "KV013",
        "KV014",
        "KV015",
        "KV016",
        "KV017",
        "KV018",
    ]
    feature_3 = [
        "KV001",
        "KV002",
        "KV003",
        "KV004",
        "KV005",
        "KV006",
        "KV007",
        "KV008",
        "KV009",
        "KV010",
        "KV011",
        "KV012",
        "KV013",
        "KV014",
        "KV015",
        "KV016",
        "KV017",
        "KV018",
    ]

    if data_type == 1:
        df = df[feature_1]
    elif data_type == 2:
        df = df[feature_2]
    elif data_type == 3:
        df = df[feature_3]
    else:
        None

    df = df.replace([np.inf, -np.inf], np.nan)  # "inf" 값을 NaN으로 대체
    df = df.dropna()

    return df


def prep_data_with_na(df, feature_col):
    df = df[feature_col]
    df = df.replace([np.inf, -np.inf], np.nan)  # "inf" 값을 NaN으로 대체
    return df


def select_bank(df, bank_list):
    """
    bank_list : list
    """
    return df[df["bank.code"].isin(bank_list)]


## fss data 전용 preprocess
## date 병합, 은행명으로 전처리


def fss_date(df, feature_col, bank_name):  # feature_col에 bank.code 포함해야함

    df["date"] = pd.to_datetime(
        df["year.code"].astype(str) + df["month.code"].astype(str), format="%Y%m"
    )

    df = df[feature_col]
    df = df.replace([np.inf, -np.inf], np.nan)  # "inf" 값을 NaN으로 대체
    df = df.dropna()
    df = df[df["bank.code"] == bank_name]
    print(df)
    df = df.drop(columns=["bank.code"])

    return df


def lstm_train_test(df, target, feature):
    """
    Example of target & feature
    feature = ['KV002','KV003','KV004','KV005','KV006','KV007','KV008','KV009','KV010',
               'KV011', 'KV012', 'KV013','KV014','KV015','KV016','KV017','KV018']
    target = ['KV001']
    """
    ## feature or label
    feature_df = pd.DataFrame(df, columns=feature)
    label_df = pd.DataFrame(df, columns=target)

    feature_df = feature_df.to_numpy()
    label_df = label_df.to_numpy()

    num_feature = len(feature)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, label_df, test_size=0.4, shuffle=False, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, shuffle=False, random_state=42
    )

    print(
        f"# of train : {len(x_train)} \n# of val : {len(x_val)} \n# of test : {len(x_test)}"
    )

    return (
        x_train.reshape(-1, 1, num_feature),
        x_val.reshape(-1, 1, num_feature),
        x_test.reshape(-1, 1, num_feature),
        y_train.reshape(-1, 1, 1),
        y_val.reshape(-1, 1, 1),
        y_test.reshape(-1, 1, 1),
    )


## classification preprocess  추가예정


def make_localhost(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 1))
    local_ip_address = s.getsockname()[0]
    return f"http://{local_ip_address}:{port}"


def segmentation(df, target, bank_name):
    df = order_df(df, 2)
    df = df[df["bank.code"] == bank_name]
    df = df.drop(columns="bank.code")
    df[target] = pd.qcut(df[target], q=4, labels=[f"range_{i}" for i in range(0, 4)])
    return df
