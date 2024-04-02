from sdv.metadata import SingleTableMetadata ## singletablemetadata
from sdv.lite import SingleTablePreset
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.sequential import PARSynthesizer


import pandas as pd
import numpy as np
# from preprocess import *

def make_meta(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    return metadata



# def feature_gen(df, gen_num, case):
#     '''
#     df : DataFrame ## feature data
#     gen_num : int
#     gen_type : str
#     {"gaussian": GaussianCopulaSynthesizer(metadata),
#      "ctgan": CTGANSynthesizer(metadata),
#      "tvae": TVAESynthesizer(metadata),
#      "copula": CopulaGANSynthesizer(metadata),}
#     '''
#     metadata = SingleTableMetadata()
#     metadata.detect_from_dataframe(df)
#     match case:
#         case "gaussian":
#             synthesize = GaussianCopulaSynthesizer(metadata)
#         case "ctgan":
#             synthesize = CTGANSynthesizer(metadata)
#         case "tvae":
#             synthesize = TVAESynthesizer(metadata)
#         case "copula":
#             synthesize = CopulaGANSynthesizer(metadata)
#         case _:
#             print('wrong type')
#     synthesize.fit(df)
#     synthetic_data = synthesize.sample(num_rows=gen_num)

#     return synthetic_data

# def sequential_gen(df, bank_list, bank_num=1, seq_len=30):
#     '''
#     df : DataFrame ## date + bank + feature
#     bank_list : list  ## bank_name(ex BANK #1)
#     bank_num : int  ## 생성할 은행의 갯수
#     seq_len : int ## 생성할 시퀀스 갯수
#     '''
#     df['bank.name'] = df['bank.code'].apply(lambda x: x)
#     df = select_bank(df, bank_list,'origin')
#     metadata = SingleTableMetadata()
#     metadata.detect_from_dataframe(df)

#     metadata.update_column(
#         column_name='bank.code',
#         sdtype='id',
#         regex_format="[A-Z]{1}")

#     metadata.set_sequence_key(column_name='bank.code')
#     metadata.set_sequence_index(column_name='date')

#     synthesizer = PARSynthesizer(metadata, context_columns=['bank.name'])
#     synthesizer.fit(df)

#     syn_data = synthesizer.sample(num_sequences=bank_num, sequence_length=seq_len)
#     syn_data.drop(columns=['bank.name'], inplace=True)
#     return syn_data