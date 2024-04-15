from .syndata import *
from .models import *
from .visualize import *
import time

class OCA:
    '''

    Ex)
    oca =OCA(df, 'fare', 'name', '설명')
    oca.run()
    oca.analysis_dict <- dadhboard dict
    

    '''
    def __init__(self, df, target ,file_name='', descriptio=''):
        '''
        df : pandas.DataFrame
        target : str
        file_name : str (파일이름)
        description : str (분석 목적?)
        '''
        
        self.df = df
        self.trans_df = None
        self.target = target
        _, _, self.cls_df = check_target_type(df, target)

    def analyze_data_quality(self):
        '''
        입력 데이터에 대한 기초분석, Quality report html 작성, meta data 관리
        '''
        html_autoviz(self.df)
        html_dqreport(self.df)
        self.data_quality_report = merge_html_files()
        self.meta_data = make_meta(self.df)

        self.eda_report_dict = {'EDA' : self.data_quality_report}
        
        return None

    def clear_file(self):
        delete_folder_contents()
        delete_files_and_folders()
        return None

    def setup(self):
        
        reg_model, reg_avail_model = auto_ml_reg(self.df, self.target)
        # reg_models = ml_create_model(reg_model, reg_avail_model)
        reg_dashboard = make_reg_dashboards(reg_model, reg_avail_model)
        
        cls_model, cls_avail_model= auto_ml_cls(self.cls_df, self.target)
        # cls_models = ml_create_model(cls_model, cls_avail_model)
        cls_dashboard = make_cls_dashboards(cls_model, cls_avail_model)
        
        make_hub(reg_dashboard, cls_dashboard)
        
        
        self.analysis_dict = create_dashboard_dict()
        self.analysis_dict.update(self.eda_report_dict)
        return None


    def run(self):
        self.analyze_data_quality()
        self.setup()
       
       
        self.clear_file()
        return self.analysis_dict

        
        