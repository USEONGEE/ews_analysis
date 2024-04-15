import os
from autoviz import AutoViz_Class
from pandas_dq import dq_report
import shutil
import chardet

def html_autoviz(df):
    '''
    df : df3
    df_type : str ("origin" or "synthetic")
    '''
    AV = AutoViz_Class()
    save_path = './analyze_data_quality'
    df_type='origin'
    
    html_maker = AV.AutoViz(
        filename="",
        sep=",",
        depVar="",
        dfte=df,
        header=0,
        verbose=2,
        lowess=True,
        chart_format="html",
        max_rows_analyzed=150000,
        max_cols_analyzed=30,
        save_plot_dir=save_path
    )
    
    
    
    ## html 병합과정
    folder_path = save_path+'/AutoViz/'  # 가져올 폴더의 경로를 지정해주세요
    html_files = [file for file in os.listdir(folder_path) if file.endswith('.html')]
    modified_list = [folder_path + item for item in html_files]

    merged_content = ''
    for file_name in modified_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            file_title = os.path.basename(file_name)
            file_content = f.read()
            merged_content += f'<br><br><hr><h2>{file_title}</h2><br><br>'
            merged_content += file_content
            merged_content += '<br><br>'

    # 결과를 새로운 파일에 저장
    with open(save_path + f'/{df_type}_merged.html', 'w', encoding='utf-8') as f:
        f.write(merged_content)

    return None
def html_dqreport(data):
    '''
    data : dataframe


    '''

    original_path = os.getcwd()
    if not os.path.exists('./analyze_data_quality'):
        os.makedirs('./analyze_data_quality')
    os.chdir('./analyze_data_quality')
    report = dq_report(data, html=True, csv_engine='pandas', verbose=1)

    os.chdir(original_path)

    return None



def merge_html_files(save_path=None):
    # 병합할 HTML 파일 경로
    save_path = './analyze_data_quality'
    dq_report_path = os.path.join(save_path, 'dq_report.html')
    origin_merged_path = os.path.join(save_path, 'origin_merged.html')
    
    # 병합된 내용을 저장할 변수
    merged_content = ''
    
    # dq_report.html 파일 내용 추가
    encoding = detect_file_encoding(dq_report_path)
    with open(dq_report_path, 'r', encoding=encoding) as f:
        
        merged_content += '<br><br><hr><h2>Data Quality Report</h2><br><br>'
        merged_content += f.read()
        
    # origin_merged.html 파일 내용 추가
    encoding = detect_file_encoding(origin_merged_path)
    with open(origin_merged_path, 'r', encoding=encoding) as f:
        merged_content += f.read()
    
    # 병합된 내용을 새로운 파일에 저장
    # TODO OS 패키지 문제 해결하기 
    merged_file_path = os.path.join(save_path, 'final_report.html')
    encoding = detect_file_encoding(merged_file_path)
    with open(merged_file_path, 'w', encoding=encoding) as f:
        f.write(merged_content)
        
    
    return merged_content


def detect_file_encoding(file_path):
    try :
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            return encoding
    except :
        return 'utf-8'




def delete_folder_contents(folder_path=None):
    # 폴더가 존재하는지 확인
    folder_path = './analyze_data_quality'
    if os.path.exists(folder_path):
        # 폴더 내의 모든 파일과 하위 폴더 삭제
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except OSError as e:
                    print(f"Error deleting file: {file_path} - {e}")
            
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted folder: {dir_path}")
                except OSError as e:
                    print(f"Error deleting folder: {dir_path} - {e}")
        
        print(f"All contents in {folder_path} deleted successfully.")
    else:
        print(f"Folder {folder_path} does not exist.")