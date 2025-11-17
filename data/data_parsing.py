import os
import pandas as pd
import numpy as np
import zipfile
import shutil
from scipy import io
from tqdm import tqdm

def dxai_parsing(dxai_root, dist_root):
    dxai_root = os.path.join(dxai_root,'Mechanical faults in rotating machinery dataset (normal, unbalance, misalignment, looseness)')

    # 1. 전체 실험 파일들이 압축되어 있기 때문에 압축을 먼저 해제해야 한다.
    for file in os.listdir(dxai_root):
        if file.endswith('.zip'):
            dist_path = os.path.join(dxai_root, file.replace('.zip', ''))
            with zipfile.ZipFile(os.path.join(dxai_root, file), 'r') as zip_ref:
                zip_ref.extractall(path=dist_path)
            os.remove(os.path.join(dxai_root, file))

    rpm                 = 1238 # RPM은 고정
    sampling_rate       = 25000 # 샘플링 레이트는 고정
    load_condition      = 'unknown' # 로드 조건은 unknown으로 고정
    severity            = 'unknown' # 심각도는 unknown으로 고정

    meta_list = []
    for test_idx, test_name in enumerate(os.listdir(dxai_root)):
        test_dir = os.path.join(dxai_root, test_name, test_name)
        if os.path.isdir(test_dir) is False:
            continue
        print(f'Processing {test_idx+1}/{len(os.listdir(dxai_root))} : {test_name}')
        class_name = test_name.split('_')[1].split(' ')[0].lower()
        dist_name = f'dxai_{test_idx}.npy'
        meta_info = {
            'file_name' : dist_name,
            'rpm'    : rpm,
            'sampling_rate' : sampling_rate,
            'load_condition' : load_condition,
            'severity' : severity,
            'class_name' : class_name,
            'sensor_position' : ['motor_y', 'motor_x', 'disk_y', 'disk_x']
        }
        
        dist_path = os.path.join(dist_root, dist_name)
        test_data = []
        for file in tqdm(os.listdir(test_dir)):
            if not file.endswith('.npy'):
                continue
            file_path = os.path.join(test_dir, file)
            data = np.load(file_path)
            test_data.append(data)
        test_np = np.concatenate(test_data, axis=1)

        np.save(dist_path, test_np)
        meta_info['data_sec'] = len(test_data)
        meta_list.append(meta_info)

    meta_pd = pd.DataFrame(meta_list)
    
    return meta_pd

def iis_parsing(iis_root, dist_root):

    sampling_rate = 4096 # 샘플링 레이트는 고정
    load_condition = 'unknown'  # 로드 조건은 unknown으로 고정

    test_cnt = 0
    meta_list = []
    for file in os.listdir(iis_root):
        
        if not file.endswith('.csv'):
            continue
        print(f'Processing {test_cnt+1}/{len(os.listdir(iis_root))} : {file}')
        test_file = os.path.join(iis_root, file)
        data_pd = pd.read_csv(test_file)
        data_pd = data_pd.dropna(axis=0)
        rpm_df = data_pd['Measured_RPM']
        change_indices = [0]  # Start with the first index
        change_indices += rpm_df.index[
            (rpm_df.diff().abs() > 100)
        ].tolist()
        if file[0] == '0':
            class_name = 'normal'
            severity = None
        elif file[0] == '1':
            class_name = 'unbalance'
            severity = f'{1.4*3.281}gram*cm'
            # severity = 'radius_14mm+mass_3.281g'
        elif file[0] == '2':
            class_name = 'unbalance'
            severity = f'{1.85*3.281}gram*cm'
            # severity = 'radius_18.5mm+mass_3.281g'
        elif file[0] == '3':
            class_name = 'unbalance'
            severity = f'{2.3*3.281}gram*cm'
            # severity = 'radius_23mm+mass_3.281g'
        elif file[0] == '4':
            class_name = 'unbalance'
            severity = f'{2.3*6.614}gram*cm'
            # severity = 'radius_23mm+mass_6.614g'
        else:
            print(f'Unknown class for file {file}')
            continue
            
        start_idx = 0
        
        for change_idx in tqdm(change_indices[1:]):
            end_idx = change_idx
            static_rpm = data_pd.iloc()[start_idx: end_idx]
            
            mean_rpm = static_rpm['Measured_RPM'].mean()
            
            data_np = static_rpm[['Vibration_1', 'Vibration_2','Vibration_3']].to_numpy().T
            
            dist_name = f'iis_{test_cnt}.npy'
            meta_info = {
                'file_name' : dist_name,
                'rpm'    : mean_rpm,
                'sampling_rate' : sampling_rate,
                'load_condition' : load_condition,
                'severity' : severity,
                'class_name' : class_name,
                'sensor_position' : ['disk_x', 'disk_y', 'motor_y'],
                'data_sec' : len(static_rpm)/sampling_rate
            }
            start_idx=end_idx
            
            if mean_rpm < 100:
                continue
            np.save(os.path.join(dist_root, dist_name), data_np)
            test_cnt+=1
            meta_list.append(meta_info)

    meta_pd = pd.DataFrame(meta_list)
    
    return meta_pd

def vat_parsing(vat_root, dist_root):

    vib_zip = os.path.join(vat_root, 'vibration.zip')
    with zipfile.ZipFile(vib_zip, 'r') as zip_ref:
                zip_ref.extractall(path=os.path.join(vat_root, 'vibration'))
    vat_root = os.path.join(vat_root, 'vibration')



    rpm = 3010 # RPM은 고정
    sampling_rate = 25600 # 샘플링 레이트는 고정
    meta_list = []
    for file_idx, file_name in tqdm(enumerate(os.listdir(vat_root))):
        
        load_condition = file_name.split('.')[0].split('_')[0]
        class_name = file_name.split('.')[0].split('_')[1].lower()
        if class_name in ['bpfi', 'bpfo', 'misalign']:
            severity = float(file_name.split('.')[0].split('_')[2])
            severity = f'{severity}mm'
        elif class_name == 'unbalance':
            severity = float(file_name.split('.')[0].split('_')[2][:-2])/1000
            severity = f'{severity}gram'
        else:
            severity = None
        if class_name == 'misalign':
            class_name = 'misalignment'
            
        file_path = os.path.join(vat_root, file_name)
        mat_file = io.loadmat(file_path)
        signal = mat_file['Signal'][0][0][1][0][0][0].transpose()
        dist_name = f'vat_{file_idx}.npy'
        meta_info = {
                'file_name' : dist_name,
                'rpm'    : rpm,
                'sampling_rate' : sampling_rate,
                'load_condition' : load_condition,
                'severity' : severity,
                'class_name' : class_name,
                'sensor_position' : ['motor_x', 'motor_y', 'disk_x', 'disk_y'],
                'data_sec' : signal.shape[1]/sampling_rate
            }
        np.save(os.path.join(dist_root, dist_name), signal)
        meta_list.append(meta_info)
    meta_pd = pd.DataFrame(meta_list)
    return meta_pd

def vbl_parsing(vbl_root, dist_root):

    file_cnt = 0
    sampling_rate = 20000 # 샘플링 레이트는 고정
    data_sec = 5 # 데이터 길이는 5초로 고정
    rpm = 3000 # RPM은 고정
    load_condition = 'unknown' # 로드 조건은 unknown으로 고정

    meta_list = []
    for class_name in os.listdir(vbl_root):
        print(f'Processing class: {class_name}')
        class_dir = os.path.join(vbl_root, class_name)
        
        for file in tqdm(os.listdir(class_dir)):
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(class_dir, file)
            
            if class_name == 'normal':
                severity = None
            elif class_name == 'unbalance':
                if file[2] =='_':
                    severity = f'{file[3:4]}gram*cm'
                else:
                    severity = f'{int(file[2:4])}gram*cm'
            elif class_name == 'misalignment':
                severity = '3mm'
            elif class_name == 'bearing':
                severity = 'hammer_attack'
            else:
                severity = None
                
            data_np = pd.read_csv(file_path).to_numpy().transpose()
            data_np[1:] # 첫번째 칼럼은 시간축
            dist_name = f'vbl_{file_cnt}.npy'
            meta_info = {
                'file_name' : dist_name,
                'rpm'    : rpm,
                'sampling_rate' : sampling_rate,
                'load_condition' : load_condition,
                'severity' : severity,
                'class_name' : class_name,
                'sensor_position' : ['motor_x', 'motor_y', 'motor_z'],
                'data_sec' : data_sec
            }
            np.save(os.path.join(dist_root, dist_name), data_np)
            meta_list.append(meta_info)
            file_cnt += 1


    meta_pd = pd.DataFrame(meta_list)
    return meta_pd

def mfd_parsing(mfd_root, dist_root):

    sampling_rate = 50000 # 샘플링 레이트는 고정
    load_condition = 'unknown' # 로드 조건은 unknown으로 고정
    mfd_columns = ['tachometer', 'motor_z', 'motor_y', 'motor_x', 'disk_z', 'disk_y', 'disk_x', 'microphone']
    data_columns = ['motor_x', 'motor_y', 'disk_x', 'disk_y']
    data_sec = 5 # 데이터 길이는 5초로 고정
    file_cnt = 0
    meta_list = []
    for class_name in os.listdir(mfd_root):
        class_dir = os.path.join(mfd_root, class_name)
        print(f'Processing class: {class_name}')
        if class_name == 'normal':
            severity = None
            for file in tqdm(os.listdir(class_dir)):
                file_path = os.path.join(class_dir, file)
                data_pd = pd.read_csv(file_path, header = None, names=mfd_columns)
                data_np = data_pd[data_columns].to_numpy().transpose()
                
                # MFD 데이터는 tachometer 칼럼을 이용하여 RPM을 계산
                tachometer_np = data_pd['tachometer'].to_numpy()
                tachometer_np = np.convolve(tachometer_np, np.ones(5) / 5, mode='same')
                binary_np = (tachometer_np >3).astype(int)
                rising_edges = np.where(np.diff(binary_np) > 0)[0]
                pulse_intervals = np.diff(rising_edges) / sampling_rate
                avg_time_per_revolution = np.mean(pulse_intervals)
                rpm = int((1 / avg_time_per_revolution) * 60)
                
                dist_name = f'mfd_{file_cnt}.npy'
                meta_info = {
                    'file_name' : dist_name,
                    'rpm'    : rpm,
                    'sampling_rate' : sampling_rate,
                    'load_condition' : load_condition,
                    'severity' : severity,
                    'class_name' : class_name,
                    'sensor_position' : data_columns,
                    'data_sec' : data_sec
                }
                np.save(os.path.join(dist_root, dist_name), data_np)
                file_cnt += 1
                meta_list.append(meta_info)
        
        elif class_name in ['overhang', 'underhang']:
            for specific_class_name in os.listdir(class_dir):
                for severity_folder in os.listdir(os.path.join(class_dir, specific_class_name)):
                    severity = severity_folder
                    class_name_full = f'{class_name}_{specific_class_name}'
                    print(f'Processing class: {class_name}, severity: {severity}')
                    for file in tqdm(os.listdir(os.path.join(class_dir, specific_class_name, severity_folder))):
                        file_path = os.path.join(class_dir, specific_class_name, severity_folder, file)
                        data_pd = pd.read_csv(file_path, header = None, names=mfd_columns)
                        data_np = data_pd[data_columns].to_numpy().transpose()
                        
                        # MFD 데이터는 tachometer 칼럼을 이용하여 RPM을 계산
                        tachometer_np = data_pd['tachometer'].to_numpy()
                        tachometer_np = np.convolve(tachometer_np, np.ones(5) / 5, mode='same')
                        binary_np = (tachometer_np >3).astype(int)
                        rising_edges = np.where(np.diff(binary_np) > 0)[0]
                        pulse_intervals = np.diff(rising_edges) / sampling_rate
                        avg_time_per_revolution = np.mean(pulse_intervals)
                        rpm = int((1 / avg_time_per_revolution) * 60)
                        
                        dist_name = f'mfd_{file_cnt}.npy'
                        meta_info = {
                            'file_name' : dist_name,
                            'rpm'    : rpm,
                            'sampling_rate' : sampling_rate,
                            'load_condition' : load_condition,
                            'severity' : severity,
                            'class_name' : class_name_full,
                            'sensor_position' : data_columns,
                            'data_sec' : data_sec
                        }
                        np.save(os.path.join(dist_root, dist_name), data_np)
                        file_cnt += 1
                        meta_list.append(meta_info)
        
        else:
            for severity_folder in os.listdir(class_dir):
                if class_name == 'imbalance':
                    severity = f'{float(severity_folder[:-1])*15.24}gram*cm'
                else:
                    severity = severity_folder
                print("Processing severity: ", severity)
                for file in tqdm(os.listdir(os.path.join(class_dir, severity_folder))):
                    file_path = os.path.join(class_dir, severity_folder, file)
                    data_pd = pd.read_csv(file_path, header = None, names=mfd_columns)
                    data_np = data_pd[data_columns].to_numpy().transpose()
                    
                    # MFD 데이터는 tachometer 칼럼을 이용하여 RPM을 계산
                    tachometer_np = data_pd['tachometer'].to_numpy()
                    tachometer_np = np.convolve(tachometer_np, np.ones(5) / 5, mode='same')
                    binary_np = (tachometer_np >3).astype(int)
                    rising_edges = np.where(np.diff(binary_np) > 0)[0]
                    pulse_intervals = np.diff(rising_edges) / sampling_rate
                    avg_time_per_revolution = np.mean(pulse_intervals)
                    rpm = int((1 / avg_time_per_revolution) * 60)
                    
                    dist_name = f'mfd_{file_cnt}.npy'
                    meta_info = {
                        'file_name' : dist_name,
                        'rpm'    : rpm,
                        'sampling_rate' : sampling_rate,
                        'load_condition' : load_condition,
                        'severity' : severity,
                        'class_name' : class_name,
                        'sensor_position' : data_columns,
                        'data_sec' : data_sec
                    }
                    np.save(os.path.join(dist_root, dist_name), data_np)
                    file_cnt += 1
                    meta_list.append(meta_info)
    meta_pd = pd.DataFrame(meta_list)
    return meta_pd

if __name__=='__main__':
    original_root = os.path.join("/home", "data")
    output_root = os.path.join(os.getcwd(), 'unzipped')
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    dist_root = os.path.join(os.getcwd(),'processed')
    if os.path.exists(dist_root):
        shutil.rmtree(dist_root)
    
    # original 안에 zip되어 있는 각 데이터셋 파일을 추출해서 unzipped 폴더에 저장
    print('Unzipping original data...')
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    if not os.path.exists(dist_root):
        os.makedirs(dist_root, exist_ok=True)
    for file in os.listdir(original_root):
        if file.endswith('.zip'):
            dist_path = os.path.join(output_root, file.replace('.zip', ''))
            with zipfile.ZipFile(os.path.join(original_root, file), 'r') as zip_ref:
                zip_ref.extractall(path=dist_path)
    print('Unzipping completed.')
    
    # 각 데이터셋을 파싱하여 메타데이터를 생성
    print('Parsing data...')
    print('Parsing DXAI dataset...')
    dxai_root = os.path.join(output_root, '1_FaultDXAI')
    dxai_meta = dxai_parsing(dxai_root, dist_root)
    
    print('Parsing IIS dataset...')
    iis_root = os.path.join(output_root, '2_FG')
    iis_meta = iis_parsing(iis_root, dist_root)
    
    print('Parsing VAT dataset...')
    vat_root = os.path.join(output_root, '3_VAT-MCD', 'Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis')
    vat_meta = vat_parsing(vat_root, dist_root)
    
    print('Parsing VBL dataset...')
    vbl_root = os.path.join(output_root, '4_VBL-VA001', 'VBL-VA001')
    vbl_meta = vbl_parsing(vbl_root, dist_root)
    
    print('Parsing MFD dataset...')
    mfd_root = os.path.join(output_root, '6_full')
    mfd_meta = mfd_parsing(mfd_root, dist_root)
    
    print('Parsing completed.')
    meta_pd = pd.concat([dxai_meta, iis_meta, vat_meta, vbl_meta, mfd_meta], ignore_index=True)
    meta_pd['dataset'] = meta_pd['file_name'].apply(lambda x: x.split('_')[0])
    meta_pd.to_csv(os.path.join(dist_root, 'meta.csv'), index=False)
    shutil.rmtree(output_root)