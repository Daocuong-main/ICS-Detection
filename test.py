import pandas as pd

files = ['datasets/hmi_normal_6h_flow_label.csv', 'datasets/mitm_hmi_scapy_modify_1plc_6h_flow_label.csv', 'datasets/mitm_hmi_forward_36plc_6h_flow_label.csv','datasets/mitm_hmi_scapy_forward_1plc_6h_flow_label.csv',
         'datasets/mitm_hmi_forward_1plc_6h_flow_label.csv']

for path in files:
    df = pd.read_csv(path)
    print(f"File: {path}")
    print(df['Label'].value_counts())
    print('-'*30)
