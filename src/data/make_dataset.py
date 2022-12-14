import pandas as pd
import os

def read_file_csv(filename):
    df = pd.read_csv(os.path.join('data/raw/', filename))
    print(filename, ' file read')
    return df

def data_preparation(df):
    df_proc = df.drop(columns=['BloodPressure', 'SkinThickness'])
    return df_proc

def data_exporting(df_proc, filename):
    df_proc.to_csv(os.path.join('data/processed/', filename))
    print(filename, 'exported file to processed data folder')

def main():
    df = read_file_csv('diabetes.csv')
    df_proc = data_preparation(df)
    data_exporting(df_proc, 'diabetes.csv')

if __name__ == "__main__":
    main()