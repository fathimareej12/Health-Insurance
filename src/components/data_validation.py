import pandas as pd
from components.data_ingestion import load_data

def dataset_check(data:pd.DataFrame):
    '''
    Performs basic dataset check
    '''

    print('\n'+'*'*50)
    print('Dataset overview')
    print('*'*60)

    print(f'Shape: {data.shape}')

    print('\n columns of dataframe \n')
    print(data.columns.tolist())

    print('\n Missing values in dataframe\n')
    print(data.isnull().sum())

    print('\n Duplicated values in dataframe\n')
    print(data.duplicated().sum())

def main():
      data=load_data()
      dataset_check(data)

if __name__ == '__main__':
    main()


