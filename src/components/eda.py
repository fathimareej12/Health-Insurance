import pandas as pd
import os

Artifacts_path = "../../artifacts/reports"
os.makedirs(Artifacts_path,exist_ok=True)

def basic_eda(data:pd.DataFrame):
    print('Exploratory data analysis'.center(50))
    print('=' * 50)

    print(f'Dataset shape: {data.shape}')
    print('\n Target variable Distribution')
    print(data['Response'].value_counts())

    print('\n Numerical variable distribution')
    print(data.describe())

    print('\n Policy sales channel Distribution')
    print(data['Policy_Sales_Channel'].value_counts())

    print('\n Age Distribution')
    print(data['Age'].value_counts())

    print(pd.crosstab(data['Vehicle_Age'], data['Vehicle_Damage']))


    with open('dataset_summary.txt','w')as file:
        print('started')
        file.write('\n Exploratory data analysis \n'.center(50))
        file.write('=' * 50)
        print('mid')

        file.write(f'\n Dataset shape: {data.shape} \n')

        file.write('\n Target variable Distribution \n')
        file.write(str(data['Response'].value_counts()))

        file.write('\n Policy sales channel Distribution \n')
        file.write(str(data['Policy_Sales_Channel'].value_counts()))

        file.write('\n Age Distribution\n')
        file.write(str(data['Age'].value_counts()))






def main():
    from data_ingestion import load_data

    data=load_data()
    basic_eda(data)

if __name__ == '__main__':
    main()


