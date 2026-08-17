import pandas as pd

def basic_eda(data:pd.DataFrame):
    print('\n','='*50)
    print('Exploratory data analysis'.center(50))
    print('='*50)

    print(f'Dataset shape: {data.shape}')
    print('\n Target variable Distribution')
    print(data['Response'].value_counts())

    print('\n Numerical variable distribution')
    print(data.describe())

    print('\n Policy sales channel Distribution')
    print(data['Policy_Sales_Channel'].value_counts())

    print('\n Age Distribution')
    print(data['Age'].value_counts())

    print(pd.crosstab(data['Vehicle_Age'],data['Vehicle_Damage']))




def main():
    from data_ingestion import load_data

    data=load_data()
    basic_eda(data)

if __name__ == '__main__':
    main()


