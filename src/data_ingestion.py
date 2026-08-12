import pandas as pd

def load_data():
    '''
    loads data from csv
    :return: a pandas dataframe
    '''


    data=pd.read_csv('Data/Raw Data/train.csv')
    return data

if __name__=='__main__':
    data=load_data()
    title='Top 10 values'
    print(title)
    print('-'*len(title))
    print(data.head(10),'\n','-'*50)
    print('Row and column counts')
    print(data.shape,'\n','-'*50)
    print('column names','\n','-'*50)
    print(data.columns)
