from data_ingestion import load_data

def main():
    data=load_data()
    print('Row and column count')
    print(data.shape,'\n','-'*50)
    print('Top 10 values')
    print(data.head(10),'\n','-'*50)


if __name__ == '__main__':
    main()

