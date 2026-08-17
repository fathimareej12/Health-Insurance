from components.data_ingestion import load_data
from components.data_validation import dataset_check
from components.eda import basic_eda

def main():
    data=load_data()
    dataset_check(data)
    basic_eda(data)

if __name__ == '__main__':
    main()


