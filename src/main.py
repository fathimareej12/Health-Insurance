from components.data_ingestion import load_data
from components.data_validation import dataset_check
def main():
    data=load_data()
    dataset_check(data)

if __name__ == '__main__':
    main()

