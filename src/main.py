from components.data_ingestion import load_data
from components.data_validation import dataset_check
from components.eda import basic_eda
from components.Visualization import create_visuals
from components.feature_selection import select_features
from components.transform import classify_features

def main():
    data=load_data()
    dataset_check(data)
    basic_eda(data)
    create_visuals(data)
    X,y = select_features(data)
    num,cat,tar=classify_features(data)
    print(num,cat,tar)
    print(X.shape)
    print(y.shape)

if __name__ == '__main__':
    main()


