from components.data_ingestion import load_data
from components.data_validation import dataset_check
from components.eda import basic_eda
from components.Visualization import create_visuals
from components.feature_selection import select_features
from components.transform import classify_features
from components.train_test_split import split_data
from components.preprocessing import preprocess_data,encode_target
from components.model_factory import get_model


def main():
    data=load_data()
    dataset_check(data)
    basic_eda(data)
    create_visuals(data)
    X,y = select_features(data)
    num,cat,tar=classify_features(data)
    X_train,X_test,y_train,y_test=split_data(X,y )
    print(num,cat,tar)
    print(X_train.columns,X_test.columns)
    X_train,X_test,preprocessor=preprocess_data(X_train,X_test,num,cat)
    print(y_train)
    print(get_model())



if __name__ == '__main__':
    main()


