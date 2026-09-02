import pandas as pd

def classify_features(data:pd.DataFrame):
    numerical_features = data.select_dtypes(
        include=['int64','float64']).columns.tolist()
    numerical_features.remove('id')
    categorical_features = data.select_dtypes(
        include=['object']).columns.tolist()
    if 'Response' in numerical_features:
        numerical_features.remove('Response')

    target = 'Response'


    return(numerical_features,
           categorical_features,
           target)

def main():
    from components.data_ingestion import load_data

    data=load_data()
    classify_features(data)

if __name__ == '__main__':
    main()