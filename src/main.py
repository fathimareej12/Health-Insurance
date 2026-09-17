from components.data_ingestion import load_data
from components.data_validation import dataset_check
from components.eda import basic_eda
from components.Visualization import create_visuals
from components.feature_selection import select_features
from components.transform import classify_features
from components.train_test_split import split_data
from components.preprocessing import preprocess_data,encode_target
from components.model_factory import get_model
from components.model_training import model_training_pipeline
from components.model_tuner import tune
from components.model_evaluation import evaluate_model
from components.experiment_tracking import log_experiment
from components.handle_imbalance import apply_undersampling


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
    X_train_bal,y_train_bal=apply_undersampling(X_train, y_train)
    #leader_board,best_model,best_model_name= model_training_pipeline(X_train, y_train, X_test, y_test)
    model_name,_,best_params,_,model=tune(X_train_bal,y_train_bal).values()
    results=evaluate_model(model=model,model_name=model_name,X_test=X_test,y_test= y_test)
    log_experiment(model,model_name,results)







if __name__ == '__main__':
    main()


