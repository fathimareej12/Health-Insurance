import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,
recall_score,f1_score)
from components.model_factory import get_model
from config.paths import (MODEL_PATH,REPORTS_PATH)


def evaluate_model(model,X_test,y_test):
    pred= model.predict(X_test)

    return {'accuracy':accuracy_score(y_test,pred),
            'precision':precision_score(y_test,pred),
            'recall':recall_score(y_test,pred),
            'f1': f1_score(y_test,pred),}

def train_model(X_train,y_train,X_test,y_test):
    models = get_model()
    leaderboard=[]
    best_model=None
    best_model_name=''
    best_f1=0

    print('training Starts here....')

    for model_name,model in models.items():
        print('Training model',model_name)
        model.fit(X_train,y_train)
        metrics=evaluate_model(model, X_test, y_test)

        leaderboard.append({'model':model_name,
                            'accuracy':round(metrics['accuracy'],4),
                            'precision':round(metrics['precision'],4),
                            'recall':round(metrics['recall'],4),
                            'f1':round(metrics['f1'],4)})

        if metrics['f1'] > best_f1:
            best_f1=metrics['f1']
            best_model=model
            best_model_name=model_name

        print(f'{model_name} completed')

    leaderboard=pd.DataFrame(leaderboard)
    leaderboard=leaderboard.sort_values(by='f1',ascending=False)
    leaderboard.reset_index(drop=True,inplace=True)

    return (leaderboard,best_model,best_model_name)

def save_best_model(model,model_name):
    joblib.dump(model,MODEL_PATH / f'{model_name}.pkl')


def model_training_pipeline(X_train,y_train,X_test,y_test):
    leader_board,best_model,best_model_name=train_model(X_train,y_train,X_test,y_test)

    print(leader_board)
    save_best_model(best_model,best_model_name)
    print('model saved' )

    return (leader_board,best_model,best_model_name)

