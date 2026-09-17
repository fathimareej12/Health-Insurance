import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,f1_score)
from components.model_factory import get_model
from config.paths import MODEL_PATH,REPORTS_PATH


models= {'Random_forest':get_model()['Random Forest'],
         'Gradient_Boosting':get_model()['Gradient Boosting']}

parameter_grid={
    'Random_forest':{
        'n_estimators':[100,200],
        'max_depth':[10,None],
        'min_samples_split':[2]
    },
    'Gradient_Boosting':{
        'n_estimators': [100],
        'max_depth': [10, 20],
        'learning_rate':[0.1,0.01]
    }
}

grid_search= GridSearchCV(estimator=models,
                          param_grid=parameter_grid,
                          cv=2,
                          n_jobs=2)

def tune(X_train,y_train):
    tuning_results={}
    bestmodel,best_score=None,-1
    best_model_name=None
    best_params=None

    print('Tuning models')
    for model_name,model in models.items():
        print(f'Tuning{model_name}')
        parameter=parameter_grid[model_name]
        grid_search=GridSearchCV(model,
                                 parameter,
                                 cv=2,
                                 n_jobs=2)
        grid_search.fit(X_train,y_train)

        current_best_score=grid_search.best_score_
        current_best_params=grid_search.best_params_
        current_best_model=grid_search.best_estimator_

        tuning_results[model_name]={
            'best_model': current_best_model,
            'best_score': current_best_score,
            'best_params': current_best_params
        }

        if current_best_score > best_score:
            best_score=current_best_score
            best_params=current_best_params
            best_model_name=model_name
            best_model=current_best_model

    return{
        'best_model_name': best_model_name,
        'best_score':best_score,
        'best_params':best_params,
        'tuning_results':tuning_results,
        'best_model':best_model

    }