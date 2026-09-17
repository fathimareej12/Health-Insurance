import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             classification_report,
                             confusion_matrix,
                             roc_curve, ConfusionMatrixDisplay
                             )

def calculate_metrics(model,X_test,y_test):
    y_pred=model.predict(X_test)
    proba=model.predict_proba(X_test)[:,1]

    metrics={
        'Accuracy':accuracy_score(y_test,y_pred),
        'precision':precision_score(y_test,y_pred,zero_division=0),
        'Recall':recall_score(y_test,y_pred,zero_division=0),
        'F1 Score':f1_score(y_test,y_pred,zero_division=0),
        'ROC-AUC':roc_auc_score(y_test,y_pred)
    }
    return(y_pred,proba,metrics)


def display_metrics(metrics):
    for metric_name,value in metrics.items():
        print(f'{metric_name}:{value: .2f}')

def display_classification_report(y_test,y_pred):
    report=classification_report(y_test,
                                 y_pred,
                                 target_names=['No Response',
                                               'Response',],
                                               zero_division=0)
    print(report)

def display_confusion_matrix(y_test,y_pred):
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    display=ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=['No Response',
                                           'Response'],
                           )
    display.plot()
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.show()


def display_roc_curve(y_test,proba,roc_auc):
    false_positive_rate,true_positive_rate,thresholds = roc_curve(y_test,proba)
    #plt.figure(figsize=(10,10))
    plt.plot(false_positive_rate,
             true_positive_rate,
             label=f'ROC-AUC={roc_auc:.2f}',)
    plt.plot([0,1],[0,1],linestyle='--',label='Random classifier')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(
        model,
        model_name,
        X_test,
        y_test,
):
    print(f'model:{model_name}')
    (y_pred,proba,metrics)= calculate_metrics(model,X_test,y_test)
    display_metrics(metrics)
    display_classification_report(y_test,y_pred)
    display_roc_curve(y_test, proba, roc_auc=metrics['ROC-AUC'])
    display_confusion_matrix(y_test,y_pred)
    print(X_test.shape)
    print(y_test.shape)
    print(proba.shape)
    print(type(proba))


    return metrics


