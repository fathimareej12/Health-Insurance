import mlflow
import mlflow.sklearn

Experiment_name='Vehicle insurance prediction'

def setup_experiment():
    mlflow.set_experiment(Experiment_name)

def log_experiment(model,model_name,metrics):
    setup_experiment()
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({'model_name':model_name})
        model_parameter=model.get_params()

        for parameter_name,parameter_value in model_parameter.items():
            mlflow.log_param(parameter_name,parameter_value)

        for metric_name,metric_value in metrics.items():
            mlflow.log_metric(metric_name,metric_value)

        mlflow.sklearn.log_model(sk_model=model,name=model_name)

        print(model_name)
        print(Experiment_name)

def main():
    print('Run log_experiment()','from main.py')