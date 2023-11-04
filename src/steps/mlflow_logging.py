import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
from datetime import datetime

from utils.plot_confusion_matrix import plot_confusion_matrix

def mlflow_logging( experiment_name , best_params , best_model , model_name , report , classes):
    try:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():

            # Logging Parameters
            mlflow.log_params(best_params)

            # Logging Metric
            confusion_matrix = report["Confusion matrix"].copy()
            del report["Confusion matrix"]
            mlflow.log_metrics(report)

            # Logging model
            if(model_name=="XGBoost"):
                mlflow.xgboost.log_model(best_model,model_name)
            else:
                mlflow.sklearn.log_model(best_model,model_name)

            # Logging Confusion Matrix as figure
            confusion_matrix_path = "confusion_matrix.png"
            plot_confusion_matrix( confusion_matrix_path , confusion_matrix , classes )
            mlflow.log_artifact(confusion_matrix_path,"confusion matrix")

            # Logging tags
            datetime_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S") 
            mlflow.set_tags({ "Model Name":model_name , "Date Time":datetime_string })

            # Deleting confusion matrix
            os.remove(confusion_matrix_path)

    except Exception as e:
        logging.error(e,exc_info=True)