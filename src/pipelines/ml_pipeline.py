from config import DATA_PATH , TRAIN_PERCENT , NEW_DIMENSIONS , GRAYSCALE , APPLY_DIM_REDUCTION , DIM_REDUCTION_ALGO , DIM_REDUCTION_PARAMS 
from config import APPLY_SCALING , SCALER_TYPE , SCALER_PARAMS , MODEL_NAME , USE_SAMPLE_WEIGHTS , RANDOM_SEARCH_CV_MODEL_DISTRIBUTIONS , RANDOM_SEARCH_CV_PARAMS
from config import LOG_TO_MLFLOW , EXPERIMENT_NAME

from utils.create_metadata_file import create_metadata_file
from utils.linearize_images import linearize_images
from utils.get_class_weights import get_class_weights

from steps.split_data import split_data 
from steps.image_loader import image_loader
from steps.dimensionality_reduction import dimensionality_reduction
from steps.scaling import scaling
from steps.label_encoding import label_encoding
from steps.get_model import get_model
from steps.randomized_search_cv import randomized_search_cv
from steps.get_metrics import get_metrics
from steps.mlflow_logging import mlflow_logging

import time
import os
import logging
import mlflow
import mlflow.sklearn


def ml_pipeline():

    try:
        # Check if metadata file exists
        if( "metadata.csv" not in os.listdir(DATA_PATH) ):
            create_metadata_file(DATA_PATH)
        
        start_time = time.time()

        # Train test Split
        X_train,X_test,y_train,y_test = split_data(f"{DATA_PATH}/metadata.csv",TRAIN_PERCENT)

        # Load Images
        train_images = image_loader(X_train,NEW_DIMENSIONS,GRAYSCALE)
        test_images = image_loader(X_test,NEW_DIMENSIONS,GRAYSCALE)

        # Linearize images
        train_images = linearize_images(train_images)
        test_images = linearize_images(test_images) 

        # Dimensionality Reduction
        if APPLY_DIM_REDUCTION==True:
            logging.debug("APPLYING DIMENSIONALITY REDUCTION")
            train_images,test_images = dimensionality_reduction(train_images = train_images, test_images = test_images , algorithm = DIM_REDUCTION_ALGO,**DIM_REDUCTION_PARAMS)
            

        # Scaling
        if APPLY_SCALING==True:
            logging.debug("APPLYING SCALING")
            train_images,test_images = scaling(train_images = train_images , test_images=test_images , scaler_type = SCALER_TYPE , **SCALER_PARAMS)

        # Label Encoding
        y_train , y_test , classes = label_encoding( y_train, y_test )
        
        # Fetch Class weights 
        class_weight = get_class_weights(y_train,USE_SAMPLE_WEIGHTS)
        RANDOM_SEARCH_CV_MODEL_DISTRIBUTIONS["class_weight"] = [class_weight]

        # Fetch the model  
        model = get_model(MODEL_NAME)
        
        # feed to randomized search CV
        best_params,best_model = randomized_search_cv(model,train_images,y_train,RANDOM_SEARCH_CV_MODEL_DISTRIBUTIONS,RANDOM_SEARCH_CV_PARAMS)
        logging.debug(f"BEST PARAMS : {best_params}")

        # Get Metrics ( Loss , Confusion Matrix , Accuracy , Balanced Accuracy , Weighted Balanced Accuracy , MacroF1  )
        report = get_metrics(best_model,class_weight,test_images,y_test)
        time_taken = time.time() - start_time
        report["time_taken"] = time_taken
        logging.debug(f"\n{report}")


        # Log to MLFlow
        if(LOG_TO_MLFLOW==True):
            logging.debug("LOGGING TO MLFLOW !!!")            
            logging.debug(f"MLFLOW EXPERIMENT NAME: {EXPERIMENT_NAME}")

            mlflow_logging( EXPERIMENT_NAME , best_params , best_model , MODEL_NAME , report , classes )
            
            logging.debug("MLFLOW LOGGING COMPLETE")

        logging.debug("!!!!!! PIPELINE COMPLETE !!!!!!")
    
    except Exception as e:
        logging.error(e,exc_info=True)

