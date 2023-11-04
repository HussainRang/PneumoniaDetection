import logging
from utils.DataScaler import StandardScalerModel,MinMaxScalerModel 

def scaling(train_images , test_images , scaler_type , **scaler_params):
    try:
        algorithms_dict = {
            "StandardScaler" : StandardScalerModel(),
            "MinMaxScaler" : MinMaxScalerModel()
        }
        
        model = algorithms_dict[scaler_type]

        train_scaled_images,test_scaled_images = model.fit(train_images=train_images , test_images=test_images , **scaler_params)

        logging.debug(f"SCALING COMPLETED \n SAMPLES \n {train_scaled_images[:2]}")

        return train_scaled_images,test_scaled_images

    except Exception as e:
        logging.error(e,exc_info=True)
    