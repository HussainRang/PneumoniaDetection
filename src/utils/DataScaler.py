from abc import ABC,abstractmethod
import logging
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class DataScalerTemplate(ABC):
    @abstractmethod
    def fit(self,train_images,test_images,**scaler_params):
        """
        Scaling the data
        """
        pass


class StandardScalerModel(DataScalerTemplate):

    def fit(self , train_images , test_images , **scaler_params):
        try:
            scaler = StandardScaler(**scaler_params)
            logging.debug("StandardScaler Initialized")

            train_scaled_images = scaler.fit_transform(train_images)
            test_scaled_images = scaler.transform(test_images)

            return train_scaled_images,test_scaled_images
        except Exception as e:
            logging.error(e,exc_info=True)


class MinMaxScalerModel(DataScalerTemplate):

    def fit(self , train_images , test_images , **scaler_params):
        try:
            scaler = MinMaxScaler(**scaler_params)
            logging.debug("StandardScaler Initialized")

            train_scaled_images = scaler.fit_transform(train_images)
            test_scaled_images = scaler.transform(test_images)

            return train_scaled_images,test_scaled_images
        except Exception as e:
            logging.error(e,exc_info=True)