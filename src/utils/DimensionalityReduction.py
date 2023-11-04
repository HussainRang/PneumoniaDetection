import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from abc import ABC , abstractmethod 

class DimensionalityReductionTemplate(ABC):
    @abstractmethod
    def fit(self,train_images,test_images,**model_params):
        """
        Initializes Model and returns the reduced images 
        """
        pass


class PCA_Model(DimensionalityReductionTemplate):
    def fit(self,train_images,test_images,**model_params):
        try:
            model = PCA(**model_params)
            logging.debug(f"PCA Model Initialized")

            train_reduced_images = model.fit_transform(train_images)
            test_reduced_images = model.transform(test_images)
            
            return train_reduced_images,test_reduced_images
        except Exception as e:
            logging.error(e,exc_info=True)


class t_SNE_Model(DimensionalityReductionTemplate):
    def fit(self,train_images,test_images,**model_params):
        try:
            model = TSNE(**model_params)
            logging.debug(f"TSNE model Initialized")

            train_reduced_images = model.fit_transform(train_images)
            test_reduced_images = model.fit_transform(test_images)

            return train_reduced_images,test_reduced_images
        except Exception as e:
            logging.error(e,exc_info=True)