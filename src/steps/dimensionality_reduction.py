import logging
from utils.DimensionalityReduction import PCA_Model,t_SNE_Model

def dimensionality_reduction( train_images , test_images , algorithm , **model_params ):
    try:
        algorithms_dict = {
            "PCA" : PCA_Model(),
            "t-SNE" : t_SNE_Model(),
        }
        model = algorithms_dict[algorithm]
        
        train_reduced_images,test_reduced_images = model.fit(train_images = train_images, test_images = test_images , **model_params)

        logging.debug(f"Shape after Dimensionality Reduction is : {train_reduced_images.shape}")
        
        return train_reduced_images,test_reduced_images

    except Exception as e:
        logging.error(e,exc_info=True)