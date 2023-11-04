import logging
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_class_weights(training_labels,use_sample_weights):
    try:

        classes = np.unique(training_labels)

        if use_sample_weights is None:
            logging.debug(f"CLASS WEIGHTS: None")
            return None
        
        elif use_sample_weights=="balanced":
            weights = compute_class_weight('balanced', classes=classes, y=training_labels)
            weights_dict = dict()
            for index,weight in enumerate(weights):
                weights_dict[index] = weight
            
            logging.debug(f"CLASS WEIGHTS: {weights_dict}")
            return weights_dict
        
        else: 
            return None
        
    except Exception as e:
        logging.error(e,exc_info=True)