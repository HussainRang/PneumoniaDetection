import logging
import numpy as np

def linearize_images(images):
    try:
        if(len(images.shape)==4):
            images = np.mean(images,axis=3,dtype=np.float32)
            logging.debug(f"Images Shape after taking mean : {images.shape}")
        
        images = np.reshape( images , ( images.shape[0] , images.shape[1]*images.shape[2] ) )
        logging.debug(f"Images Shape change to : {images.shape}")
        
        return images
    except Exception as e:
        logging.error(e,exc_info=True)

