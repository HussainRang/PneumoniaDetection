import cv2 as cv
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImageLoader:
    def __init__(self,new_dimension):
        self.new_dimension = new_dimension


    def load_images(self,image_paths,as_grayscale):
        
        try:
            images = []
            
            for image_path in tqdm(image_paths):
                
                if (as_grayscale==True):
                    image = cv.imread(image_path,cv.COLOR_BGR2GRAY)
                else:
                    image = cv.imread(image_path)

                image = cv.resize(image,self.new_dimension)
                images.append(image)

            images = np.array(images)
            images = images/255
            
            logger.debug(f"Images Loaded")

            return images
        
        except Exception as e:
            logging.error(e,exc_info=True)
