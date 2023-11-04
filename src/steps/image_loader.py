from utils.ImageLoader import ImageLoader
import logging

def image_loader(image_paths , new_dimension , as_grayscale):
    try:
        image_loader = ImageLoader(new_dimension)
        
        logging.debug("Loading Images")
        images = image_loader.load_images(image_paths,as_grayscale=as_grayscale)
        
        logging.debug(f"IMAGES : \n {images.shape}")
        logging.debug("Images Loaded Successfully!!")
        
        return images
    
    except Exception as e:
        logging.error(e,exc_info=True)