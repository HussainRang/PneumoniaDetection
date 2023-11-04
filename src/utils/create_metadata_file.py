import os
import pandas as pd
import logging
from tqdm import tqdm

def create_metadata_file(DATA_PATH):
    
    try:
        logging.debug("Creating metadata file")
        
        categories = os.listdir(f"{DATA_PATH}/images")
        file_names = []
        classes = []
        
        for category in tqdm(categories):
            for file_name in os.listdir(f"{DATA_PATH}/images/{category}"):
                if(file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg')):
                    file_names.append(f"{DATA_PATH}/images/{category}/{file_name}")
                    classes.append(category)


        dataframe = pd.DataFrame({
            "name":file_names,
            "category":classes
        })

        logging.debug(f"DATAFRAME \n{dataframe.head()}")
        dataframe.to_csv(f"{DATA_PATH}/metadata.csv")
        
        logging.debug("Metadata file created")
        return
    
    except Exception as e:
        logging.error(e)
