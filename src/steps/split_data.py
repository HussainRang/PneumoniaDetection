from utils.DataSplitter import DataSplitter
import logging

def split_data( metadata_file_path:str , train_percent:float ):
    try:
    
        logging.debug("Splitting Data")
        
        data_splitter = DataSplitter( metadata_file_path )
        X_train , X_test , y_train , y_test = data_splitter.get_splits( train_percent )
        
        logging.debug("Data Splitted Successfully!!!")
        return X_train , X_test , y_train , y_test
    
    except Exception as e:
        logging.error( e , exc_info=True )