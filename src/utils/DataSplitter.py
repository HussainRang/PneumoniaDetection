from sklearn.model_selection import train_test_split
import pandas as pd
import logging

class DataSplitter : 
    def __init__(self,metadata_file_path:str):
        self.metadata_file_path = metadata_file_path

    def get_splits(self,train_percent:float):
        try:
            metadata_df = pd.read_csv(self.metadata_file_path)
            
            X = metadata_df["name"]
            y = metadata_df["category"]

            X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=train_percent,random_state=42,stratify=y)
            
            X_train.reset_index( drop=True , inplace=True)
            X_test.reset_index( drop=True , inplace=True)
            y_train.reset_index( drop=True , inplace=True)
            y_test.reset_index( drop=True , inplace=True)

            logging.debug(f"X_train\n{X_train[:5]}")
            logging.debug(f"y_train\n{y_train[:5]}")
            
            return X_train,X_test,y_train,y_test

        except Exception as e:
            logging.error(e,exc_info=True)
        