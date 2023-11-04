import logging
from sklearn.preprocessing import LabelEncoder


def label_encoding(y_train,y_test):
    try:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        
        classes = list(label_encoder.classes_)
        logging.debug(f"Label Encoding Classes: {label_encoder.classes_}")
        logging.debug(f"Classes {classes} converted to: {label_encoder.transform(classes)}")

        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

        return y_train , y_test , classes
    
    except Exception as e:
        logging.error(e,exc_info=True)