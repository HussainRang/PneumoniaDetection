import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import log_loss,accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,balanced_accuracy_score


class MetricsCalculatorTemplate:

    def __init__(self,model,X_test,y_test):
        pass

    @abstractmethod
    def get_loss(self):
        pass
    
    @abstractmethod
    def get_accuracy(self):
        pass

    @abstractmethod
    def get_confusion_matrix(self):
        pass
    
    @abstractmethod
    def get_precision_for_classes(self):
        pass

    @abstractmethod
    def get_recall_for_classes(self):
        pass

    @abstractmethod
    def get_macroaverage_precision(self):
        pass

    @abstractmethod
    def get_macroaverage_recall(self):
        pass

    @abstractmethod
    def get_balanced_accuracy(self):
        pass

    @abstractmethod
    def get_weighted_balanced_accuracy(self):
        pass

    @abstractmethod
    def get_macro_F1score(self):
        pass

    @abstractmethod
    def get_micro_F1score(self):
        pass


class MetricsCalculator(MetricsCalculatorTemplate):

    def __init__(self,model,test_images,y_test,class_weights):
        self.model = model
        self.test_images = test_images
        self.y_test = y_test
        self.class_weights = class_weights


    def calcuate_probabilities(self):
        self.predicted_probabs = self.model.predict_proba(self.test_images)
        self.predicted_labels = self.model.predict(self.test_images)
        logging.debug(f"True Labels: \n {self.y_test[:2]}")
        logging.debug(f"Predicted Probabilities: \n {self.predicted_probabs[:2]}")
        logging.debug(f"Predicted Labels: \n {self.predicted_labels[:2]}")


    def calculate_sample_weights(self):
        if(self.class_weights==None):
            self.sample_weight=None
        else:
            self.sample_weight = [self.class_weights[index] for index in self.y_test]        


    def get_loss(self):
        try:
            loss = log_loss( y_true = self.y_test , y_pred = self.predicted_probabs , sample_weight=self.sample_weight )
            logging.debug("Loss Calculated Succesfully!")
            return loss
        except Exception as e:
            logging.error(e,exc_info=True)


    def get_accuracy(self):
        try:
            accuracy = accuracy_score(self.y_test,self.predicted_labels)
            logging.debug("Accuracy Calculated Successfully!!")
            return accuracy
        except Exception as e:
            logging.error(e,exc_info=True)


    def get_confusion_matrix(self):
        try:
            conf_mat = confusion_matrix(self.y_test,self.predicted_labels)
            logging.debug("Confusion Matrix Fetched Successfully!!")
            return conf_mat
        except Exception as e:
            logging.error(e,exc_info=True)


    def  get_macroaverage_precision(self):
        try:
            macro_avg_prec = precision_score(y_true=self.y_test,y_pred=self.predicted_labels,average="macro")
            logging.debug("Macro Average Precision Extracted Successfully!!")
            return macro_avg_prec
        except Exception as e:
            logging.error(e,exc_info=True)

    
    def get_macroaverage_recall(self):
        try:
            macro_avg_recall = recall_score(y_true=self.y_test,y_pred=self.predicted_labels,average="macro")
            logging.debug("Macro Average Recall Extracted Successfully!!")
            return macro_avg_recall
        except Exception as e:
            logging.error(e,exc_info=True)


    def get_balanced_accuracy(self):
        try:
            balanced_accuracy = balanced_accuracy_score(y_true=self.y_test , y_pred=self.predicted_labels)
            return balanced_accuracy
        except Exception as e: 
            logging.error(e,exc_info=True)
    

    def get_weighted_balanced_accuracy(self):
        try:
            weighted_balanced_accuracy = balanced_accuracy_score(y_true=self.y_test , y_pred=self.predicted_labels , sample_weight=self.sample_weight)
            return weighted_balanced_accuracy
        except Exception as e:
            logging.error(e,exc_info=True)

    
    def get_macro_F1score(self):
        try:
            macro_f1_score = f1_score(self.y_test,self.predicted_labels,average='macro')
            return macro_f1_score
        except Exception as e:
            logging.error(e,exc_info=True)


    def get_micro_F1score(self):
        try:
            micro_f1_score = f1_score(self.y_test,self.predicted_labels,average='micro')
            return micro_f1_score
        except Exception as e:
            logging.error(e,exc_info=True)
        