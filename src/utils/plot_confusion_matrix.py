import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import logging

def plot_confusion_matrix(save_pathname , confusion_matrix , classes):
    
    try:
        logging.debug("PLOTTING CONFUSION MATRIX")
        sns.heatmap(confusion_matrix, annot=True , cmap="OrRd" , xticklabels=classes , yticklabels=classes , fmt='g' )
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.savefig(save_pathname)
        return
    
    except Exception as e:
        logging.error(e,exc_info=True)