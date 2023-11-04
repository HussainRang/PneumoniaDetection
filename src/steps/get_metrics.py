import logging

from utils.MetricsCalculator import MetricsCalculator

def get_metrics(model,sample_weights,test_images,y_test):
    try:
        logging.debug("Calculating Metrics")
        metrics_calculator = MetricsCalculator(model,test_images,y_test,sample_weights)
        metrics_calculator.calcuate_probabilities()
        metrics_calculator.calculate_sample_weights()
        
        loss = metrics_calculator.get_loss()
        accuracy = metrics_calculator.get_accuracy()
        confusion_matrix = metrics_calculator.get_confusion_matrix()
        macro_precision = metrics_calculator.get_macroaverage_precision()
        macro_recall = metrics_calculator.get_macroaverage_recall()
        balanced_accuracy = metrics_calculator.get_balanced_accuracy()
        weighted_balanced_accuracy = metrics_calculator.get_weighted_balanced_accuracy()
        macro_F1_score = metrics_calculator.get_macro_F1score()
        micro_F1_score = metrics_calculator.get_micro_F1score()

        report = {
        "Loss" : loss ,
        "Accuracy" : accuracy ,
        "Confusion matrix" : confusion_matrix ,
        "macro_precision" : macro_precision ,
        "macro_recall" : macro_recall ,
        "Balanced_accuracy" : balanced_accuracy ,
        "Weighted_Balanced_accuracy" : weighted_balanced_accuracy ,
        "Macro_F1_score" : macro_F1_score ,
        "Micro_F1_score" : micro_F1_score 
        }

        logging.debug(f"REPORT_CREATED")

        return report

    except Exception as e:
        logging.error(e,exc_info=True)