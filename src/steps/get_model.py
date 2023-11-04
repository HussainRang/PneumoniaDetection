import logging
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB,ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier

def get_model(model_name):
    try:
        model_dict = {
            "LogisticRegression": LogisticRegression(),
            "GaussianNaiveBayes": GaussianNB(),
            "GaussianNaiveBayes": MultinomialNB(),
            "BernoulliNaiveBayes": BernoulliNB(),
            "ComplementNaiveBayes": ComplementNB(),
            "DescisionTree": DecisionTreeClassifier(),
            "SupportVectorClassifier": SVC(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBoost": XGBClassifier()
        }
        
        model = model_dict[model_name]
        logging.debug(f"Initialized {model_name} Model")
        return model
    
    except Exception as e:
        logging.error(e,exc_info=True)