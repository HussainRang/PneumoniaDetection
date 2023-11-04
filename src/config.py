DATA_PATH = "../data"
LOGS_PATH = "./logs"
MODEL_TYPE = "ML"

# TRAIN TEST SPLIT
TRAIN_PERCENT = 0.7

# LOADING IMAGES (1D tuple with 2 elements)
NEW_DIMENSIONS = (256,256)
GRAYSCALE = False

# DIMENSIONALITY REDUCTION
APPLY_DIM_REDUCTION = True
DIM_REDUCTION_ALGO = "PCA"  # PCA , t-SNE 
DIM_REDUCTION_PARAMS = dict(
    n_components = 3
)

# SCALING
APPLY_SCALING = True
SCALER_TYPE = "StandardScaler" # StandardScaler , MinMaxScaler
SCALER_PARAMS = dict(

)


# MODEL
MODEL_NAME = "LogisticRegression" # LogisticRegression , GaussianNaiveBayes , MultinomialNaiveBayes , BernoulliNaiveBayes , 
                                # ComplementNaiveBayes , DescisionTree , SupportVectorClassifier , AdaBoostClassifier , 
                                # GradientBoostingClassifier , RandomForestClassifier , XGBoost
                                
USE_SAMPLE_WEIGHTS = "balanced"  # None , balanced 


# RANDOM SEARCH CV
RANDOM_SEARCH_CV_MODEL_DISTRIBUTIONS = dict(
    penalty = ['l1','l2','elasticnet',None],
    multi_class = ["multinomial"],
    verbose = [2],

)
RANDOM_SEARCH_CV_PARAMS = dict(
    n_iter = 20,
    refit = True,
    verbose = 2,
    cv = 5,
    random_state = 42,
)


# FOR MLFlow
LOG_TO_MLFLOW = True
EXPERIMENT_NAME = "Pneumonia Detection"



