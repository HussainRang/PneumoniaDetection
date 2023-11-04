# CONFIG FILE

The config file is used by both `main.py` file and `pipelines` to run the project. It consists of the configuration for the models , scalars etc which are used by the `steps`. An example of the config file is shown below.

```py
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
DIM_REDUCTION_ALGO = "PCA"  
DIM_REDUCTION_PARAMS = dict(
    n_components = 3
)

# SCALING
APPLY_SCALING = True
SCALER_TYPE = "StandardScaler"
SCALER_PARAMS = dict(

)


# MODEL
MODEL_NAME = "LogisticRegression" 
USE_SAMPLE_WEIGHTS = "balanced" 


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
```
All the components of the config file would be explained below: 

<br/>
<br/>

### DATA_PATH 
It is the path where the data is stored. The user could change it according to its project structure.

### LOGS_PATH
This is the path where the `logs` directory is made which contains the `logs.log` file for each run.

### MODEL_TYPE 
This is used initialize the type of run. It initializes the steps according to the type of model which is needed to be trained 
```py
MODEL_TYPE = "ML"
```

### TRAIN_PERCENT
This is a decimal number and specifies the size of training set split and it should be less than 1. If `TRAIN_PERCENT` is 0.7 that would mean that the size of the training split would contain 70% of the total data and testing split would contain 30% of the data.
```py
TRAIN_PERCENT = decimal && <1
```

### NEW_DIMENSIONS
When an image is loaded , then it would be converted to the value given in the `NEW_DIMENSION`. This value should be a 1D tuple which contains 2 integer numbers. If value of `NEW_DIMENSION` is (256,256) then it means that all the images would be converted to the dimensions of 256px width and 256px height because the model expects all images to be of same size.
```py
NEW_DIMENSIONS = Tuple(int,int)
```

### GRAYSCALE
The value of `GRAYSCALE` should be either `True` or `False`. If the `GRAYSCALE` is set to `True` then the images which are loaded would be converted to grayscale else if it is set to `False` then the images would be loaded as coloured.
```py
GRAYSCALE = True || False
```

### APPLY_DIM_REDUCTION
The value of `APPLY_DIM_REDUCTION` should be either `True` or `False`. If it is set to true then only the dimension reduction algorithm would be applied on the data.
```py
APPLY_DIM_REDUCTION = True || False
```

### DIM_REDUCTION_ALGO
This value specifies the algorithm to be used for dimensionality reduction.
```py
DIM_REDUCTION_ALGO = "PCA" || "t-SNE"
```

### DIM_REDUCTION_PARAMS
This should be a dictionary which contains the parameters for the dimensionality reduction algorithm. User could get the parameters from the documentation of the algorithms given below:

- "PCA" : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 
- "t-SNE" : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

### APPLY_SCALING
The value of `APPLY_SCALING` could be either `True` or `False`. If the value of the `APPLY_SCALING` is set to `True` only then the scaling algorithm would be applied but when its value is set to `False` then the scaling algorithm would not be applied to.
```py
APPLY_SCALING = True || False
```

### SCALER_TYPE
This value specifies the algorithm to be used for scaling the data.
```py
SCALER_TYPE = "StandardScaler" || "MinMaxScaler"
```

### SCALER_PARAMS
This should be a dictionary which contains the parameters for the scaling algorithm. User could get the parameters from the documentation of the algorithms given below:

- "StandardScaler" : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) 
- "MinMaxScaler" : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 


### MODEL_NAME
This value specifies the ML model to be used.
```py
MODEL_NAME = "LogisticRegression",
             "GaussianNaiveBayes",
             "MultinomialNaiveBayes", 
             "BernoulliNaiveBayes",
             "ComplementNaiveBayes",
             "DescisionTree",
             "SupportVectorClassifier",
             "AdaBoostClassifier",
             "GradientBoostingClassifier",
             "RandomForestClassifier",
             "XGBoost" 
```


### USE_SAMPLE_WEIGHTS
This value specifies the types of the weights to be used for each class in the dataset.
```py
USE_SAMPLE_WEIGHTS = None || "balanced"
```


### RANDOM_SEARCH_CV_MODEL_DISTRIBUTIONS
This value should be a dictionary consisting a list for all the parameters which should be tried for the model. Each key would represent a parameter for the model and the value for that parameter should be a list. This dictionary would be passed on to the random search algorithm with the model which would return the best parameters and the model trained with the best parameters. Consider this for example
```py   
# MODEL
MODEL_NAME = "LogisticRegression" 
USE_SAMPLE_WEIGHTS = "balanced" 


# RANDOM SEARCH CV
RANDOM_SEARCH_CV_MODEL_DISTRIBUTIONS = dict(
    penalty = ['l1','l2','elasticnet',None],
    multi_class = ["multinomial"],
    verbose = [2],

)
```
Here for Logistic Regression model we have defined the parameters of the model which is `penalty` , `multi_class` and `verbose`. All the parameters are passed as list. The random search CV would use all these parameters and would give the best parameters for the model whoch in this case were:
```py
{ 
  'verbose': 2, 
  'penalty': 'l2', 
  'multi_class': 'multinomial', 
  'class_weight': {
                    0: 0.9996788696210661, 
                    1: 1.0016087516087515, 
                    2: 0.998716714789862
                  }
} 
```

The parameters for each model could be fetched from the documentation given below:

- LogisticRegression : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- GaussianNaiveBayes : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- MultinomialNaiveBayes : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
- BernoulliNaiveBayes : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)
- ComplementNaiveBayes : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB)
- DescisionTree : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- SupportVectorClassifier : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- AdaBoostClassifier : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- GradientBoostingClassifier : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- RandomForestClassifier : [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- XGBoost : [documentation](https://xgboost.readthedocs.io/en/stable/parameter.html#global-configuration)


### RANDOM_SEARCH_CV_PARAMS
This should be a dictionary consisting of the parameters to passed to `RandomizedSearchCV` instead of the `RANDOM_SEARCH_CV_MODEL_DISTRIBUTIONS`. These parameters could be fetched from the documentation linked [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)


### LOG_TO_MLFLOW
This should be either `True` or `False`. If it is set to true the metrics, parameters and model would be looged to the `MLFlow` but if it is false the they won't be logged.
```py
LOG_TO_MLFOW = True || False 
```

### EXPERIMENT_NAME 
This is for the `EXPERIMENT_NAME` for the MLFlow.