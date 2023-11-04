# HOW TO CUSTOMIZE THE PROJECT

The user can customize the project by adding:
- Dimensionality Reduction Algorithm
- Scaler
- Custom Class Weights
- Model
- Metrics

We would tell you the steps for each of the above.

<br/>

---

<br/>

### ADDING NEW DIMENSIONALITY REDUCTION ALGORITHM
- For adding your new dimensionality reduction algorithm, add a new class to `utils/DimensionalityReduction.py`.
- This class should contain a class with `fit` method implemented in it where the algorithm would be initialized and would be trained.
- The `fit` method should take `train_images` and `test_images` and the parameter that it requires as `model_params`.
- Then it should be added to the dictionary in `steps/dimensionality_reduction.py`.
- Now you can use the key as `DIM_REDUCTION_ALGO` and you can give parameters for the algorithm to `DIM_REDUCTION_PARAMS` in `config.py`.

<br/>

---

<br/>

### ADDING NEW SCALING ALGORITHM 
- For adding your new scaling algorithm, add a new class to `utils/DataScaler.py`.
- This class should contain a class with `fit` method implemented in it where the algorithm would be initialized and would be trained.
- The `fit` method should take `train_images` and `test_images` and the parameter that it requires as `scaler_params`.
- Then it should be added to the dictionary in `steps/scaling.py`.
- Now you can use the key as `SCALER_TYPE` and you can give parameters for the algorithm to `SCALER_PARAMS` in `config.py`.

<br/>

---

<br/>

### ADDING CUSTOM CLASS WEIGHTS
- To add your custom class weights, you can create a function or you can initialize a value in the `utils/get_class_weights.py` with the key in the `elif` statement.
- You can then use this key as `USE_SAMPLE_WEIGHTS` in the `config.py` file.

<br/>

---

<br/>

### ADDING MODELS
- To add a new model , you should go to `steps/get_model.py` and add a key to the dictionary and value as the model.
- You can then use the key for `MODEL_NAME` in `config.py` file.

<br/>

---

<br/>

### ADDING METRICS
- To add a new metric , first go to `utils/MetricsCalculator.py` file and update the `MetricCalculatorTemplate` by adding a function which calculates and returns your metric but without the implementation part.
- The move to the main `MetricCalculator` class and implement the function there .
- Then move to the `steps/get_metrics.py` file and call the method there and store the metric in the `report` dictionary.
- The metric should be a number else it won't be logged to the MLFlow.  