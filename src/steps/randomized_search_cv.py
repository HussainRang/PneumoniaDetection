from sklearn.model_selection import RandomizedSearchCV

def randomized_search_cv( model , train_images , train_labels , model_distribution_params , random_search_params ):
    random_search = RandomizedSearchCV( model , model_distribution_params , **random_search_params )
    random_search.fit(train_images,train_labels)
    return random_search.best_params_,random_search.best_estimator_