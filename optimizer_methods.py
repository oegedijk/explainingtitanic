import numpy as np
import pandas as pd

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK


def classifier_optimize(tree_data=None, lin_data=None, models=['RandomForestClassifier'], metric='accuracy', 
                greater_is_better=True, needs_proba=True, cv=5, n_jobs=5, 
                n_evals=100, trials = None, show_progressbar=True):
    """
    returns the best model with the best hyperparameters.
    
    tree_data and lin_data are tuples of (X_train, y_train) with data prepared for tree based models 
    or linear models, respectively. 
    
    """
    if set(models) & set(['RandomForestClassifier', 
                          'BalancedRandomForestClassifier', 
                          'XGBClassifier']):
        assert tree_data is not None, \
            "Need to proide tree_data if you're going to fit a tree based model!"
        X_train, y_train = tree_data[0], tree_data[1]
        sample_ratio = (y_train.sum()+1)/(len(y_train)-y_train.sum())
        
    if 'LogisticRegression' in models:
        assert lin_data is not None, \
            "Need to provide lin_data if you're going to fit a LogisticRegression!"
        
    if trials is None:
        trials = Trials()
        
    model_space = []
    
    if 'RandomForestClassifier' in models:
        model_space.append(
                {
                'model_type' : 'RandomForestClassifier',
                'n_estimators': hp.quniform('n_estimators', 80, 200, 10),
                'max_depth': hp.choice('dtree_max_depth',
                     [None, hp.quniform('dtree_max_depth_int', 5, 200, 5)]),
                'min_samples_split': hp.quniform('min_samples_split', 2, 32, 1),
                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 16, 1),
                'min_impurity_decrease' : hp.loguniform('min_impurity_decrease', np.log(1e-10), np.log(1e-3)),

                'max_features' : hp.uniform('max_features', 0.05, 1.0),
                'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
                })
        
    if 'BalancedRandomForestClassifier' in models:
        model_space.append(
                {
                'model_type' : 'BalancedRandomForestClassifier',
                'n_estimators': hp.quniform('n_estimators2', 80, 200, 10),
                'max_depth': hp.choice('dtree_max_depth2',
                     [None, hp.quniform('dtree_max_depth_int2', 5, 200, 5)]),
                'min_samples_split': hp.quniform('min_samples_split2', 2, 32, 1),
                'min_samples_leaf': hp.quniform('min_samples_leaf2', 1, 16, 1),
                'min_impurity_decrease' : hp.loguniform('min_impurity_decrease2', np.log(1e-10), np.log(1e-3)),
                'max_features' : hp.uniform('max_features2', 0.05, 1.0),
                'sampling_strategy' : hp.uniform('sampling_strategy', sample_ratio, 1.0)
                })
        
    if 'XGBClassifier' in models:
        model_space.append(
                {
                'model_type' : 'XGBClassifier',
                'learning_rate' : hp.loguniform('xgb_learning_rate', np.log(0.0001), np.log(0.1)),
                'n_estimators' : hp.quniform('xgb_n_estimators', 30, 250, 5),
                'max_depth': hp.quniform('xgb_max_depth_int', 3, 25, 1),
                'min_child_weight' : hp.quniform('xgb_min_child_weight_int', 1, 7, 1),
                'gamma' : hp.uniform('xgb_gamma', 0.01, 0.4),
                'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.5, 0.99),
                'subsample' : hp.uniform('xgb_subsample', 0.5, 0.99),   
                'reg_alpha': hp.loguniform('xgb_alpha', np.log(1e-10), np.log(0.1)),
                'reg_lambda': hp.loguniform('xgb_lambda', np.log(1e-10), np.log(0.1)),
                'scale_pos_weight' : hp.uniform('xgb_scale_post_weight', 1.0, 5 * 1/sample_ratio)
                })
        
    if 'LogisticRegression' in models:
        model_space.append(
                {
                'model_type' : 'LogisticRegression', 
                'solver' : 'saga',
                'penalty': hp.choice('lr_penalty',
                     ['l1', 'l2', 'elasticnet']),
                'C' :  hp.loguniform('lr_C', np.log(1e-10), np.log(10)),
                'l1_ratio' : hp.uniform('lr_l1_ratio', 0, 1),
                'class_weight' : hp.choice('lr_class_weight', [None, 'balanced']),  
                })
    
    search_space = hp.choice('classifier_type', model_space)
        
    
    int_parameters = ['n_estimators',  'max_depth', 'min_samples_split', 'min_samples_leaf']

    def model_wrapper(space):
        """
        wrapper function that takes in a parameter space, and returns
        the result of a trained RandomForest model.
        """
        local_space = space.copy()
        model_type = local_space['model_type']
        del local_space['model_type']
        
        for param in int_parameters:
            if param in local_space and local_space[param] is not None:
                local_space[param] = int(local_space[param])

        # default: use the data set prepared for tree based methods:
        
        
        if model_type=='RandomForestClassifier':
            X_train, y_train, X_val, y_val = tree_data
            model = RandomForestClassifier(**local_space, n_jobs=n_jobs)
        elif model_type== 'BalancedRandomForestClassifier':
            X_train, y_train, X_val, y_val = tree_data
            model = BalancedRandomForestClassifier(**local_space, n_jobs=n_jobs)   
        elif model_type == 'XGBClassifier':
            X_train, y_train, X_val, y_val = tree_data
            model = XGBClassifier(**local_space)
        elif model_type == 'LogisticRegression':
            assert lin_data is not None
            X_train, y_train, X_val, y_val = lin_data
            model = LogisticRegression(**local_space)
            
        if not isinstance(metric, str):
            scorer = make_scorer(metric, greater_is_better, needs_proba)
        else:
            scorer = metric
        if cv > 0:    
            scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=n_jobs)
            score = scores['test_score'].mean()
        else: 
            scores = []
            for i in range(10):
                model.fit(X_train, y_train)
                if needs_proba:
                    score = metric(y_val, model.predict_proba(X_val)[:,1])
                else:
                    score = metric(y_val, model.predict(X_val))
                scores.append(score)
            score = np.mean(scores)
        
        return {'loss': -score, 'status': STATUS_OK, 'space': space}

    best = fmin(model_wrapper, search_space, 
                algo=tpe.suggest, trials=trials, 
                max_evals=len(trials) + n_evals, verbose=1, show_progressbar=True)

    best_model = trials.best_trial['result']['space']
    
    # convert hyperopt's floats to ints:
    for param in int_parameters:
        if param in best_model and best_model[param] is not None:
            best_model[param] = int(best_model[param])
            
    return best_model, trials


def topx_precision_score(y_true, y_pred, topx=100):
    """ 
    Returns the precision score for the topx highest predictions.
    This is useful for cases where you are resource constrained and will
    only act upon the highest scores of your model, and so you are interested
    in the precision of only the highest scores.
    """ 
    assert len(y_true)==len(y_pred)
    assert topx < len(y_true)
    return y_true.values[np.argpartition(y_pred, -topx)[-topx:]].mean()


def topx_perc_precision_score(y_true, y_pred, topx_perc=0.10):
    """
    Returns the precision score for the topx_perc highest percentage predictions.
    This is useful for cases where you are resource constrained and will
    only act upon the highest scores of your model, and so you are interested
    in the precision of only the highest scores.
    """ 
    assert len(y_true)==len(y_pred)
    assert topx_perc > 0.0
    assert topx_perc < 1.0
    
    topx = int(float(topx_perc)*len(y_true))

    return topx_precision_score(y_true, y_pred, topx)


def get_data_subset(data_tuple, used_cols = None):
    """
    Given a tuple of (X_train, y_train)
        return (X_train[used_cols], y_train)
    or (X_train, y_train, X_test, y_test) 
        return (X_train[used_cols], y_train, X_test[used_cols], y_test)
    """ 
    assert len(data_tuple)==2 or len(data_tuple)==4, \
        """data_tuple should be either (X_train, y_train) or 
                                (X_train, y_train, X_test, y_test)"""

    if len(data_tuple) == 2:
        X_train, y_train = data_tuple
        if used_cols is not None:
            return (X_train[used_cols], y_train)
        return (X_train, y_train)
      
    if len(data_tuple) == 4:
        X_train, y_train, X_test, y_test = data_tuple
        if used_cols is not None:
            return (X_train[used_cols], y_train, X_test[used_cols], y_test)
        return (X_train, y_train, X_test, y_test)


def get_best_model_and_data(best_model, tree_data=None, lin_data=None, used_cols=None, 
                            tree_transformer=None, lin_transformer=None):
    """
    returns the best model with appropriate (tree of linear) data.
    
    if used_cols is given only returns those columns
    """
    transformer = None
    
    if best_model['model_type'] in ['RandomForestClassifier', 'BalancedRandomForestClassifier', 'XGBClassifier']:
        assert tree_data is not None,\
            "You need to provide tree_data tuple!"
        data = get_data_subset(tree_data, used_cols)
        if tree_transformer is not None:
            transformer = tree_transformer
            
    elif best_model['model_type'] in ['LogisticRegression']:
        assert lin_data is not None, \
            "You need to provide a lin_data tuple!"
        data = get_data_subset(lin_data, used_cols)
        if lin_transformer is not None:
            transformer = lin_transformer
        

    if best_model['model_type'] =='RandomForestClassifier':
        model = RandomForestClassifier(
                **{k:best_model[k] for k in best_model if k != 'model_type'})

    elif best_model['model_type'] =='BalancedRandomForestClassifier':
        model = BalancedRandomForestClassifier(
                **{k:best_model[k] for k in best_model if k != 'model_type'})

    elif best_model['model_type'] =='XGBClassifier':
        model = XGBClassifier(
                **{k:best_model[k] for k in best_model if k != 'model_type'})

    elif best_model['model_type'] =='LogisticRegression':
        model = LogisticRegression(
                **{k:best_model[k] for k in best_model if k != 'model_type'})
     
    if transformer is None:
        return model, data
    else:
        return model, data, transformer
        
    