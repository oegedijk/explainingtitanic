from pdpbox import pdp
from explainer_methods import *
from explainer_plots import *

class ModelBunch:
    def __init__(self, model, transformer, target, use_columns=None):
        self.model = model
        self.transformer = transformer
        self.target = target
        self.use_columns = use_columns
        
    def transform(self, raw_data):
        transformed_data = self.transformer.transform(raw_data)
        if self.use_columns is not None:
            if self.target in transformed_data.columns:
                return (transformed_data[self.use_columns], 
                        transformed_data[self.target])
            else: 
                return transformed_data[self.use_columns], None
        else:
            if self.target in transformed_data.columns:
                return (transformed_data.drop([self.target], axis=1), 
                        transformed_data[self.target])
            else:
                return transformed_data.drop([self.target], axis=1), None
            
    def predict(self, data):
        if (self.use_columns is not None 
            and set(data.columns.tolist())==set(self.use_columns)):
            return self.model.predict(data)
        else:
            return self.model.predict(self.transform(data)[0][self.use_columns])
    
    def predict_proba(self, data):
        assert hasattr(self.model, "predict_proba")
        
        if set(data.columns.tolist())==set(self.use_columns):
            return self.model.predict_proba(data)
        else:
            return self.model.predict_proba(
                                self.transform(data)[0][self.use_columns])


class BaseExplainer:
    def __init__(self, model_bunch, raw_data, metric, permutation_cv=None):
        self.model  = model_bunch.model
        self.transformer = model_bunch.transformer
        self.target  = model_bunch.target
        self.use_columns = model_bunch.use_columns

        self.raw_data = raw_data
        self.metric = metric
        self.permutation_cv = permutation_cv
        self.cats = self.raw_data.select_dtypes(include='object').columns.tolist()
        self.X = self.transformer.transform(self.raw_data)[self.use_columns].reset_index(drop=True)
        self.y = self.raw_data[self.target].reset_index(drop=True)

        self.columns = self.X.columns.tolist()
        
        self._preds, self._pred_probas = None, None            
        self._importances, self._importances_cats = None, None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X.iloc[idx], self.y[idx]
    
    @property
    def preds(self):
        if self._preds is None:
            print("Calculating predictions...")
            self._preds = self.model.predict(self.X)
        return self._preds
    
    @property
    def pred_probas(self):
        if self._pred_probas is None and hasattr(self.model, 'predict_proba'):
            print("Calculating prediction probabilities...")
            self._pred_probas =  self.model.predict_proba(self.X)
            if len(self._pred_probas.shape) == 2 and self._pred_probas.shape[1]==2:
                # if binary classifier, take prediction of positive class. 
                self._pred_probas = self.pred_probas[:,1]
        return self._pred_probas 
    
    @property
    def importances(self):
        if self._importances is None:
            print("Calculating importances...")
            if self.permutation_cv is None:
                self._importances = permutation_importances(
                            self.model, self.X, self.y, self.metric)
            else:
                self._importances = cv_permutation_importances(
                            self.model, self.X, self.y, self.metric,
                            cv=self.permutation_cv)
        return self._importances
    
    @property
    def importances_cats(self):
        if self._importances_cats is None:
            if self.permutation_cv is None:
                self._importances_cats = permutation_importances(
                            self.model, self.X, self.y, self.metric, self.cats)
            else:
                self._importances_cats = cv_permutation_importances(
                            self.model, self.X, self.y, self.metric, seld.cats,
                            cv=self.permutation_cv)
        return self._importances_cats

    def calculate_properties(self):
        _ = (self.preds, self.pred_probas, 
                        self.importances, self.importances_cats)
        return 

    def permutation_importances_df(self, topx=None, cutoff=None, cats=False):
        importance_df = self.importances_cats.reset_index() if cats \
                                else self.importances.reset_index()
    
        if topx is None: topx = len(importance_df)
        if cutoff is None: cutoff = importance_df.Importance.min()
        return importance_df[importance_df.Importance > cutoff].head(topx)

    def pdp(self, feature_name, num_grid_points=20):
        assert feature_name in self.X.columns or feature_name in self.cats, \
            f"{feature_name} not in columns of dataset"

        features = get_feature_dict(self.X.columns, self.cats)[feature_name]

        # if only a single value (i.e. not onehot encoded, take that value 
        # instead of list):
        if len(features)==1: features=features[0]
        pdp_result = pdp.pdp_isolate(
                model=self.model, dataset=self.X, 
                model_features=self.X.columns, 
                num_grid_points=num_grid_points, feature=features)
        return pdp_result

    def get_feature_plus_prediction(self, idx, feature_name):
        assert (feature_name in self.X.columns) or (feature_name in self.cats),\
            f"{feature_name} not in columns of dataset"
        assert idx >= 0 and idx < len(self.X),\
            f"index {idx} out of range of dataset"

        if feature_name in self.X.columns:
            feature_value = self.X[feature_name].iloc[idx]
        elif feature_name in self.cats:
            feat_dict = get_feature_dict(self.X.columns, self.cats)
            cat_features = feat_dict[feature_name]
            feature_value = np.argmax(self.X[cat_features].loc[idx])

        prediction = self.pred_probas[idx] if self.pred_probas is not None \
                            else self.preds[idx]
        return (feature_value, prediction)
    
    def is_classifier(self):
        return False


class TreeClassifierExplainer(BaseExplainer):
    def __init__(self, model_bunch, raw_data, metric, labels=None):
        super().__init__(model_bunch, raw_data, metric)

        self._X_cats, self._columns_cats = None, None
        self._shap_explainer, self._shap_base_value,  = None, None
        self._shap_values, self._shap_interaction_values = None, None
        self._shap_values_cats, self._shap_interaction_values_cats = None, None
        self._mean_abs_shap, self._mean_abs_shap_cats = None, None

        self._shadow_trees = None
        self.labels = labels if labels is not None else ['0', '1']

    @property 
    def X_cats(self):
        if self._X_cats is None:
            self._X_cats, self._shap_values_cats = \
                merge_categorical_shap_values(self.X, self.shap_values, self.cats)
        return self._X_cats
    
    @property 
    def columns_cats(self):
        if self._columns_cats is None:
            self._columns_cats = self.X_cats.columns.tolist()
        return self._columns_cats

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            print("Generating shap TreeExplainer...")
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer
    
    @property 
    def shap_base_value(self):
        if self._shap_base_value is None:
            try:
                self._shap_base_value = self.shap_explainer.expected_value[1]
            except:
                self._shap_base_value = self.shap_explainer.expected_value
        return self._shap_base_value
    
    @property 
    def shap_values(self):
        if self._shap_values is None:
            print("Calculating shap values...")
            try:
                self._shap_values = self.shap_explainer.shap_values(self.X)[1]
            except:
                self._shap_values = self.shap_explainer.shap_values(self.X)
        return self._shap_values

    @property 
    def shap_values_cats(self):
        if self._shap_values_cats is None:
            self._X_cats, self._shap_values_cats = \
                merge_categorical_shap_values(self.X, self.shap_values, self.cats)
        return self._shap_values_cats
    
    @property 
    def shap_interaction_values(self):
        if self._shap_interaction_values is None:
            print("Calculating shap interaction values...")
            self._shap_interaction_values = normalize_shap_interaction_values(
                self.shap_explainer.shap_interaction_values(self.X)[1],
                self.shap_values)
        return self._shap_interaction_values

    @property 
    def shap_interaction_values_cats(self):
        if self._shap_interaction_values_cats is None:
            print("Calculating categorical shap interaction values...")
            self._shap_interaction_values_cats = \
                merge_categorical_shap_interaction_values(
                    self.X, self.X_cats, self.shap_interaction_values)
        return self._shap_interaction_values_cats
    
    @property
    def shadow_trees(self):
        if self._shadow_trees is None:
            print("Generating shadow trees...")
            self._shadow_trees = get_shadow_trees(self.model, self.X, self.y)
        return self._shadow_trees
    
    @property
    def mean_abs_shap(self):
        if self._mean_abs_shap is None:
            self._mean_abs_shap = mean_absolute_shap_values(
                                self.columns, self.shap_values)
        return self._mean_abs_shap
    
    @property 
    def mean_abs_shap_cats(self):
        if self._mean_abs_shap_cats is None:
            self._mean_abs_shap_cats = mean_absolute_shap_values(
                                self.columns, self.shap_values, self.cats)
        return self._mean_abs_shap_cats

    def calculate_properties(self):
        super(TreeClassifierExplainer, self).calculate_properties()
        _ = (self.shap_base_value, self.shap_values, 
                        self.shap_interaction_values, self.shadow_trees, 
                        self.mean_abs_shap, self.mean_abs_shap_cats,
                        self.shap_values_cats, 
                        self.shap_interaction_values_cats)
        return

    def __getitem__(self, index):
        if self.pred_probas is not None:
            return (self.X.iloc[index], self.y[index], 
                    self.pred_probas[index], self.shap_values[index])
        else:
            return (self.X.iloc[index], self.y[index], 
                    self.pred[index], self.shap_values[index])
        
    def random_index(self, y=None, pred_proba_min=None, pred_proba_max=None):
        if y is None:
            y = self.y.unique().tolist()
        else:
            if not isinstance(y, list):
                y = [y]
        idx = None
        if self.pred_probas is not None:
            if pred_proba_min is None: pred_proba_min = self.pred_probas.min()
            if pred_proba_max is None: pred_proba_max = self.pred_probas.max()
            if len(self.y[(self.y.isin(y)) & 
                            (self.pred_probas >= pred_proba_min) & 
                            (self.pred_probas <= pred_proba_max)]) > 0:
                idx = np.random.choice(self.y[(self.y.isin(y)) & 
                                            (self.pred_probas >= pred_proba_min) & 
                                            (self.pred_probas <= pred_proba_max)].index)
            else: idx = None
        else:
            if len(self.y[(self.y.isin(y))]) > 0:
                idx = np.random.choice(self.y[(self.y.isin(y))].index)
            else: idx = None
        return idx
    
    def mean_abs_shap_df(self, topx=None, cutoff=None, cats=False):
        shap_df = self.mean_abs_shap_cats if cats else self.mean_abs_shap
        
        if topx is None: topx = len(shap_df)
        if cutoff is None: cutoff = shap_df['MEAN_ABS_SHAP'].min()
        return shap_df[shap_df['MEAN_ABS_SHAP'] > cutoff].head(topx)
    
    def shap_top_interactions(self, col_name, topx=None, cats=False):
        if cats:
            col_idx = self.X_cats.columns.get_loc(col_name)
            top_interactions = self.X_cats.columns[np.argsort(-np.abs(
                        self.shap_interaction_values_cats[:, col_idx, :]).mean(0))].tolist()
            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]
        else:
            col_idx = self.X.columns.get_loc(col_name)
            top_interactions = self.X.columns[np.argsort(-np.abs(
                        self.shap_interaction_values[:, col_idx, :]).mean(0))].tolist()
            if topx is None: topx = len(top_interactions)
            return top_interactions[:topx]

    def shap_interaction_values_by_col(self, col_name, cats=False):
        if cats:
            return self.shap_interaction_values_cats[:, 
                        self.X_cats.columns.get_loc(col_name), :]
        else:
            return self.shap_interaction_values[:, 
                        self.X.columns.get_loc(col_name), :]
    
    def importances_df(self, type="permutation", topx=None, cutoff=None, cats=False):
        if type=='permutation':
            return self.permutation_importances_df(topx, cutoff, cats)
        elif type=='shap':
            return self.mean_abs_shap_df(topx, cutoff, cats)
    
    def precision_df(self, bin_size=0.05):
        assert self.pred_probas is not None
        return get_precision_df(self.pred_probas, self.y.values, bin_size)
        
    def contrib_df(self, index, topx=None, cutoff=None):
        """
        Return a contrib_df DataFrame that lists the contribution of each input
        variable for the RandomForrestClassifier predictor rf_model. 
        """    
        return get_contrib_df(self.shap_base_value, self.shap_values[index], 
                              self.columns, self.raw_data.iloc[index], 
                              self.cats, topx, cutoff)

    def contrib_summary_df(self, idx, topx=None, cutoff=None):
        return get_contrib_summary_df(self.contrib_df(idx, topx, cutoff))
    
    def shadowtree_df(self, tree_idx, idx):
        assert tree_idx >= 0 and tree_idx < len(self.shadow_trees), \
        f"tree index {tree_idx} outside 0 and number of trees ({len(self.shadow_trees)}) range"
        assert idx >= 0 and tree_idx < len(self.X), \
        f"tree index {idx} outside 0 and size of X ({len(self.X)}) range"
        return get_shadowtree_df(self.shadow_trees[tree_idx], self.X.iloc[idx])

    def shadowtree_df_summary(self, tree_idx, idx):
        return shadowtree_df_summary(self.shadowtree_df(tree_idx, idx))

    def plot_precision(self, bin_size=0.10, cutoff=0.5):
        precision_df = self.precision_df(bin_size=bin_size)
        return plotly_precision_plot(precision_df, cutoff=cutoff)

    def plot_importances(self, type='permutation', topx=10, cats=False):
        importances_df = self.importances_df(type=type, topx=topx, cats=cats)
        return plotly_importances_plot(importances_df)  
    
    def plot_contributions(self, idx, topx=None, cutoff=None):
        contrib_df = self.contrib_df(idx, topx, cutoff)
        return plotly_contribution_plot(contrib_df)

    def plot_shap_summary(self, topx=10, cats=False):
        if cats:
            return plotly_shap_scatter_plot(
                                self.shap_values_cats, 
                                self.X_cats, 
                                self.importances_df(type='shap', topx=topx, cats=True)\
                                        ['Feature'].values.tolist())
        else:
            return plotly_shap_scatter_plot(
                                self.shap_values, 
                                self.X, 
                                self.importances_df(type='shap', topx=topx)\
                                        ['Feature'].values.tolist())

    def plot_shap_interaction_summary(self, col, topx=10, cats=False):
        interact_cols = self.shap_top_interactions(col, cats=cats)
        if cats:
            return plotly_shap_scatter_plot(
                self.shap_interaction_values_by_col(col, cats=cats), 
                self.X_cats, interact_cols[:topx])
        else:
            return plotly_shap_scatter_plot(
                self.shap_interaction_values_by_col(col), 
                self.X, interact_cols[:topx])

    def plot_confusion_matrix(self, cutoff=0.5, normalized=False):
        return plotly_confusion_matrix(
                self.y, self.pred_probas,
                cutoff=cutoff, normalized=normalized, 
                labels=self.labels)

    def plot_roc_auc(self, cutoff=0.5):
        return plotly_roc_auc_curve(self.y, self.pred_probas, cutoff=cutoff)

    def plot_pr_auc(self, cutoff=0.5):
        return plotly_pr_auc_curve(self.y, self.pred_probas, cutoff=cutoff)

    def plot_pdp(self, idx, col, num_grid_points=20):
        pdp_result = self.pdp(col, num_grid_points)
        feature_val, pred = self.get_feature_plus_prediction(idx, col)
        return plotly_pdp(pdp_result, idx, feature_val, pred, feature_name=col)  

    def plot_trees(self, idx):
        return plotly_tree_predictions(self.model, self.X.iloc[[idx]])

    def plot_dependence(self, col, color_col=None, highlight_idx=None, cats=False):
        if cats:
            return plotly_dependence_plot(self.X_cats, self.shap_values_cats, 
                                            col, color_col, 
                                            highlight_idx=highlight_idx)
        else:
            return plotly_dependence_plot(self.X, self.shap_values, 
                                            col, color_col, 
                                            highlight_idx=highlight_idx)

    def plot_interaction_dependence(self, col, interact_col, highlight_idx=None, cats=False):
        return plotly_dependence_plot(self.X_cats, 
                self.shap_interaction_values_by_col(col, cats), 
                interact_col, col, highlight_idx=highlight_idx)


    