from pdpbox import pdp
from explainer_methods import *

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
            and data.columns.tolist()==self.use_columns):
            return self.model.predict(data)
        else:
            return self.model.predict(self.transform(data)[0][self.use_columns])

    def predict_proba(self, data):
        assert hasattr(self.model, "predict_proba")

        if data.columns.tolist()==self.use_columns:
            return self.model.predict_proba(data)
        else:
            return self.model.predict_proba(self.transform(data)\
                                            [0][self.use_columns])


class ExplainerBunch:
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
            self._preds = self.model.predict(self.X)
        return self._preds

    @property
    def pred_probas(self):
        if self._pred_probas is None and hasattr(self.model, 'predict_proba'):
            self._pred_probas =  self.model.predict_proba(self.X)
            if len(self._pred_probas.shape) == 2 and self._pred_probas.shape[1]==2:
                # if binary classifier, take prediction of positive class.
                self._pred_probas = self.pred_probas[:,1]
        return self._pred_probas

    @property
    def importances(self):
        if self._importances is None:
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

    def permutation_importances_df(self, topx=None, cutoff=None, cats=False):
        importance_df = self.importances_cats.reset_index() if cats \
                                else self.importances.reset_index()

        if topx is None: topx = len(importance_df)
        if cutoff is None: cutoff = importance_df.Importance.min()
        return importance_df[importance_df.Importance > cutoff].head(topx)

    def pdp(self, feature_name, num_grid_points=20):
        assert feature_name in self.X.columns, \
            f"{feature_name} not in columns of dataset"

        pdp_result = pdp.pdp_isolate(
                model=self.model, dataset=self.X, model_features=self.X.columns,
                num_grid_points=num_grid_points, feature=feature_name)
        return pdp_result

    def get_feature_prediction_tuple(self, idx, feature_name):
        assert feature_name in self.X.columns,\
            f"{feature_name} not in columns of dataset"
        assert idx >= 0 and idx < len(self.X),\
            f"index {idx} out of range of dataset"

        feature_value = self.X[feature_name].iloc[idx]
        prediction = self.pred_probas[idx] if self.pred_probas is not None \
                            else self.preds[idx]
        return (feature_value, prediction)

    def is_classifier(self):
        return False


class TreeExplainer(ExplainerBunch):
    def __init__(self, model_bunch, raw_data, metric, classifier=True):
        super().__init__(model_bunch, raw_data, metric)

        self.classifier = classifier
        self._shap_explainer, self._shap_base_value,  = None, None
        self._shap_values, self._shap_interaction_values = None, None
        self._mean_abs_shap, self._mean_abs_shap_cats = None, None
        self._shadow_trees = None

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    @property
    def shap_base_value(self):
        if self._shap_base_value is None:
            self._shap_base_value = self.shap_explainer.expected_value
        return self._shap_base_value

    @property
    def shap_values(self):
        if self._shap_values is None:
            self._shap_values = self.shap_explainer.shap_values(self.X)[1]
        return self._shap_values

    @property
    def shap_interaction_values(self):
        if self._shap_interaction_values is None:
            self._shap_interaction_values = normalize_shap_interaction_values(
                self.shap_explainer.shap_interaction_values(self.X)[1],
                self.shap_values)
        return self._shap_interaction_values

    @property
    def shadow_trees(self):
        if self._shadow_trees is None:
            self._shadow_trees = shadow_trees = get_shadow_trees(self.model, self.X, self.y)
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
            idx = np.random.choice(self.y[(self.y.isin(y)) &
                                          (self.pred_probas >= pred_proba_min) &
                                          (self.pred_probas <= pred_proba_max)].index)
        else:
            idx = np.random.choice(self.y[(self.y.isin(y))].index)
        return idx

    def mean_abs_shap_df(self, topx=None, cutoff=None, cats=False):
        shap_df = self.mean_abs_shap_cats if cats else self.mean_abs_shap

        if topx is None: topx = len(shap_df)
        if cutoff is None: cutoff = shap_df['MEAN_ABS_SHAP'].min()
        return shap_df[shap_df['MEAN_ABS_SHAP'] > cutoff].head(topx)

    def shap_top_interactions(self, col_name, topx=None):
        col_idx = self.X.columns.get_loc(col_name)
        top_interactions = self.X.columns[np.argsort(-np.abs(
                    self.shap_interaction_values[:, col_idx, :]).mean(0))].tolist()
        if topx is None: topx = len(top_interactions)
        return top_interactions[:topx]

    def shap_interaction_values_by_col(self, col_name):
        return self.shap_interaction_values[:, self.X.columns.get_loc(col_name), :]

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

    def shadowtree_df(self, tree_idx, idx):
        assert tree_idx >= 0 and tree_idx < len(self.shadow_trees), \
        f"tree index {tree_idx} outside 0 and number of trees ({len(self.shadow_trees)}) range"
        assert idx >= 0 and tree_idx < len(self.X), \
        f"tree index {idx} outside 0 and size of X ({len(self.X)}) range"
        return get_shadowtree_df(self.shadow_trees[tree_idx], self.X.iloc[idx])

    def isclassifier(self):
        return self.classifier
