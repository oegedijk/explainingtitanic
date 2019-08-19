import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

from sklearn_pandas import DataFrameMapper


def show_cardinality_of_cats(df):
    """
    Returns a sorted DataFrame with the cardinality of every
    categorical variable in df
    """
    cardinality_df = pd.DataFrame(columns=['Column', 'Cardinality'])
    for col in df.select_dtypes(include='object').columns:
        cardinality_df = cardinality_df.append(
                {
                    'Column': col,
                    'Cardinality': df[col].nunique()
                },
                ignore_index=True)

    cardinality_df = cardinality_df.sort_values('Cardinality', ascending=False)\
                                   .reset_index(drop=True)
    return cardinality_df


def clean_data(df, drop_columns=[], substring_drop_list=[],
                drop_dates=True, verbose=1):
    """
    drops columns from dataframe df.

    drop_columns: all columns in drop_columns will be dropped
    substring_drop_list: all columns that have a substring matching the strings
    in substring_drop_list will be dropped
    drop_dates: if True, trom all "datetime" column types
    """

    if drop_dates:
        date_columns = df.select_dtypes(include="datetime").columns.tolist()
    else:
        date_columns = []

    substring_drop_columns = []
    for substring in substring_drop_list:
        substring_drop_columns = substring_drop_columns + \
            [col for col in df.columns if substring in col]

    for col in drop_columns + date_columns + substring_drop_columns:
        if col in df.columns:
            if verbose:
                print(f'dropping {col}')
            df = df.drop([col], axis=1)
        else:
            if verbose:
                print(f'{col} no longer in d')
    return df


def get_transformed_X_y(raw_data_df, transformer, target, add_random=False):
    """
    Returns a tuple X, y of transformed predictors and labels.

    transformer should be an object with a .transform() methods that
    returns a transformed DataFrame.
    if add_random = True, then a random column is added. (this can be useful
    to compare feature importances against)
    """
    df =  transformer.transform(raw_data_df.copy())

    X = df.drop([target], axis=1)
    X.columns = [col.replace('<', '') for col in X.columns]
    y = df[target]

    if add_random:
        X['RANDOM_COLUMN'] = np.random.rand(len(X))
    return X, y


def gen_features(columns, classes=None, input_df=True,
                 suffix = None, alias = None):
    """
    Return a list of feature transformers.
    - alias to rename the column s
    - suffix to add an suffic to column (e.g. `_enc`)
    """
    if classes is None:
        return [(column, None) for column in columns]
    else:
        classes = [cls for cls in classes if cls is not None]

    # placeholder for all the
    feature_defs = []

    for column in columns:
        feature_transformers = []

        classes = [cls for cls in classes if cls is not None]
        if not classes:
            feature_defs.append((column, None))
        else:

            # collect all the transformer classes for this column:
            for definition in classes:
                if isinstance(definition, dict):
                    params = definition.copy()
                    klass = params.pop('class')
                    feature_transformers.append(klass(**params))
                else:
                    feature_transformers.append(definition())

            if not feature_transformers:
                # if no transformer classes found, then return as is (None)
                feature_transformers = None

            if input_df:
                if alias:
                    feature_defs.append((column,
                                         feature_transformers,
                                         {'input_df' : True, 'alias' : alias}))
                elif suffix:
                    feature_defs.append((column,
                                        feature_transformers,
                                        {'input_df' : True,
                                         'alias' : str(column)+str(suffix)}))
                else:
                    feature_defs.append((column,
                                         feature_transformers,
                                         {'input_df' : True}))

            else:
                if alias:
                    feature_defs.append((column,
                                        feature_transformers,
                                        {'alias' : alias}))
                elif suffix:
                    feature_defs.append((column,
                                         feature_transformers,
                                         {'alias' : str(column)+str(suffix)}))
                else:
                    feature_defs.append((column, feature_transformers))

    return feature_defs


class DummyTransform(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Dummy transformer to make sure column is retained in dataset
        even though it does not need to be transformerd.
        """
        self.verbose = verbose
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print('Fit: DummyTransform for: {}...'.format(self.name))
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print('Transform: DummyTransform for: {}...'.format(self.name))
        return X


class ExtremeValueFill(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Fills missing numerical values with -999
        """
        self.verbose = verbose
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print('Fit: Filling numerical NaN {}...'.format(self.name))
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print('Transform: Filling numerical NaN {}...'.format(self.name))
        return X.fillna(-999)


class NumericFill(TransformerMixin):
    def __init__(self, fill='ExtremeValue', name="", verbose = 0):
        """
        Fill missing numerical values with either -999 or the mean
        of the column.
        """
        self.verbose = verbose
        self.name = name
        self.fill = fill
        self._mean = None
        self._median = None
        self._filler = None

    def fit(self, X, y=None):
        # coerce column to either numeric, or to nan:
        if self.fill=='ExtremeValue':
            self._filler = -999
        elif self.fill=='mean':
            self._filler = pd.to_numeric(X, errors='coerce').mean()
        elif self.fill=='median':
            self._filler =  pd.to_numeric(X, errors='coerce').median()

        if self.verbose:
            print(f'Fit: Filling numerical NaN {self.name} with \
                    {self.fill}: {self._filler}...')
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: Filling numerical NaN {self.name} with \
                    {self.fill}: {self._filler}...')
        return pd.to_numeric(X, errors='coerce').fillna(self._filler)


class StandardScale(TransformerMixin):
    def __init__(self,  name="", verbose = 0):
        """
        Scale numerical features to mean=0, sd=1
        """
        self.verbose = verbose
        self.name = name
        self._mean = None
        self._sd = None

    def fit(self, X, y=None):
        self._mean = pd.to_numeric(X, errors='coerce').mean()
        self._sd = pd.to_numeric(X, errors='coerce').std()

        if self.verbose:
            print(f'Fit: StandarScaling {self.name}: \
                    ({self._mean}, {self._sd}...')
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: StandarScaling {self.name}: \
                    ({self._mean}, {self._sd}...')

        X = X.copy()
        X = pd.to_numeric(X, errors='coerce').fillna(self._mean)
        X -= self._mean
        if self._sd > 0:
            X /= self._sd
        return X.astype(np.float32)


class OneHot(TransformerMixin):
    def __init__(self, topx = None, name = "", verbose = 0):
        """
        One hot encodes column. Adds a column _na, and codes any label not
        seen in the training data as _na. Also makes sure all columns in the
        training data will get created in the transformed dataframe.
        If topx is given only encodes the topx most frequent labels,
        and labels everything else _na.
        """
        self.verbose = verbose
        self.topx = topx
        self.name = name

    def fit(self, X, y=None):
        if self.verbose:
            print('Fit: One-hot coding categorical variable {}...'\
                        .format(self.name))
        X = X.copy()
        if self.topx is None:
            # store the particular categories to be encoded:
            self.categories = X.unique()
            # Then do a simple pd.get_dummies to get the columns
            self.columns = pd.get_dummies(pd.DataFrame(X),
                                          prefix = "",
                                          prefix_sep = "",
                                          dummy_na=True).columns
        else:
            # only take the topx most frequent categories
            self.categories = [x for x in  X.value_counts()\
                                             .sort_values(ascending=False)\
                                             .head(self.topx).index]
            # set all the other categories to np.nan
            X.loc[~X.isin(self.categories)] = np.nan
            self.columns = pd.get_dummies(pd.DataFrame(X),
                                             prefix = "",
                                             prefix_sep = "",
                                             dummy_na=True).columns
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print(f'Transform: One-hot coding categorical variable {self.name}...')

        X = X.copy()
        # set all categories not present during fit() to np.nan:
        X.loc[~X.isin(self.categories)] = np.nan

        # onehot encode using pd.get_dummies
        X_onehot = pd.get_dummies(pd.DataFrame(X), prefix = "",
                                  prefix_sep = "", dummy_na=True)

        # add in columns missing in transform() that were present during fit()
        missing_columns = set(self.columns) - set(X_onehot.columns)
        for col in missing_columns:
            X_onehot[col]=0
        # make sure columns are in the same order
        X_onehot = X_onehot[self.columns]
        assert set(X_onehot.columns) == set(self.columns)
        # save the column names so that they can be assigned by DataFrameMapper
        self._feature_names = X_onehot.columns
        return X_onehot

    def get_feature_names(self):
        # helper function for sklearn-pandas to assign right column names
        return self._feature_names


class IsolationForestTransform(TransformerMixin):
    def __init__(self, name="", verbose = 0):
        """
        Dummy transformer to make sure column is retained in dataset
        even though it does not need to be transformerd.
        """
        self.verbose = verbose
        self.name = name
        self.fitted = False
        self.input_columns = []
        self._feature_names = []

    def fit(self, X, y=None):
        self.input_columns = list(X.columns)
        if self.verbose:
            print('Fit: IsolationForestTransform for: {}...'.format(self.name))
        self.isofor = IsolationForest(behaviour="new", contamination='auto')
        self.isofor.fit(X.replace([np.inf, -np.inf]).fillna(-999))
        self.fitted = True
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print('Transform: IsolationForestTransform for: {}...'.format(self.name))
        assert self.fitted

        return self.isofor.decision_function(
                        X.replace([np.inf, -np.inf]).fillna(-999))


def fit_transformer(X, target=None, topx=None, isofors = {},
                        numfill='ExtremeValue', stdscale=False, verbose=1):
    """
    Returns fitted transformer object.
    Fills all numerical columns except target with extreme value (-999).

    One hot encodes all categorical columns.
    """
    num_columns = list(set(X.select_dtypes(include=np.number).columns)
                            - set([target]))

    obj_columns = X.select_dtypes(include=['object']).columns.tolist()

    mappers = []

    # dummy transform to make sure target stays in the transformed dataframe:
    if target is not None:
        mappers = mappers + gen_features(
            columns=[target],
            classes = [{'class' : DummyTransform,
                        'name': target,
                        'verbose':verbose}],
            input_df = True
            )

    # Fill missing values in numerical columns:
    for num_col in num_columns:
        if not stdscale:
            mappers = mappers + gen_features(
                columns=[num_col],
                classes = [{'class' : NumericFill,
                            'fill' : numfill,
                            'name': num_col,
                            'verbose':verbose}],
                input_df = True
            )
        else:
            mappers = mappers + gen_features(
                columns=[num_col],
                classes = [{'class': NumericFill,
                            'fill': numfill,
                            'name': num_col,
                            'verbose': verbose},
                           {'class': StandardScale,
                            'name': num_col,
                            'verbose': verbose}],
                input_df = True
            )


    for onehot_col in obj_columns:
        mappers = mappers + gen_features(
            columns=[onehot_col],
            classes=[{'class' : OneHot,
                      'name': onehot_col,
                      'topx': topx,
                      'verbose':verbose}],
            input_df = True
        )

    for isofor_name, isofor_cols in isofors.items():
         mappers = mappers + gen_features(
            columns=[isofor_cols],
            classes = [{'class' : IsolationForestTransform,
                        'name': isofor_name,
                        'verbose': verbose}],
            alias = isofor_name,
            input_df = True
        )


    if verbose:
        print("Columns being transformed: ")
        print("numeric columns: ", num_columns)
        print("categorical columns: ", obj_columns)

    mapper = DataFrameMapper(mappers, df_out=True)

    if verbose: print("fitting transformer...")
    X = X.copy()
    if target is not None:
        mapper.fit(X, X[target])
    else:
        mapper.fit(X)
    return mapper
