import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance
from skrebate import ReliefF, MultiSURFstar

from utils.metrics import compute_mrae


class FSMethod(object):

    def __init__(self, return_scores=False, seed_value=3242):
        self.df_features = None
        self.selected_feature_indices = []
        self.selected_feature_names = []
        self.verbose = False
        self.df_feature_scores = None
        self.return_scores = return_scores
        self.seed_value = seed_value

    def fit(self, df_features: pd.DataFrame, y_label: np.array, k: int,
            list_categorical_vars: list, list_numerical_vars: list,
            verbose):

        self.df_features = df_features

    def extract_features(self, k=None):
        x_features = self.df_features.values
        if k is not None and k < len(self.selected_feature_indices):
            return x_features[:, self.selected_feature_indices[:k]]
        x_features_filtered = x_features[:, self.selected_feature_indices]

        df_features_selected = self.df_features.iloc[:, self.selected_feature_indices]

        if not self.return_scores:
            return df_features_selected
        else:
            return df_features_selected, self.df_feature_scores

    def _validate_categorical_columns(self, data, categorical_columns):
        """
        Check whether ``discrete_columns`` exists in ``train_data``.
        Args:
            data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            categorical_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(data, pd.DataFrame):
            invalid_columns = set(categorical_columns) - set(data.columns)
        elif isinstance(data, np.ndarray):
            invalid_columns = []
            for column in categorical_columns:
                if column < 0 or column >= data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')


class Relief(FSMethod):

    def __init__(self, return_scores):
        super().__init__(return_scores)

    def fit(self, df_features: pd.DataFrame, y_label: np.array, k: int,
            list_categorical_vars: list, list_numerical_vars: list,
            verbose=False):

        self.df_features = df_features
        x_features = df_features.values

        relief = ReliefF(n_features_to_select=k, n_neighbors=100)
        relief.fit(x_features, y_label)
        x_selected_features, list_selected_features_indices, scores_sorted = relief.transform(x_features)

        list_selected_feature_names = df_features.iloc[:, list_selected_features_indices].columns.values

        df_feature_scores = pd.DataFrame(np.zeros((len(list_selected_features_indices), 2)), columns=['var_name', 'score'])
        df_feature_scores['var_name'] = np.array(list_selected_feature_names)
        df_feature_scores['score'] = scores_sorted

        self.selected_feature_indices = list_selected_features_indices[:k]
        self.selected_feature_names = list_selected_feature_names[:k]
        self.df_feature_scores = df_feature_scores


class Surf(FSMethod):

    def __init__(self, return_scores):
        super().__init__(return_scores)

    def fit(self, df_features: pd.DataFrame, y_label: np.array, k: int,
            list_categorical_vars: list, list_numerical_vars: list,
            verbose=False):

        self.df_features = df_features
        x_features = df_features.values

        surf = MultiSURFstar(n_features_to_select=k)
        surf.fit(x_features, y_label)
        x_selected_features, list_selected_features_indices, scores_sorted = surf.transform(x_features)

        list_selected_feature_names = df_features.iloc[:, list_selected_features_indices].columns.values

        df_feature_scores = pd.DataFrame(np.zeros((len(list_selected_features_indices), 2)), columns=['var_name', 'score'])
        df_feature_scores['var_name'] = np.array(list_selected_feature_names)
        df_feature_scores['score'] = scores_sorted

        self.selected_feature_indices = list_selected_features_indices[:k]
        self.selected_feature_names = list_selected_feature_names[:k]
        self.df_feature_scores = df_feature_scores


class MRMR(FSMethod):

    def __init__(self, return_scores):
        super().__init__(return_scores)
        self.selected_feature_names = None

    def fit(self, df_features: pd.DataFrame, y_label: np.unique, k: int,
            list_categorical_vars: list, list_numerical_vars: list,
            verbose=False):

        self.df_features = df_features
        v_col_names = df_features.columns.values

        tuple_scores_df_filtered_features = mrmr_classif(X=df_features, y=y_label, K=k,
                                                         cat_features=list_categorical_vars,
                                                         cat_encoding='target',
                                                         return_scores=self.return_scores,
                                                         show_progress=verbose)

        v_scores_features = np.array(tuple_scores_df_filtered_features[1])
        v_scores_features_sorted = v_scores_features[v_scores_features.argsort()[::-1]]
        v_selected_feature_names_sorted = v_col_names[v_scores_features.argsort()][::-1]
        list_selected_features_index = [df_features.columns.get_loc(var_name) for var_name in v_selected_feature_names_sorted]

        self.selected_feature_indices = list_selected_features_index[:k]
        self.selected_feature_names = list(v_selected_feature_names_sorted[:k])

        m_scores = np.c_[v_selected_feature_names_sorted, v_scores_features_sorted]
        df_feature_scores = pd.DataFrame(m_scores, columns=['var_name', 'score'])

        self.df_feature_scores = df_feature_scores


class PermutationImportance(FSMethod):

    def __init__(self, return_scores, seed_value, estimator, param_grid_estimator):
        super().__init__(return_scores, seed_value)
        self.estimator = estimator
        self.param_grid_estimator = param_grid_estimator

    def fit(self, df_features: pd.DataFrame, y_label: np.unique, k: int,
            list_categorical_vars: list, list_numerical_vars: list,
            verbose=False):

        self.df_features = df_features
        x_features = df_features.values
        v_col_names = df_features.columns.values

        grid_cv = GridSearchCV(estimator=self.estimator,
                               param_grid=self.param_grid_estimator,
                               cv=3,
                               return_train_score=True,
                               scoring=make_scorer(compute_mrae, greater_is_better=False)
                               )

        grid_cv.fit(x_features, y_label)
        best_estimator = grid_cv.best_estimator_

        result = permutation_importance(best_estimator, x_features, y_label,
                                        n_repeats=50, random_state=self.seed_value)

        v_feature_importance = result.importances_mean
        v_feature_importance_sorted = v_feature_importance[v_feature_importance.argsort()][::-1]
        v_col_names_sorted = v_col_names[v_feature_importance.argsort()[::-1]]
        m_scores = np.c_[v_col_names_sorted, v_feature_importance_sorted]
        df_feature_scores = pd.DataFrame(m_scores, columns=['var_name', 'score'])

        list_selected_features_index = [df_features.columns.get_loc(var_name) for var_name in v_col_names_sorted]

        self.selected_feature_indices = list_selected_features_index[:k]
        self.selected_feature_names = list(v_col_names_sorted)
        self.df_feature_scores = df_feature_scores

