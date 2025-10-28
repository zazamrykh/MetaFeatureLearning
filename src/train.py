import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, f1_score

from scipy import sparse

from src.data.dataset import Dataset
from src.utils.func import one_hot_encode_df, standardize_inplace

class Trainer(ABC):
    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def predict():
        pass

    @abstractmethod
    def eval():
        pass


class Trainer():
    def __init__(
            self,
            is_classification=True,
            model=None):
        assert model is not None, 'You must specify model'
        self.is_classification = is_classification
        self.enc = None
        self.model = model

    def _get_Xy(self, dataset: Dataset, normalize=True):
        df = dataset.df.copy()
        y = df['target']
        X_df = df.drop(columns=['target'])

        # one-hot признаков
        if self.enc is None:
            X_enc, enc = one_hot_encode_df(X_df)  # должен вернуть ndarray (используй sparse_output=False)
            self.enc = enc
        else:
            X_enc = self.enc.transform(X_df)

        # В X получаем ndarray
        if isinstance(X_enc, pd.DataFrame):
            X = X_enc.to_numpy()
        else:
            X = X_enc.toarray() if sparse.issparse(X_enc) else np.asarray(X_enc)

        if self.is_classification:
            # Классификация: работаем с метками, не нормализуем y
            # Удаляем пропуски в y
            y_mask = ~y.isna()
            if not y_mask.all():
                X = X[y_mask.to_numpy()]
                y = y[y_mask]

            # Если тип y не числовой — кодируем в категории (складываем в numpy массив)
            if not np.issubdtype(y.dtype, np.number):
                y = pd.Categorical(y)
                y = y.codes.astype(int)  # -1 для NaN уже удалены, оставшиеся >=0
            else:
                y = y.to_numpy()

            # Нормализация X опциональна
            if normalize:
                X = standardize_inplace(X)
        else:
            # Регрессия: приводим y к float и чистим нечисловые/Inf
            y_num = pd.to_numeric(y, errors='coerce').to_numpy(dtype=float)
            mask = np.isfinite(y_num)
            if not mask.all():
                X = X[mask]
                y_num = y_num[mask]

            if normalize:
                X = standardize_inplace(X)
                y_num = standardize_inplace(y_num)

            y = y_num

        return X, y


    def fit(self, dataset : Dataset):
        X, y = self._get_Xy(dataset)
        self.model.fit(X, y)


    def predict(self, sample : pd.Series | np.ndarray):
        return self.model.predict(sample)


    def eval(self, dataset : Dataset) -> float:
        X, y_target = self._get_Xy(dataset)
        y_pred = self.predict(X)

        class_number = int(dataset.meta.get('qualities', {}).get('NumberOfClasses', 0))

        if class_number == 0:
            # regression
            result = mean_squared_error(y_target, y_pred)
        else:
            # classification
            result = f1_score(y_target, y_pred, average='macro')

        return result
