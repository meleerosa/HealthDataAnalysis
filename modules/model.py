import logging
import os
from datetime import datetime
from math import sqrt
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from modules.config import Config
from modules.utils.decorator import TryDecorator
from modules.utils.file_handler import chk_and_make_dir

model_logger = logging.getLogger("model")

class Model:
    """
    Model class XGBoost 모델을 만들고 훈련, 테스트 평가를 하는 클래스

    Attributes:
        _input_data (pandas.DataFrame): 전처리가 완료된 입력 데이터
        _train_input (numpy.ndarray): 훈련을 위한 스케일링 처리된 입력 데이터
        _train_target (numpy.ndarray): 훈련을 위한 스케일링 처리된 입력(target) 데이터
        _test_input (numpy.ndarray): 테스트를 위한 스케일링 처리된 입력 데이터
        _test_target (numpy.ndarray): 테스트를 위한 스케일링 처리된 입력(target) 데이터
        _input_scaler (sklearn.preprocessing.RobustScaler): 입력 데이터를 스케일링하기 위한 scaler 객체
        _target_scaler (sklearn.preprocessing.RobustScaler): 타겟 데이터를 스케일링하기 위한 scaler 객체
        _model (xgboost.XGBRegressor): XGBoost 모델 객체
        _config (dict): config 파라미터
        _model_path (str): 모델을 세이브 할 디렉토리 경로

    """
    _input_data: pd.DataFrame
    _train_input: np.ndarray
    _train_target: np.ndarray
    _test_input: np.ndarray
    _test_target: np.ndarray
    _input_scaler: RobustScaler
    _target_scaler: RobustScaler 
    _model: xgb.XGBRegressor
    _config: dict
    _model_path: str
    
    @property
    def h_param(self):
        """
        하이퍼파라미터를 출력하는 getter 메소드

        Returns:
            dict: 하이퍼파라미터 값
        """
        return self._h_param

    def __init__(self, preprocessed_data, h_param: Dict = None) -> None:
        """모델 클래스의 생성자

        Args:
            preprocessed_data (pandas.DataFrame): 전처리된 입력 데이터
            h_param (Dict): 하이퍼파라미터 값 (default: None).

        """
        self._config = Config.instance().config
        self._model_logger = logging.getLogger("Model")
        self._preprocessed_data = preprocessed_data
        if h_param is None:
            self._h_param = self._config["hyper_parameter"]
        else:
            self._h_param = h_param


    @TryDecorator(logger= model_logger)
    def _split_data(self) -> None:
        """
        전처리된 데이터를 train 셋과 test 셋으로 분할하고 input data를 scaling

        Raises:
            Exception: 전처리 데이터가 빈 경우

        """
        if self._preprocessed_data is None or self._preprocessed_data.empty:
            raise Exception("preprocessed data is Empty")
        
        train_set, test_set = train_test_split(
            self._preprocessed_data, test_size= 0.2, random_state= 42, shuffle= True
        )

        self._train_input = train_set.drop(columns = ['(혈청지오티)ALT'])
        self._train_target = train_set["(혈청지오티)ALT"]
        self._test_input = test_set.drop(columns = ['(혈청지오티)ALT'])
        self._test_target = test_set["(혈청지오티)ALT"]

        #scaling
        self._input_scaler = RobustScaler()
        self._target_scaler = RobustScaler()
        for col in self._train_input.columns:
            self._train_input[col] = self._input_scaler.fit_transform(self._train_input[col].values.reshape(-1,1))
            self._test_input[col] = self._input_scaler.transform(self._test_input[col].values.reshape(-1,1))

        self._train_target = self._target_scaler.fit_transform(self._train_target.values.reshape(-1,1))
        self._test_target = self._target_scaler.transform(self._test_target.values.reshape(-1,1))

        self._train_input = self._train_input.values
        self._test_input = self._test_input.values

        model_logger.info("Model Data Count-------------------------------------")
        model_logger.info(
            "preprocessed dataset  : " + str(len(self._preprocessed_data))
        )
        model_logger.info("train dataset      : " + str(len(self._train_input)))
        model_logger.info("test dataset       : " + str(len(self._test_input)))
        model_logger.info("-----------------------------------------------------")

    def _build_model(self) -> None:
        self._model = xgb.XGBRegressor(
        n_estimators = 200,
        reg_alpha = 0,
        reg_lambda = 1,
        booster = 'gbtree',
        learning_rate = 0.03,
        gamma = 0.1,
        subsample = 0.4,
        colsample_bytree = 1,
        max_depth = 7
        )

    def _fit(self) -> None:
        self._model.fit(self._train_input, self._train_target)

    def evaluate_model(self) -> dict:
        actual = self._target_scaler.inverse_transform(self._test_target.reshape(-1,1))
        predict = self._target_scaler.inverse_transform(self._model.predict(self._test_input).reshape(-1,1))
        actual_train = self._target_scaler.inverse_transform(self._train_target.reshape(-1,1))
        predict_train = self._target_scaler.inverse_transform(self._model.predict(self._train_input).reshape(-1,1))

        eval_metric = {
            'Train MSE': mean_squared_error(actual_train, predict_train),
            'Train RMSE': sqrt(mean_squared_error(actual_train, predict_train)),
            'Train MAE': mean_absolute_error(actual_train, predict_train),
            'MSE': mean_squared_error(actual, predict),
            'RMSE': sqrt(mean_squared_error(actual, predict)),
            'MAE': mean_absolute_error(actual, predict)
        }

        return eval_metric


    def predict(self, input_data: Any) -> Any:
        pred = []
        for each_input in input_data:
            pred.append(self._model.predict(each_input))
        return pred

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            model_logger.error("Model does not exsist.")
        else:
            self._model = xgb.load_model(model_path)

    def _save_model(self, model_path: str) -> None:
        chk_and_make_dir(model_path)
        self._model.save(model_path)

    def save_model(self, model_path: str = None) -> None:
        current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._model_path = model_path
        if self._model_path is None:
            self._model_path = os.path.join(
                self._config["path"]["output"], current_dt, "model"
            )
        self._save_model(self._model_path)

    def load_model(self, model_dt: str) -> None:
        model_path = os.path.join(self._config["path"]["output"], model_dt, "model")
        self._load_model(model_path)

    def fit_and_evaluate(self) -> Dict:
        self._split_data()
        self._build_model()
        self._fit()
        return self.evaluate_model()

    def fit_and_predict_test(self) -> Any:
        self._split_data()
        self._build_model()
        self._fit()
        self.save_model()
        self.evaluate_model()
        return self.predict(self._test_input)

