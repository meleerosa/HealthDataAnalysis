from modules.analysis import Analysis
from modules.dataload import Dataload
from modules.preprocess import Preprocess
from modules.regression_model import RegressionModel

if __name__ == '__main__':

    # raw data EDA
    raw_eda = Dataload('raw')
    raw_data = raw_eda.load_data()
    # raw_eda.data_info()
    # raw_eda.data_describe()
    # raw_eda.plot_distribution()

    # 데이터 전처리하기
    exp99 = Preprocess(raw_data)
    selected_columns = [
            '연령대 코드(5세단위)', '신장(5Cm단위)',
            '허리둘레', '시력(우)', '식전혈당(공복혈당)', 'HDL 콜레스테롤', 'LDL 콜레스테롤',
            '혈색소', '(혈청지오티)AST', '(혈청지오티)ALT', '감마 지티피', '흡연상태',
            '음주여부', '치석'
            ]


    exp99.select_columns(name = 'exp99',columns = selected_columns)
    exp99.dropna()
    exp99.drop_anomalies() # 특이값 처리 ex) 시력 9.9, 허리둘레 1000cm
    train, test = exp99.split_dataset()
    # train = exp99.drop_outliers(train, 1.5)
    X_train, y_train, X_test, y_test = exp99.make_Xy(train, test, y= '(혈청지오티)ALT')
    feature_scaler, target_scaler ,X_train, y_train, X_test, y_test = exp99.scale_dataset(X_train, y_train, X_test, y_test, scale_type= 'robust')


    # 전처리한 데이터 파악하기
    exp99_eda = Dataload('exp99')
    exp99_eda.load_data(train)
    exp99_eda.data_info()
    exp99_eda.data_describe()
    exp99_eda.plot_distribution()
    exp99_eda.show_vif()

    # 전처리한 데이터 분석하기
    exp99_analysis = Analysis('exp99',train)
    exp99_analysis.corr_heatmap()
    exp99_analysis.corr_scatter()

    # 모델
    exp99_model = RegressionModel('exp99', X_train, y_train, X_test, y_test)
    ols = exp99_model.build_OLS_model()
    xgb = exp99_model.build_XGBregressor_model()
    exp99_model.evaluation_matrix(ols, target_scaler)
    exp99_model.evaluation_matrix(xgb, target_scaler)
    exp99_model.show_prediction(ols, target_scaler, 100)
    exp99_model.show_prediction(xgb, target_scaler, 100)
    exp99_model.show_prediction(ols, target_scaler, 200)
    exp99_model.show_prediction(xgb, target_scaler, 200)
    exp99_model.show_prediction(ols, target_scaler, 500)
    exp99_model.show_prediction(xgb, target_scaler, 500)