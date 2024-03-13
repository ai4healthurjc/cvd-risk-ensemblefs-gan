from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw')
PATH_PROJECT_DATA_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed')
PATH_PROJECT_PARTITIONS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'partitions')

PATH_PROJECT_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'models')
PATH_PROJECT_REPORTS = Path.joinpath(PATH_PROJECT_DIR, 'reports')
PATH_PROJECT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'metrics')
PATH_PROJECT_RESULTS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'results')
PATH_PROJECT_OVERSAMPLING = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'oversampling')
PATH_PROJECT_OVERSAMPLING_DOWNLOAD = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'oversampling', 'download')
PATH_PROJECT_OVERSAMPLING_UPLOAD = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'oversampling', 'upload')
PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')
PATH_PROJECT_COEFS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'coefs')
PATH_PROJECT_INTERPRETER = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'interpreter')
PATH_PROJECT_SYNTHETIC_DATA = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'synthetic_data')
PATH_PROJECT_FS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'fs')
PATH_PROJECT_MODEL_SHAP = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'model_shap', 'knn')
PATH_PROJECT_PARAMS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'params')
PATH_PROJECT_DATA_PARTITIONS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'partitions')

NAME_DATASET_FRAM = 'bbdd_fram.csv'
NAME_DATASET_DIABETES = 'diabetes_data.csv'
TYPE_FEATURE_CONTINUOUS = 'c'
TYPE_FEATURE_DISCRETE = 'd'
BBDD_FRAM = 'fram'
BBDD_STENO = 'steno'

URL_STENO_CALCULATOR = 'https://steno.shinyapps.io/T1RiskEngine/'

TOT = 1e-4
MAX_ITERS = 20000
TEST_SIZE_PROPORTION = 0.2
VALIDATION_SIZE_PROPORTION = 0.2

LISTS_BBDD_CLINICAL = ['fram', 'steno']
LISTS_BBDD_GENERAL = ['dermat', 'heart']
