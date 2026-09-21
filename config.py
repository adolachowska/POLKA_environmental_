XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 1,
    'objective': 'mlogloss'
}

#docker run --rm -v "${PWD}:/app" polka-ml-env python main.py#
#docker build -t polka-ml-env .#