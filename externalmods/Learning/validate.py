import time
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from . import prep, io
from . import train

def kfold_cross_val(df, target_col, bucket_mod, data_path, verbose):
    # evaluate using 10-fold cross validation
    print("\nStarting KFOLD Cross-Validation...")
    start = time.time()
    X, y = prep.split_Xy(df, target_col)
    # run estimator
    if target_col == "mem_bin":
        # Y = y.reshape(-1, 1)
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        # y_enc = keras.utils.to_categorical(y)
        estimator = KerasClassifier(build_fn=train.memory_classifier, epochs=30, batch_size=32, verbose=verbose)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
    elif target_col == "memory":
        estimator = KerasRegressor(build_fn=train.memory_regressor, epochs=150, batch_size=32, verbose=verbose)
        kfold = KFold(n_splits=10, shuffle=True)
    elif target_col == "wallclock":
        estimator = KerasRegressor(build_fn=train.wallclock_regressor, epochs=300, batch_size=64, verbose=verbose)
        kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, y, cv=kfold, n_jobs=-1)
    end = time.time()
    duration = io.proc_time(start, end)
    if target_col == "mem_bin":
        score = np.mean(results)
    else:
        score = np.sqrt(np.abs(np.mean(results)))
    print(f"\nKFOLD scores: {results}\n")
    print(f"\nMean Score: {score}\n")
    print("\nProcess took ", duration)
    kfold_dict = {f"kfold_{target_col}": {"results": list(results), "score": score, "time": duration}}
    keys = io.save_to_pickle(kfold_dict, target_col=target_col)
    io.s3_upload(keys, bucket_mod, f"{data_path}/results")


def run_kfold(df, bucket_mod, data_path, models, verbose):
    for target in models:
        kfold_cross_val(df, target, bucket_mod, data_path, verbose)