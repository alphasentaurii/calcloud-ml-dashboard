from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from . import io


""" ----- PREPROCESSING ----- """

# No longer using this because of lambda ingest into DDB
def combine_s3_datasets(keys, dropnans=1):
    """Pass in a list of dataframes or local csv filepaths for features, targets and predictions (must be in that order).
    args:
    `keys` (list): [features_df, targets_df, preds_df] or ['features.csv', 'targets.csv', 'preds.csv']
    `dropnans` (default=1)
    0: leave data as is (do not remove NaNs)
    1: drop NaNs from features+targets
    2: drop NaNs from features+targets+preds
    """
    # load files from csv
    F = pd.read_csv(keys[0], index_col="ipst")
    T = pd.read_csv(keys[1], index_col="ipst")
    P = pd.read_csv(keys[2], index_col="ipst")
    # print data summaries
    print("Features: ", len(F))
    print("Targets: ", len(T))
    print("Preds: ", len(P))
    # combine into single df
    data = F.join(T, how="left")
    if dropnans == 1:
        df0 = data.dropna(axis=0, inplace=False)
        print(f"NaNs Removed: {len(data) - len(df0)}")
        df1 = df0.join(P, how="left")
    elif dropnans == 2:
        df0 = data.join(P, how="left")
        df1 = df0.dropna(axis=0, inplace=False)
        print(f"NaNs Removed: {len(df0) - len(data)}")
    else:
        df1 = data.join(P, how="left")
    print(df1.isna().sum())
    # drop duplicates
    df1["ipst"] = df1.index
    df1.set_index("ipst", inplace=True, drop=False)
    df = df1.drop_duplicates(subset="ipst", keep="last", inplace=False)
    print("Final: ", len(df))
    io.save_dataframe(df, "batch.csv")
    return df


def combine_from_s3(keys, bucket_mod, prefix):
    master_data = None
    io.s3_download(keys, bucket_mod, prefix) # s3://bucket_mod/prefix/keys
    if "features.csv" in keys: # join along columns (features + targets)
        df = combine_s3_datasets(keys, dropnans=1)
        io.s3_upload(["batch.csv"], bucket_mod, prefix)
        try:
            io.s3_download(["master.csv"], bucket_mod, "latest")
            master_data = pd.read_csv("master.csv", index_col="ipst")
            df_list = [df, master_data]
        except Exception as e:
            print("Master dataset not found in s3.")
            print(e)
            df_list = [df]
    else:
        df_list = []
        for dataset in keys:
            data = pd.read_csv(dataset, index_col="ipst")
            df_list.append(data)
        if len(df_list) == 1:
            df_list.append(master_data)
    return df_list


# DynamoDB removes need for this (keeping bc it's useful for combining DFs)
def combine_training_sets(df_list):
    """Takes a list of dataframes and combines them into one
    Removes duplicates and keeps most recent (assumes list ordered newest to oldest)
    """
    if len(df_list) == 1:
        print("Single batch only (skipping join)")
        return df_list[0]
    else:
        n_combined = 0
        for df in df_list:
            n_combined += len(df)
            print("+ ", len(df))
        print("Combined: ", n_combined)
        df_tmp = pd.concat([d for d in df_list], axis=0, verify_integrity=False)
        df_tmp["ipst"] = df_tmp.index
        df_tmp.set_index("ipst", inplace=True, drop=False)
        df = df_tmp.drop_duplicates(subset="ipst", keep="last", inplace=False)
        print(f"Removed {n_combined - len(df)} duplicates")
        print("Final DF: ", len(df))
    return df


def update_power_transform(df):
    pt = PowerTransformer(standardize=False)
    df_cont = df[["n_files", "total_mb"]]
    pt.fit(df_cont)
    input_matrix = pt.transform(df_cont)
    # FILES (n_files)
    f_mean = np.mean(input_matrix[:, 0])
    f_sigma = np.std(input_matrix[:, 0])
    # SIZE (total_mb)
    s_mean = np.mean(input_matrix[:, 1])
    s_sigma = np.std(input_matrix[:, 1])
    files = input_matrix[:, 0]
    size = input_matrix[:, 1]
    x_files = (files - f_mean) / f_sigma
    x_size = (size - s_mean) / s_sigma
    normalized = np.stack([x_files, x_size], axis=1)
    idx = df_cont.index
    df_norm = pd.DataFrame(normalized, index=idx, columns=["x_files", "x_size"])
    df["x_files"] = df_norm["x_files"]
    df["x_size"] = df_norm["x_size"]
    pt_transform = {"lambdas": pt.lambdas_, "f_mean": f_mean, "f_sigma": f_sigma, "s_mean": s_mean, "s_sigma": s_sigma}
    print(pt_transform)
    return df, pt_transform


def preprocess(bucket_mod, prefix, src, table_name, p_key, attr):
    # MAKE TRAINING SET - single df for ingested data
    if src == "ddb":  # dynamodb 'calcloud-hst-data'
        ddb_data = io.ddb_download(table_name, p_key, attr)
        io.write_to_csv(ddb_data, "batch.csv")
        df = pd.read_csv("batch.csv", index_col="ipst")
        df_list = [df]
    elif src.startswith("s3"): # "s3:multi", "s3:[list of files]"
        s3keys = src.split(':')[-1].lstrip("'[").rstrip("]'").split(', ')
        df_list = combine_from_s3(s3keys, bucket_mod, prefix)
    # combine previous with new data
    df = combine_training_sets(df_list)
    # update power transform
    df, pt_transform = update_power_transform(df)
    io.save_dataframe(df, "latest.csv")
    data_dict = {"pt_transform": pt_transform}
    if src == "s3":  # save pt metadata and updated dataframe
        keys = io.save_to_pickle(data_dict, df_key="latest.csv")
    else:  # (DDB) save just pt metadata
        keys = io.save_to_pickle(data_dict)
    io.s3_upload(keys, bucket_mod, prefix)
    return df


def encode_target_data(y_train, y_test):
    # reshape target data
    # y_train = y_train.reshape(-1, 1)
    # y_test = y_test.reshape(-1, 1)
    # label encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_enc = encoder.transform(y_train)
    y_train = tf.keras.utils.to_categorical(y_train_enc)
    # test set
    encoder.fit(y_test)
    y_test_enc = encoder.transform(y_test)
    y_test = tf.keras.utils.to_categorical(y_test_enc)
    # ensure train/test targets have correct shape (4 bins)
    print(y_train.shape, y_test.shape)
    return y_train, y_test


def make_tensors(X_train, y_train, X_test, y_test):
    """Convert Arrays to Tensors"""
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    return X_train, y_train, X_test, y_test


def split_Xy(df, target_col):
    targets = df[target_col]
    cols = [
        "n_files",
        "total_mb",
        "wallclock",
        "memory",
        "mem_bin",
        "mem_pred",
        "bin_pred",
        "wall_pred",
        "ipst",
        "timestamp",
    ]
    drop_cols = [col for col in cols if col in df.columns]
    features = df.drop(columns=drop_cols, axis=1, inplace=False)
    X = features.values
    y = targets.values
    return X, y


def prep_data(df, target_col, tensors=True):
    # split
    X, y = split_Xy(df, target_col)
    # encode if classifier
    if target_col == "mem_bin":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        y_train, y_test = encode_target_data(y_train, y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if tensors is True:
        # convert arrays into tensors (better performance for tensorflow)
        X_train, y_train, X_test, y_test = make_tensors(X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test

