
import pdb
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_train_data(
    dataset_name, scaler=None, train_shuffle=True, no_transform=False, verbose=False
):

    if verbose:
        print("Loading {} train data...".format(dataset_name))

    if scaler is None:
        if verbose:
            print(
                "No scaler provided. Using default sklearn.preprocessing.StandardScaler"
            )
        scaler = StandardScaler()

    if dataset_name == "BATADAL":
        try:
            df_train = pd.read_csv(
                "data/" + dataset_name + "/train_dataset.csv",
                parse_dates=["DATETIME"],
                dayfirst=True,
            )
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find BATADAL train dataset. Did you unpack BATADAL.tar.gz?"
            )
        sensor_cols = [
            col
            for col in df_train.columns
            if col not in ["Unnamed: 0", "DATETIME", "ATT_FLAG"]
        ]
        # drop colunm
        drop_cols = ["PRESSURE_J415", "PRESSURE_J317", "STATUS_PU6", "FLOW_PU6"]
        df_train = df_train.drop(columns=drop_cols, errors="ignore")  # pandas
        sensor_cols = [c for c in sensor_cols if c not in drop_cols]

    elif dataset_name == "SWAT":
        try:
            df_train = pd.read_csv(
                "data/" + dataset_name + "/SWATv0_train.csv", dayfirst=True
            )
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find SWAT train dataset. Did you request the dataset and process it?"
            )
        sensor_cols = [
            col for col in df_train.columns if col not in ["Timestamp", "Normal/Attack"]
        ]

    elif dataset_name == "SWAT-CLEAN":
        try:
            df_train = pd.read_csv("data/SWAT/SWATv0_train.csv")
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find SWAT train dataset. Did you request the dataset and process it?"
            )
        remove_list = [
            "Timestamp",
            "Normal/Attack",
            "P202",
            "P401",
            "P404",
            "P502",
            "P601",
            "P603",
        ]
        ks_columns = [
            "AIT201",
            "AIT202",
            "AIT203",
            "P201",
            "P205",
            "AIT401",
            "AIT402",
            "AIT501",
            "AIT502",
            "AIT503",
            "PIT502",
        ]
        sensor_cols = [
            col for col in df_train.columns if col not in (remove_list + ks_columns)
        ]
        target_col = "Normal/Attack"

    elif dataset_name == "WADI":
        try:
            df_train = pd.read_csv("data/" + dataset_name + "/WADI_train.csv")
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find WADI train dataset. Did you request the dataset and process it?"
            )
        remove_list = [
            "Row",
            "Date",
            "Time",
            "Attack",
            "2B_AIT_002_PV",
            "2_LS_001_AL",
            "2_LS_002_AL",
            "2_P_001_STATUS",
            "2_P_002_STATUS",
        ]
        sensor_cols = [col for col in df_train.columns if col not in remove_list]

    elif dataset_name == "WADI-CLEAN":
        try:
            df_train = pd.read_csv("data/WADI/WADI_train.csv")
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find WADI train dataset. Did you request the dataset and process it?"
            )
        remove_list = [
            "Row",
            "Date",
            "Time",
            "Attack",
            "2B_AIT_002_PV",
            "2_LS_001_AL",
            "2_LS_002_AL",
            "2_P_001_STATUS",
            "2_P_002_STATUS",
        ]
        ks_columns = [
            "1_AIT_001_PV",
            "1_AIT_004_PV",
            "1_AIT_005_PV",
            "2_FIC_101_SP",
            "2_LT_001_PV",
            "2A_AIT_001_PV",
            "2A_AIT_003_PV",
            "2A_AIT_004_PV",
            "2B_AIT_001_PV",
            "2B_AIT_004_PV",
            "3_AIT_002_PV",
            "3_AIT_005_PV",
        ]
        sensor_cols = [
            col for col in df_train.columns if col not in (remove_list + ks_columns)
        ]

    else:
        raise SystemExit(f"Cannot find dataset name: {dataset_name}.")

    # scale sensor data
    if no_transform:
        X = pd.DataFrame(
            index=df_train.index, columns=sensor_cols, data=df_train[sensor_cols].values
        )
    else:
        X_prescaled = df_train[sensor_cols].values
        X = pd.DataFrame(
            index=df_train.index,
            columns=sensor_cols,
            data=scaler.fit_transform(X_prescaled),
        )

        # Need fitted scaler for future attack/test data
        pickle.dump(scaler, open(f"models/{dataset_name}_scaler.pkl", "wb"))
        if verbose:
            print("Saved scaler parameters to {}.".format("scaler.pkl"))

    return X.values, sensor_cols


def load_test_data(dataset_name, scaler=None, no_transform=False, verbose=False):

    if verbose:
        print("Loading {} test data...".format(dataset_name))

    if scaler is None:
        if verbose:
            print("No scaler provided, trying to load from models directory...")
        scaler = pickle.load(open(f"models/{dataset_name}_scaler.pkl", "rb"))
        print("Successful.")

    if dataset_name == "BATADAL":
        try:
            df_test = pd.read_csv(
                "data/" + dataset_name + "/test_dataset_1.csv",
                parse_dates=["DATETIME"],
                dayfirst=True,
            )
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find BATADAL test dataset. Did you unpack BATADAL.tar.gz?"
            )
        sensor_cols = [
            col
            for col in df_test.columns
            if col not in ["Unnamed: 0", "DATETIME", "ATT_FLAG"]
        ]
        target_col = "ATT_FLAG"
        # drop colunm
        drop_cols = ["PRESSURE_J415", "PRESSURE_J317", "STATUS_PU6", "FLOW_PU6"]
        df_test = df_test.drop(columns=drop_cols, errors="ignore")  # pandas
        sensor_cols = [c for c in sensor_cols if c not in drop_cols]
    elif dataset_name == "SWAT":
        try:
            df_test = pd.read_csv("data/" + dataset_name + "/SWATv0_test.csv")
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find SWAT test dataset. Did you request the dataset and process it?"
            )
        sensor_cols = [
            col for col in df_test.columns if col not in ["Timestamp", "Normal/Attack"]
        ]
        target_col = "Normal/Attack"

    elif dataset_name == "SWAT-CLEAN":
        try:
            df_test = pd.read_csv("data/SWAT/SWATv0_test.csv")
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find SWAT test dataset. Did you request the dataset and process it?"
            )
        remove_list = [
            "Timestamp",
            "Normal/Attack",
            "P202",
            "P401",
            "P404",
            "P502",
            "P601",
            "P603",
        ]
        ks_columns = [
            "AIT201",
            "AIT202",
            "AIT203",
            "P201",
            "P205",
            "AIT401",
            "AIT402",
            "AIT501",
            "AIT502",
            "AIT503",
            "PIT502",
        ]
        sensor_cols = [
            col for col in df_test.columns if col not in (remove_list + ks_columns)
        ]
        target_col = "Normal/Attack"

    elif dataset_name == "WADI":
        try:
            df_test = pd.read_csv("data/" + dataset_name + "/WADI_test.csv")
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find WADI test dataset. Did you request the dataset and process it?"
            )
        remove_list = [
            "Row",
            "Date",
            "Time",
            "Attack",
            "2B_AIT_002_PV",
            "2_LS_001_AL",
            "2_LS_002_AL",
            "2_P_001_STATUS",
            "2_P_002_STATUS",
        ]
        sensor_cols = [col for col in df_test.columns if col not in remove_list]
        target_col = "Attack"

    elif dataset_name == "WADI-CLEAN":
        try:
            df_test = pd.read_csv("data/WADI/WADI_test.csv")
        except FileNotFoundError:
            raise SystemExit(
                "Unable to find WADI test dataset. Did you request the dataset and process it?"
            )
        remove_list = [
            "Row",
            "Date",
            "Time",
            "Attack",
            "2B_AIT_002_PV",
            "2_LS_001_AL",
            "2_LS_002_AL",
            "2_P_001_STATUS",
            "2_P_002_STATUS",
        ]
        ks_columns = [
            "1_AIT_001_PV",
            "1_AIT_004_PV",
            "1_AIT_005_PV",
            "2_FIC_101_SP",
            "2_LT_001_PV",
            "2A_AIT_001_PV",
            "2A_AIT_003_PV",
            "2A_AIT_004_PV",
            "2B_AIT_001_PV",
            "2B_AIT_004_PV",
            "3_AIT_002_PV",
            "3_AIT_005_PV",
        ]
        sensor_cols = [
            col for col in df_test.columns if col not in (remove_list + ks_columns)
        ]
        target_col = "Attack"

    else:
        raise SystemExit(f"Cannot find dataset name: {dataset_name}.")

    # scale sensor data
    if no_transform:
        Xtest = pd.DataFrame(
            index=df_test.index, columns=sensor_cols, data=df_test[sensor_cols]
        )
    else:
        Xtest = pd.DataFrame(
            index=df_test.index,
            columns=sensor_cols,
            data=scaler.transform(df_test[sensor_cols]),
        )

    Ytest = df_test[target_col]

    return Xtest.values, Ytest.values, Xtest.columns