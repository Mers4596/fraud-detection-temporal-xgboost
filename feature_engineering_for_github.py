import pandas as pd
import numpy as np


# --------------------------------------------------
# TIME FEATURES
# --------------------------------------------------

def create_time_features(df):
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"],
        format="%d-%m-%Y %H:%M",
        dayfirst=True
    )

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["trans_date_trans_time"].dt.weekday >= 5).astype(int)
    df["is_night"] = (df["hour"] < 6).astype(int)

    return df


# --------------------------------------------------
# AMOUNT FEATURES
# --------------------------------------------------

def create_amount_features(df):
    df["amt_log"] = np.log1p(df["amt"])

    threshold = df["amt"].quantile(0.95)
    df["high_amount_flag"] = (df["amt"] > threshold).astype(int)

    return df


# --------------------------------------------------
# VELOCITY FEATURES
# --------------------------------------------------

def create_velocity_features(df):

    df = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    result = []

    for cc, group in df.groupby("cc_num", sort=False):
        group = group.sort_values("trans_date_trans_time").copy()
        group.set_index("trans_date_trans_time", inplace=True)

        group["tx_count_1h"] = group["amt"].rolling("1h").count()
        group["tx_count_24h"] = group["amt"].rolling("24h").count()

        group["tx_sum_1h"] = group["amt"].rolling("1h").sum()
        group["tx_sum_24h"] = group["amt"].rolling("24h").sum()
        group["tx_mean_24h"] = group["amt"].rolling("24h").mean()

        group.reset_index(inplace=True)
        result.append(group)

    df = pd.concat(result, ignore_index=True)

    return df


# --------------------------------------------------
# MASTER FUNCTION
# --------------------------------------------------

def apply_feature_engineering(df):
    df = create_time_features(df)
    df = create_amount_features(df)
    df = create_velocity_features(df)
    return df
