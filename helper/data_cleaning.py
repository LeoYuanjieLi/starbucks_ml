import numpy as np
import pandas as pd


def clean_profile(profile_path):
    profile = pd.read_json(profile_path, orient="records", lines=True)

    # drop & clean up invalid data
    profile = profile.dropna()
    profile = profile[profile["gender"] != 0]
    profile.rename(columns={"id": "consumer_id"}, inplace=True)

    # split date
    profile["became_member_on"] = profile["became_member_on"].apply(lambda x: str(x))
    profile["year"] = profile["became_member_on"].str[:4]
    profile["month"] = profile["became_member_on"].str[4:6]
    profile["day"] = profile["became_member_on"].str[6:]

    profile.drop(["became_member_on"], axis=1, inplace=True)

    # one-hot-encode date
    years = one_hot_encode(profile["year"])
    months = one_hot_encode(profile["month"])

    profile.drop(["year", "month", "day"], axis=1, inplace=True)

    result = pd.concat([profile, years, months], axis=1)

    result.set_index("consumer_id", inplace=True)
    return result


def one_hot_encode(series):
    unique = sorted(series.unique())

    result = {}

    for val in unique:
        result[val] = np.zeros(len(series))

    result = pd.DataFrame(result, index=series.index)

    for row in result.index:
        for col in result.columns:
            if series[row] == col:
                result.at[row, col] = 1

    return result
