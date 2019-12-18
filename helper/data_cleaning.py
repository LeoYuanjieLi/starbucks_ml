import numpy as np
import pandas as pd


def clean_profile(profile_path):
    profile = pd.read_json(profile_path, orient="records", lines=True)

    # drop & clean up invalid data
    profile = profile.dropna()
    profile = profile[profile["gender"] != "O"]
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

    # one-hot-encode gender
    gender = one_hot_encode(profile["gender"])
    profile.drop(["gender"], axis=1, inplace=True)

    # one-hot-encode age
    age = one_hot_encode_age(profile)
    profile.drop(["age"], axis=1, inplace=True)

    # one-hot-encode income
    income = one_hot_encode_income(profile)
    profile.drop(["income"], axis=1, inplace=True)

    result = pd.concat([profile, years, months, gender, age, income], axis=1)

    result.set_index("consumer_id", inplace=True)
    return result


def one_hot_encode_age(profile):
    age = profile[profile["age"] < 118]["age"]

    index = age.index
    n = len(age)
    df = pd.DataFrame(
        {
            "0-9": np.zeros(n),
            "10-19": np.zeros(n),
            "20-29": np.zeros(n),
            "30-39": np.zeros(n),
            "40-49": np.zeros(n),
            "50-59": np.zeros(n),
            "60-69": np.zeros(n),
            "70-79": np.zeros(n),
            "80-89": np.zeros(n),
            "90-99": np.zeros(n),
        },
        index=index,
    )

    age_map = {
        0: "0-9",
        1: "10-19",
        2: "20-29",
        3: "30-39",
        4: "40-49",
        5: "50-59",
        6: "60-69",
        7: "70-79",
        8: "80-89",
        9: "90-99",
    }
    for i in index:
        key = age_map[age.loc[i] % 10]
        df.at[i, key] = 1

    return df


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


def one_hot_encode_channels(portfolio):

    n = len(portfolio)

    channels = pd.DataFrame(
        {
            "email": np.zeros(n),
            "mobile": np.zeros(n),
            "web": np.zeros(n),
            "social": np.zeros(n),
        },
        index=portfolio.index,
    )

    for i in portfolio.index:
        cur_channel = portfolio["channels"][i]

        for c in cur_channel:
            channels.at[i, c] = 1

    return channels


def one_hot_encode_income(profile):
    income = profile.dropna()["income"]
    n = len(income)

    min_income, max_income = int(round(min(income), -4)), int(round(max(income), -4))

    data = {}
    step = 10000
    for i in range(min_income, max_income + step, step):
        data[f"{i}-{i+step}"] = np.zeros(n)
    index = income.index

    df = pd.DataFrame(data, index=index)

    for i in index:
        key = int(round(income.loc[i], -4))
        key = f"{key}-{key+step}"  # 30000_40000,
        df[key][i] = 1

    return df


def clean_portfolio(portfolio_path):
    portfolio = pd.read_json(portfolio_path, orient="records", lines=True)
    portfolio.rename(columns={"id": "offer_id"}, inplace=True)
    portfolio.set_index("offer_id", inplace=True)
    portfolio = portfolio[portfolio["offer_type"] != "informational"]

    # one hot encoding offer type
    offer_type = one_hot_encode(portfolio["offer_type"])
    portfolio = pd.concat((portfolio, offer_type), axis=1)
    portfolio.drop(["offer_type"], axis=1, inplace=True)

    # one hot encoding channels
    channels = one_hot_encode_channels(portfolio)
    portfolio = pd.concat((portfolio, channels), axis=1)
    portfolio.drop(["channels"], axis=1, inplace=True)

    return portfolio


def join_data(df_1, df_2):
    columns = [*df_1.columns, *df_2.columns]

    df = pd.DataFrame({col: [] for col in columns})

    i = 0
    for offer in df_1.index:
        for consumer in df_2.index:
            df.loc[i] = [*df_1.loc[offer], *df_2.loc[consumer]]
            i += 1

    return df


def join_consumer_data(consumer_data_list):
    if len(consumer_data_list) < 2:
        raise ValueError("list must contain at least 2 DataFrame")

    res = consumer_data_list[0]

    for d in consumer_data_list[1:]:
        res = res.join(d, how="inner")

    return res
