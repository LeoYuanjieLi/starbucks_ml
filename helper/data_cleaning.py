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
    portfolio.duration = portfolio.duration.astype(int)
    return portfolio


def join_coupon_profile_data(coupon, profile):
    columns = [*coupon.columns, *profile.columns]

    df = pd.DataFrame({col: [] for col in columns})

    i = 0
    for offer in coupon.index:
        for consumer in profile.index:
            df.loc[i] = [*coupon.loc[offer], *profile.loc[consumer]]
            i += 1

    return df


def join_consumer_data(consumer_data_list):
    if len(consumer_data_list) < 2:
        raise ValueError("list must contain at least 2 DataFrame")

    res = consumer_data_list[0]

    for d in consumer_data_list[1:]:
        res = res.join(d, how="inner")

    return res


def clean_transcript(transcript_path):
    transcript = pd.read_json(transcript_path, orient="records", lines=True)
    key_value_pairs = transcript.value.tolist()
    transcript.rename(columns={"person": "consumer_id"}, inplace=True)
    values = []
    for kv in key_value_pairs:
        values.append(list(kv.values())[0])

    transcript["offer_id"] = values
    transcript = transcript.drop(["value"], axis=1)

    return transcript


def generate_consumer_trend(transcript):
    transactions = transcript[transcript["event"] == "transaction"]
    transactions.rename(columns={"offer_id": "amount"}, inplace=True)
    consumer_trend = transactions.groupby(["consumer_id", "time"]).amount.sum()
    consumer_trend = consumer_trend.reset_index()

    consumer_trend_dict = {
        time: np.zeros(consumer_trend.consumer_id.unique().shape[0])
        for time in transcript.time.unique()
    }
    consumer_trend_dict["consumer_id"] = consumer_trend.consumer_id.unique()
    consumer_trend_df = pd.DataFrame(consumer_trend_dict)
    consumer_trend_df.set_index("consumer_id", inplace=True)

    for index, row in consumer_trend.iterrows():
        consumer_trend_df.loc[row["consumer_id"]][row["time"]] = row["amount"]

    keys = list(consumer_trend_df.keys())

    consumer_trend_day_df = pd.DataFrame(index=consumer_trend_df.index)

    for i in range(0, len(keys), 4):
        consumer_trend_day_df[i // 4] = np.sum(
            consumer_trend_df[[keys[i], keys[i + 1], keys[i + 2], keys[i + 3]]], axis=1
        )

    return consumer_trend_day_df


def generate_consumer_data(consumer_trend_day_df):
    feature_group = pd.DataFrame(index=consumer_trend_day_df.index)
    feature_group["Avg Daily spending"] = consumer_trend_day_df.mean(axis=1)
    feature_group["Highest daily spending"] = consumer_trend_day_df.max(axis=1)
    feature_group["Lowest daily spending"] = np.zeros(feature_group.shape[0])

    for row in consumer_trend_day_df.index:
        smallest = float("inf")
        for col in consumer_trend_day_df.columns:
            if 0 < consumer_trend_day_df[col][row] < smallest:
                smallest = consumer_trend_day_df[col][row]
        feature_group.at[row, "Lowest daily spending"] = smallest

    feature_group["count days no spending"] = np.zeros(feature_group.shape[0])

    for row in consumer_trend_day_df.index:
        count = consumer_trend_day_df.loc[row][
            consumer_trend_day_df.loc[row] == 0
        ].shape[0]

        feature_group.at[row, "count days no spending"] = count

    feature_group["count days spending 0_to_5"] = np.zeros(feature_group.shape[0])

    for row in consumer_trend_day_df.index:
        count = consumer_trend_day_df.loc[row][
            (consumer_trend_day_df.loc[row] > 0) & (consumer_trend_day_df.loc[row] <= 5)
        ].shape[0]

        feature_group.at[row, "count days spending 0_to_5"] = count

    feature_group["count days spending 5_to_10"] = np.zeros(feature_group.shape[0])

    for row in consumer_trend_day_df.index:
        count = consumer_trend_day_df.loc[row][
            (consumer_trend_day_df.loc[row] > 5)
            & (consumer_trend_day_df.loc[row] <= 10)
        ].shape[0]

        feature_group.at[row, "count days spending 5_to_10"] = count

    feature_group["count days spending 10_to_15"] = np.zeros(feature_group.shape[0])

    for row in consumer_trend_day_df.index:
        count = consumer_trend_day_df.loc[row][
            (consumer_trend_day_df.loc[row] > 5)
            & (consumer_trend_day_df.loc[row] <= 10)
        ].shape[0]

        feature_group.at[row, "count days spending 10_to_15"] = count

    feature_group["count days spending 15_to_20"] = np.zeros(feature_group.shape[0])

    for row in consumer_trend_day_df.index:
        count = consumer_trend_day_df.loc[row][
            (consumer_trend_day_df.loc[row] > 15)
            & (consumer_trend_day_df.loc[row] <= 20)
        ].shape[0]

        feature_group.at[row, "count days spending 15_to_20"] = count

    feature_group["count days spending 20_plus"] = np.zeros(feature_group.shape[0])

    for row in consumer_trend_day_df.index:
        count = consumer_trend_day_df.loc[row][
            consumer_trend_day_df.loc[row] > 20
        ].shape[0]

        feature_group.at[row, "count days spending 20_plus"] = count

    feature_group["std_daily_spending"] = consumer_trend_day_df.std(axis=1)

    return feature_group


def generate_view_and_complete_day_df(transcript):
    viewed = transcript[transcript["event"] == "offer viewed"]
    completed = transcript[transcript["event"] == "offer completed"]
    consumer = np.asarray(
        list(
            set(viewed.consumer_id.unique()).union(
                set(completed.consumer_id.unique().tolist())
            )
        )
    )

    viewed_trend_dict = {
        time: np.asarray(["" for i in range(consumer.shape[0])], dtype=str)
        for time in transcript.time.unique()
    }

    viewed_trend_dict["consumer_id"] = consumer

    viewed_trend = pd.DataFrame(viewed_trend_dict)

    viewed_trend.set_index("consumer_id", inplace=True)

    for index, row in viewed.iterrows():
        viewed_trend.loc[row["consumer_id"]][row["time"]] = row["offer_id"]

    viewed_trend_day_df = pd.DataFrame(index=viewed_trend.index)
    keys = list(viewed_trend.columns)
    for i in range(0, len(keys), 4):
        viewed_trend_day_df[i // 4] = ""
        for j in range(0, 4):
            viewed_trend_day_df[i // 4] += viewed_trend[keys[i + j]]
            viewed_trend_day_df[i // 4] += "_"

    for col in viewed_trend_day_df.columns:
        for row in viewed_trend_day_df.index:
            temp = [
                item
                for item in viewed_trend_day_df.loc[row][col].split("_")
                if len(item) > 0
            ]
            viewed_trend_day_df.at[row, col] = "_".join(temp)

    # completed data information
    completed_trend_dict = {
        time: np.asarray(["" for i in range(consumer.shape[0])], dtype=str)
        for time in transcript.time.unique()
    }

    completed_trend_dict["consumer_id"] = consumer

    completed_trend = pd.DataFrame(completed_trend_dict)

    completed_trend.set_index("consumer_id", inplace=True)

    for index, row in completed.iterrows():
        completed_trend.loc[row["consumer_id"]][row["time"]] = row["offer_id"]

    keys = list(completed_trend.columns)
    completed_trend_day_df = pd.DataFrame(index=completed_trend.index, dtype=str)

    for i in range(0, len(keys), 4):
        completed_trend_day_df[i // 4] = ""
        for j in range(0, 4):
            completed_trend_day_df[i // 4] += completed_trend[keys[i + j]]
            completed_trend_day_df[i // 4] += "_"

    for col in completed_trend_day_df.columns:
        for row in completed_trend_day_df.index:
            temp = [
                item
                for item in completed_trend_day_df.loc[row][col].split("_")
                if len(item) > 0
            ]

            completed_trend_day_df.at[row, col] = "_".join(temp)

    return viewed_trend_day_df, completed_trend_day_df


def generate_target(viewed_trend_day_df, completed_trend_day_df, portfolio):
    # build valid
    valid_complete_day_df = pd.DataFrame(index=completed_trend_day_df.index)

    for col in completed_trend_day_df.columns:
        valid_complete_day_df[col] = [
            "" for i in range(completed_trend_day_df.shape[0])
        ]

    for col in completed_trend_day_df.columns:
        for row in completed_trend_day_df.index:
            coupon = completed_trend_day_df[col][row]
            if coupon and len(coupon) == 32:
                start = max(col - portfolio["duration"][coupon], 0)
                for day in range(start, col + 1):
                    if viewed_trend_day_df[day][row] == coupon:
                        valid_complete_day_df.at[row, day] = coupon

            elif coupon and len(coupon) > 32:
                coupon_1, coupon_2 = coupon.split(
                    "_"
                )  # we know the max coupon complete of a day is 2, see code above

                start_1 = max(col - portfolio["duration"][coupon_1], 0)
                for day in range(start_1, col + 1):
                    if viewed_trend_day_df[day][row] == coupon_1:
                        valid_complete_day_df.at[row, day] = (
                            valid_complete_day_df[day][row] + coupon_1
                        )

                start_2 = max(col - portfolio["duration"][coupon_2], 0)
                for day in range(start_2, col + 1):
                    if viewed_trend_day_df[day][row] == coupon_2:
                        if valid_complete_day_df[day][row] == "":
                            valid_complete_day_df.at[row, day] = coupon_2
                        else:
                            valid_complete_day_df.at[row, day] = (
                                valid_complete_day_df[day][row] + "_" + coupon_2
                            )

    target_coupon = pd.DataFrame(index=valid_complete_day_df.index)

    for coupon in portfolio.index:
        if portfolio["difficulty"][coupon] != 0:
            target_coupon[coupon] = np.zeros(target_coupon.shape[0], dtype=int)

    for col in valid_complete_day_df.columns:
        for row in valid_complete_day_df.index:
            coupon = valid_complete_day_df[col][row]
            if coupon != "":
                if len(coupon) == 32:
                    target_coupon.at[row, coupon] = 1
                else:
                    coupon_1, coupon_2 = coupon.split("_")
                    target_coupon.at[row, coupon_1] = 1
                    target_coupon.at[row, coupon_2] = 1

    return target_coupon


def get_days_not_affect_by_coupon(viewed_trend_day_df, portfolio):
    day_all = set([i for i in range(30)])

    days = viewed_trend_day_df.columns

    days_not_affect_by_coupon = []

    for consumer in viewed_trend_day_df.index:
        day_spend_affect_by_coupon = set()

        for d in days:
            coupon = viewed_trend_day_df.loc[consumer][d]
            if coupon not in {
                "5a8bc65990b245e5a138643cd4eb9837",
                "3f207df678b143eea3cee63160fa8bed",
                "",
            }:  # no coupon or coupon is informational
                for i in range(portfolio.at[coupon, "duration"]):
                    if int(d) + i < len(days):
                        day_spend_affect_by_coupon.add(int(d) + i)

        day_not_affect_by_coupon = list(day_all.difference(day_spend_affect_by_coupon))

        days_not_affect_by_coupon.append(day_not_affect_by_coupon)

    df_days_not_affect_by_coupon = pd.DataFrame(
        {"days_not_affect_by_coupon": days_not_affect_by_coupon},
        index=viewed_trend_day_df.index,
    )
    return df_days_not_affect_by_coupon


def cal_avg_spend_without_coupon(consumer_trend_day_df, days_not_affect_by_coupon):
    consumer_trend_day_df["avg_spend_without_coupon"] = [
        np.nan for i in range(consumer_trend_day_df.shape[0])
    ]
    for consumer in days_not_affect_by_coupon.index:
        if consumer in consumer_trend_day_df.index:
            days = days_not_affect_by_coupon["days_not_affect_by_coupon"].values[0]
            avg = np.mean(consumer_trend_day_df.loc[consumer][days])
            consumer_trend_day_df.at[consumer, "avg_spend_without_coupon"] = avg

    consumer_sensitivity = consumer_trend_day_df[["avg_spend_without_coupon"]]
    consumer_trend_day_df.drop(["avg_spend_without_coupon"], axis=1, inplace=True)
    return consumer_sensitivity


def generate_consumer_spending_couponx(
    coupon_name,
    viewed_trend_day_df,
    portfolio,
    consumer_sensitivity,
    consumer_trend_day_df,
):
    days_affect_by_coupon_X = get_coupon_X_days(
        coupon_name, viewed_trend_day_df, portfolio
    )
    days_affect_by_coupon_X = pd.DataFrame(
        {f"{coupon_name}_type": days_affect_by_coupon_X},
        index=viewed_trend_day_df.index,
    )

    consumer_sensitivity[f"{coupon_name}_type_spend"] = np.zeros(
        consumer_sensitivity.shape[0]
    )

    for consumer in consumer_sensitivity.index:
        if consumer in days_affect_by_coupon_X.index:
            days = days_affect_by_coupon_X.loc[consumer][f"{coupon_name}_type"]
            if days:
                consumer_sensitivity.at[
                    consumer, f"{coupon_name}_type_spend"
                ] = np.mean(consumer_trend_day_df.loc[consumer][days])


def get_coupon_X_days(coupon_name, viewed_trend_day_df, portfolio):
    days_affect_by_coupon_X = []
    for consumer in viewed_trend_day_df.index:
        day_spend_affect_by_coupon = set()
        days = viewed_trend_day_df.columns
        for d in days:
            coupon = viewed_trend_day_df.loc[consumer][d]
            if coupon == coupon_name:
                for i in range(portfolio.at[coupon, "duration"]):
                    d = int(d)
                    if d + i < len(days):
                        day_spend_affect_by_coupon.add(d + i)

        days_affect_by_coupon_X.append(list(day_spend_affect_by_coupon))

    return days_affect_by_coupon_X


def generate_coupon_sensitivity(
    consumer_sensitivity, viewed_trend_day_df, portfolio, consumer_trend_day_df
):
    for coupon in portfolio.index:
        generate_consumer_spending_couponx(
            coupon,
            viewed_trend_day_df,
            portfolio,
            consumer_sensitivity,
            consumer_trend_day_df,
        )

    for coupon in portfolio.index:
        consumer_sensitivity[f"{coupon}_type_sensitivity"] = (
            consumer_sensitivity[f"{coupon}_type_spend"]
            - consumer_sensitivity["avg_spend_without_coupon"]
        ) / portfolio.loc[coupon]["difficulty"]

    consumer_sensitivity = consumer_sensitivity[consumer_sensitivity.columns[-8:]]
