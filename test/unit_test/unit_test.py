import os
import pandas as pd
from helper.data_cleaning import (
    clean_profile,
    one_hot_encode,
    one_hot_encode_channels,
    join_data,
    clean_portfolio,
    one_hot_encode_age,
    one_hot_encode_income,
)


def test_one_hot_encode(test_data_dir):
    profile = clean_profile(os.path.join(test_data_dir, "profile.json"))

    print(one_hot_encode(profile["year"])[:5])


def test_one_hot_encode_channels(test_data_dir):
    portfolio_path = os.path.join(test_data_dir, "portfolio.json")
    portfolio = pd.read_json(portfolio_path, orient="records", lines=True)
    channels = one_hot_encode_channels(portfolio)
    print(channels)


def test_join_data(test_data_dir):
    profile = clean_profile(os.path.join(test_data_dir, "profile.json"))
    portfolio = clean_portfolio(os.path.join(test_data_dir, "portfolio.json"))
    res = join_data(portfolio, profile[:5])
    assert res.shape[0] == portfolio.shape[0] * profile[:5].shape[0]


def test_one_hot_encode_age(test_data_dir):
    profile_path = os.path.join(test_data_dir, "profile.json")
    profile = pd.read_json(profile_path, orient="records", lines=True)
    profile.set_index("id", inplace=True)
    res = one_hot_encode_age(profile)

    print(res)


def test_one_hot_encode_income(test_data_dir):
    profile_path = os.path.join(test_data_dir, "profile.json")
    profile = pd.read_json(profile_path, orient="records", lines=True)
    profile.set_index("id", inplace=True)
    print(one_hot_encode_income(profile))
