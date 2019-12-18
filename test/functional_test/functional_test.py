import os

from helper.data_cleaning import clean_profile, clean_portfolio


def test_clean_profile(test_data_dir):
    print(clean_profile(os.path.join(test_data_dir, "profile.json"))[:5])


def test_clean_portfolio(test_data_dir):
    print(clean_portfolio(os.path.join(test_data_dir, "portfolio.json")))
