import os

from helper.data_cleaning import clean_profile, one_hot_encode


def test_one_hot_encode(test_data_dir):
    profile = clean_profile(os.path.join(test_data_dir, "profile.json"))

    print(one_hot_encode(profile["year"])[:5])
