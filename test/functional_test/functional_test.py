import os

from helper.data_cleaning import clean_profile


def test_clean_profile(test_data_dir):
    print(clean_profile(os.path.join(test_data_dir, "profile.json"))[:5])
