import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from helper.data_cleaning import join_coupon_profile_data, join_consumer_data


def calc_coupon_x_consumer_data(
    index, portfolio, profile, target, consumer_feature, consumer_sensitivity
):
    index = portfolio.index[index : index + 1]
    coupon_x = portfolio.loc[index]
    coupon_x_consumer_data = join_coupon_profile_data(coupon_x, profile)
    coupon_x_consumer_data.set_index(profile.index, inplace=True)

    coupon_x_target = target[index]

    train_df = join_consumer_data(
        [
            coupon_x_target,
            consumer_feature,
            coupon_x_consumer_data,
            consumer_sensitivity,
        ]
    )

    return train_df


def run_predict_random_forest(
    index, portfolio, profile, target, consumer_feature, consumer_sensitivity
):

    coupon_x_consumer_data = calc_coupon_x_consumer_data(
        index, portfolio, profile, target, consumer_feature, consumer_sensitivity
    )
    columns = coupon_x_consumer_data.columns

    X_train, X_test, y_train, y_test = train_test_split(
        coupon_x_consumer_data[columns[1:]],
        coupon_x_consumer_data[columns[0]],
        test_size=0.30,
        random_state=8,
    )

    clf = RandomForestClassifier(
        random_state=8, n_estimators=800, min_samples_split=40, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    feat_importance = pd.Series(clf.feature_importances_, index=columns[1:])
    feat_importance.nlargest(20).plot(kind="barh")
    y_pred = clf.predict(X_test)
    print(f"coupon {columns[0]}")
    print(f"roc_auc score is {roc_auc_score(y_test, y_pred)}")
    print(f"accuracy score is {accuracy_score(y_test, y_pred)}")
    print(f"recall score is {recall_score(y_test, y_pred)}")
    print(f"precision score is {precision_score(y_test, y_pred)}")
    print(f"f1 score is {f1_score(y_test, y_pred)}")
    print("---------------------------------------------------------------")

    return {
        "clf": clf,
        "roc_auc": roc_auc_score(y_test, y_pred),
        "consumer_data": coupon_x_consumer_data,
    }


def run_predict_knn(
    index, portfolio, profile, target, consumer_feature, consumer_sensitivity
):
    coupon_x_consumer_data = calc_coupon_x_consumer_data(
        index, portfolio, profile, target, consumer_feature, consumer_sensitivity
    )

    columns = coupon_x_consumer_data.columns

    X_train, X_test, y_train, y_test = train_test_split(
        coupon_x_consumer_data[columns[1:]],
        coupon_x_consumer_data[columns[0]],
        test_size=0.30,
        random_state=8,
    )

    clf = KNeighborsClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"coupon {columns[0]}")
    print(f"roc_auc score is {roc_auc_score(y_test, y_pred)}")
    print(f"accuracy score is {accuracy_score(y_test, y_pred)}")
    print(f"recall score is {recall_score(y_test, y_pred)}")
    print(f"precision score is {precision_score(y_test, y_pred)}")
    print(f"f1 score is {f1_score(y_test, y_pred)}")
    print("---------------------------------------------------------------")

    return {
        "clf": clf,
        "roc_auc": roc_auc_score(y_test, y_pred),
        "consumer_data": coupon_x_consumer_data,
    }
