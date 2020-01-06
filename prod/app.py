# Serve model as a flask application
from flask import Flask, request
from sklearn.externals import joblib
from pathlib import Path


models = []
app = Flask(__name__)


def load_model():
    global models
    # model variable refers to the global variable
    path_list = Path("./models").glob("*.sav")
    for path in path_list:
        models.append(joblib.load(path))


@app.route("/")
def home_endpoint():
    return "Hello World!"


@app.route("/predict", methods=["POST"])
def get_prediction():
    # Works only for a single sample
    if request.method == "POST":
        data = request.values["data"]  # Get data posted as a json
        print("data is", data)
        prediction = recommend_coupon_to_consumer(
            data, models
        )  # runs globally loaded model on the data

        return f"Best coupon for this consumer is {prediction[0]}, the expected spending is {prediction[1]}"


def recommend_coupon_to_consumer(consumer_id, clf_list):
    expectations = []

    for clf in clf_list:
        pred = clf["clf"].predict(
            clf["consumer_data"].loc[[consumer_id], clf["consumer_data"].columns[1:]]
        )[0]
        difficulty = clf["consumer_data"].loc[consumer_id, "difficulty"]
        roc_auc = clf["roc_auc"]

        expectations.append(pred * difficulty * roc_auc)

    max_i = 0
    max_e = expectations[max_i]

    for i, e in enumerate(expectations):
        if e > max_e:
            # update max_i and max_e
            max_i = i
            max_e = e

    best = clf_list[max_i]["consumer_data"].columns[0]
    print(
        f"best coupon for the consumer is: {best}, consumer expected spend is {max_e} "
    )
    return best, max_e


if __name__ == "__main__":
    load_model()  # load model at the beginning once only
    app.run(host="0.0.0.0", port=80)
