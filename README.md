# Starbuck Machine Learning Project

**Author:** Leo Li
**Mentor:** Kanja Saha

---

# Installation

make sure you are using python 3.7+

```bash
pip install -r requirement.txt
```

# Project Proposal

## 1. Problem Statement

This is a supervised learning classification problem. Specifically, it is a multi-label classification problem.

#### Here is a brief description of the problem

We have 8 different type of coupons. For each consumer, during the given time period of 30 days, we know if a user has successfully used a coupon. Since a consumer can use multiple coupon (these events are not exclusive to each other), it is a `multi-label binary classification problem`.

A succcessful model can predict how likely a consumer will use one of the 8 types of coupons. Hense help the brand to send promotion to the target consumer, and, of course, capture to most value out of the person.

## 2. Domain Knowledge

The business world is changing. With massive consumer traffic data available we can understand consumer behavior deeper than ever before. Brands can capture more value by `correctly giving out coupons` to stimulate consumers that are “sensitive” to these rewards, in the same time, `avoid` giving coupon to consumer that are already buying the product.

The machine learning knowledge here is to tackle a `multi-label  binary classification problem`. A good reference paper can be found here:
[Relevance efficacy for multilabel classification](https://link.springer.com/content/pdf/10.1007%2Fs13748-012-0030-x.pdf)

## 3. Benchmark Model

Use `K-nearest neighbor` model as it is a fast and standard method for binary classification machine learning problems.

Reason for using KNN:
1. It provides a quick way to train the model and test the result.
2. Because we only have `12` features, KNN will provide a fairly good result as its dimension is low.

That being said, a quick and fairly accurate model will make it a good benchmark model.

## 4. Evaluation Metrics

### ROC_AUC
Since it is a binary classfication problem, we will evaluate with `Area under the Curve (AUC) of Receiver Operating Characteristic (ROC) `
![roc_auc](https://miro.medium.com/max/722/1*pk05QGzoWhCgRiiFbz-oKQ.png)
source: [understanding-auc-roc-curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

### Recall

We would rather give coupon to a consumer, even he/she won't use it, than missing a consumer that are potentially buying coffee. So recall is a more important measure than precision
![recall](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png)
source: [Recall definition from Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)

### Feature Importance

We will also run a `feature importance analysis` on the features. This can be done using `feature_importances_`

```python
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(os.path.join('assets/', 'train_test.csv'))

X = df[:, 1:]
y = df[:, 0]
clf = RandomForestClassifier(n_estimators=100,
                             random_state=11,
                            min_samples_split=10)
                            
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
```

## 5. Dataset and Inputs

#### Raw Dataset
#### 5.1 Portfolio

There are a total of `10` type of promotions, but `2` of them are just informational, hense we don't care. The other `8`
 are actual coupons

```bash
id	                             reward	   channels	  difficulty	duration	offer_type				
ae264e3637204a6fb9bb56bc8210ddfd	10	[email, mobile, social]	10	7	bogo
4d5c57ea9a6940dd891ad53e9dbe8da0	10	[web, email, mobile, social]	10	5	bogo
3f207df678b143eea3cee63160fa8bed	0	[web, email, mobile]	0	4	informational
9b98b8c7a33c4b65b9aebfe6a799e6d9	5	[web, email, mobile]	5	7	bogo
0b1e1539f2cc45b7b9fa7c272da2e1d7	5	[web, email]	20	10	discount
2298d6c36e964ae4a3e7e9706d1fb8c2	3	[web, email, mobile, social]	7	7	discount
fafdcd668e3743c1bb461111dcafc2a4	2	[web, email, mobile, social]	10	10	discount
5a8bc65990b245e5a138643cd4eb9837	0	[email, mobile, social]	0	3	informational
f19421c1d4aa40978ebb69ca19b0e20d	5	[web, email, mobile, social]	5	5	bogo
2906b810c7d4411798c6938adc9daaa5	2	[web, email, mobile]	10	7	discount
```
Things that are interesting to us:

`difficulty`: how much do you need to spend to trigger the reward;

`duration`: maximum time for the user to complete the offer;

#### 5.2 Consumer Profile
Let's print out the top 5 rows of the table.
```bash
	gender	age	id	became_member_on	income
0	None	118	68be06ca386d4c31939f3a4f0e3dd783	20170212	NaN
1	F	55	0610b486422d4921ae7d2bf64640c50b	20170715	112000.0
2	None	118	38fe809add3b4fcf9315a9694bb96ff5	20180712	NaN
3	F	75	78afa995795e4d85b5d9ceeca43f5fef	20170509	100000.0
4	None	118	a03223e636434f42ac4c3df47e8bac43	20170804	NaN
```
In this dataset there are data that are `not valid`, if we look at `gender`, `age` and `income` variables, some values don't make sense or are `None`s.

The total valid consumer number is `14825`.

#### 5.3 Transcript
```bash
person	event	value	time
0	78afa995795e4d85b5d9ceeca43f5fef	offer received	{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}	0
1	a03223e636434f42ac4c3df47e8bac43	offer received	{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}	0
2	e2127556f4f64592b11af22de27a7932	offer received	{'offer id': '2906b810c7d4411798c6938adc9daaa5'}	0
3	8ec6ce2a7e7949b1bf142def7d0e0586	offer received	{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}	0
4	68617ca6246f4fbc85e91a2a49552598	offer received	{'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}	0
```
event has 4 types: `offer received`, `offer viewed`, `offer complete`, and `transactions`.

We want to know if the user has successfully complete an offer. We will use the above information to create the target table


We need to create both `features` and `targets`, let's work on `target` first.

#### 5.4 Target Inputs

This table represent for each `consumer (row)`, if they have complete one of the eight `coupon (column)` during the given time frame (30 days)
```bash
	person	ae264e3637204a6fb9bb56bc8210ddfd	4d5c57ea9a6940dd891ad53e9dbe8da0	9b98b8c7a33c4b65b9aebfe6a799e6d9	0b1e1539f2cc45b7b9fa7c272da2e1d7	2298d6c36e964ae4a3e7e9706d1fb8c2	fafdcd668e3743c1bb461111dcafc2a4	f19421c1d4aa40978ebb69ca19b0e20d	2906b810c7d4411798c6938adc9daaa5							
7366bef4c288476dab78b09a33d0e692	0	1	0	0	0	0	0	0
b854f7297f9e4ecb9d02845941b87479	0	0	0	0	0	0	0	0
b912b714bf5e40609f6ff25a9a542a9c	0	0	0	0	0	1	0	0
46b3c686bbbd4495940b4da488e10ed6	1	1	0	0	1	0	0	0
a14d4f2ec359464f8d4aeac17b755903	1	0	0	0	1	1	0	
```
Total `14825` rows and `8` columns
Columns are `coupon id`, rows are `consumer id`

#### 5.5 Feature Inputs

1. Profile related
`gender`, `age`, `income`, `time since membership`

I would love to see how well only these 4 features performs, as these features are easy to obtain for Starbucks. And then decided if we should include more information



2. Consumer behavior related
`Avg Daily spending on Coffee(float)`, `Highest daily spending(float)`,`lowest spending (not include 0)(float)`, `std spending (not include 0)(float)`, `days in a month spending below $5(int)`, `days in a month spending from $5 to $10(int)`, `days in a month spending from $10 to $15(int)`, `days in a month spending from $15 to $20(int)`, `days in a month spending over $20(int)`

Here are the top 5 rows of the dataframe
```bash
Avg Daily spending	Highest daily spending	Lowest daily spending	count days no spending	count days spending 0_to_5	count days spending 5_to_10	count days spending 10_to_15	count days spending 15_to_20	count days spending 20_plus	std_daily_spending
person										
0009655768c64bdeb2e877511632db8f	4.253333	28.16	8.57	22.0	0.0	1.0	1.0	1.0	2.0	7.867232
00116118485d4dfda04fdbaba9a87b5c	0.136333	3.39	0.70	28.0	2.0	0.0	0.0	0.0	0.0	0.627653
0011e0d4e6b944f998e987f904e8c1e5	2.648667	23.03	8.96	25.0	0.0	1.0	1.0	0.0	2.0	6.461309
0020c2b971eb4e9188eac86d93036a77	6.562000	49.63	17.24	24.0	0.0	0.0	0.0	1.0	5.0	14.444319
0020ccbbb6d84e358d3414a3ff76cffd	5.135000	30.84	6.81	19.0	0.0	3.0	3.0	2.0	1.0	7.923300
```

## 6. Solution Statement

Use `Adaboost/Gradient Boosting` along with `Decision Tree` Classifier or `Random Forest` method to predict what coupon should we give to a user based on input features


### Workflow
There are `3` steps for our pipeline 

 #### 6.1 Data Cleaning and Reformatting
 The raw data we have are `portfolio`, `profile` and `portfolio`. We want to know for each consumer, what are the trend
 of their buying activities (trends) as well as   
 
 - We dropped user profile that has missing data 
 - we group the transaction into a day by day format (column) by each consumer (row)
 - based on `view` and `complete` info of transaction on each day, we can calculate on each day if a consumer has a
 valid coupon completion, and what is the `id` of this coupon (see `out/valid_complete_day_df.csv`), and we can group 
 them by coupon `id` (see `training_data/target.csv`), this is our target data.
 
 ![consumer 1](https://github.com/LeoYuanjieLi/starbucks_ml/blob/master/assets/user_1.png)
 ![consumer 2](https://github.com/LeoYuanjieLi/starbucks_ml/blob/master/assets/user_2.png)
 
 consumer trend examples
 
 #### 6.2 Feature Engineering
 
 #### Feature - Profile
 - we turn `profile` data into a one-hot-encoding format:
    - Rename "id" to "consumer_id"
    - Drop all `nan` data
    - Drop "gender" that are in the `O` catergory
    - Split date into `year`, `month`, `day`
    - One Hot Encode `date`, `gender`, `age` and `income`

```bash
2013	2014	2015	2016	2017	2018	01	02	03	04	...	30000-40000	40000-50000	50000-60000	60000-70000	70000-80000	80000-90000	90000-100000	100000-110000	110000-120000	120000-130000
consumer_id																					
0610b486422d4921ae7d2bf64640c50b	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
78afa995795e4d85b5d9ceeca43f5fef	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0
e2127556f4f64592b11af22de27a7932	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0
389bc3fa690240e798340f5a15918d5c	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1.0	0.0	0.0	...	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
2eeac8d8feae4a8cad5a6af0499a211d	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
```
     
 
 #### Feature - Portfolio
 Things that are interesting to us are:

- `difficulty`: how much do you need to spend to trigger the reward;

- `duration`: maximum time for the user to complete the offer;

- `offer_type`: Buy one get one free or just discount

We know that there are 10 different offers, and `8` of them are actually coupons (we don't care about informational).

- Offer types are our `target data`. In the end, we want need a True/False Table for each coupon map to each user

Perform the following steps to preprocess the data.

- Rename "id" to "offer_id"
- Drop "informational" offer type
- One Hot Encode `offer type` and `channels`
 
```bash
	reward	difficulty	duration	bogo	discount	email	mobile	web	social
offer_id									
ae264e3637204a6fb9bb56bc8210ddfd	10	10	7	1.0	0.0	1.0	1.0	0.0	1.0
4d5c57ea9a6940dd891ad53e9dbe8da0	10	10	5	1.0	0.0	1.0	1.0	1.0	1.0
9b98b8c7a33c4b65b9aebfe6a799e6d9	5	5	7	1.0	0.0	1.0	1.0	1.0	0.0
0b1e1539f2cc45b7b9fa7c272da2e1d7	5	20	10	0.0	1.0	1.0	0.0	1.0	0.0
2298d6c36e964ae4a3e7e9706d1fb8c2	3	7	7	0.0	1.0	1.0	1.0	1.0	1.0
fafdcd668e3743c1bb461111dcafc2a4	2	10	10	0.0	1.0	1.0	1.0	1.0	1.0
f19421c1d4aa40978ebb69ca19b0e20d	5	5	5	1.0	0.0	1.0	1.0	1.0	1.0
2906b810c7d4411798c6938adc9daaa5	2	10	7	0.0	1.0	1.0	1.0	1.0	0.0
```

#### Features - Consumer Trend

Based on the transcript information, create a table that contains the following features

- Avg Daily spending: The mean of spending of each day
- Highest daily spending: The max of daily spending
- Lowest daily spending: The min of daily spending
- std_daily_spending: The standard deviation of daily spending
- Count days no spending: The number of days consumer did not spend on starbucks coffee
- count days spending 0_to_5: The number of days consumer spend 0 - 5 on starbucks coffee
- count days spending 5_to_10: The number of days consumer spend 5 - 10 on starbucks coffee
- count days spending 10_to_15: The number of days consumer spend 10 - 15 on starbucks coffee
- count days spending 15_to_20: The number of days consumer spend 15 - 20 on starbucks coffee
- count days spending 20_plus: The number of days consumer spend 20 plus on starbucks coffee
 
 ```bash
                                   Avg Daily spending	Highest daily spending	Lowest daily spending	count days no spending	count days spending 0_to_5	count days spending 5_to_10	count days spending 10_to_15	count days spending 15_to_20	count days spending 20_plus	std_daily_spending
consumer_id										
0009655768c64bdeb2e877511632db8f	4.253333	28.16	8.57	22.0	0.0	1.0	1.0	1.0	2.0	7.867232
00116118485d4dfda04fdbaba9a87b5c	0.136333	3.39	0.70	28.0	2.0	0.0	0.0	0.0	0.0	0.627653
0011e0d4e6b944f998e987f904e8c1e5	2.648667	23.03	8.96	25.0	0.0	1.0	1.0	0.0	2.0	6.461309
0020c2b971eb4e9188eac86d93036a77	6.562000	49.63	17.24	24.0	0.0	0.0	0.0	1.0	5.0	14.444319
0020ccbbb6d84e358d3414a3ff76cffd	5.135000	30.84	6.81	19.0	0.0	3.0	3.0	2.0	1.0	7.923300
```

#### Feature - Consumer Coupon Sensitivity

Let's take a look at how consumer `react` to coupons:

![consumer example](https://github.com/LeoYuanjieLi/starbucks_ml/blob/master/assets/consumer_example.png)


Based on this data visualization, we can tell the `view` activity has so what a connection to the spending activity:

For these 2 consumer, at least, it looks like `more spending activities are followed by the view acitivity`.

It will be helpful to create a feature to describe this phenomenon - `Coupon Sensitivity`

Here is how it works:

For the duration of the coupon (in the following days), how much `more` does this person spend compare to the purchase that are not initiated by the coupon - those purchases not in the following days of a coupon view event.

Formula for `coupon sensitivity`

`(avg_couponx_effective_spend - avg_spend_without_coupon) / coupon_difficulty`

- avg_couponx_effective_spend: if a consumer views a coupon, if he/she spend in the following days within its duration, these amount are called couponx_effective_spend, we then take the average.

- avg_spend_without_coupon: if a consumer spend on coffee in the days that are not affect by ANY coupon

- coupon difficulty: the difficulty of completing the coupon, this is to normalize each coupon

```bash
	avg_spend_without_coupon	ae264e3637204a6fb9bb56bc8210ddfd_type_spend	4d5c57ea9a6940dd891ad53e9dbe8da0_type_spend	9b98b8c7a33c4b65b9aebfe6a799e6d9_type_spend	0b1e1539f2cc45b7b9fa7c272da2e1d7_type_spend	2298d6c36e964ae4a3e7e9706d1fb8c2_type_spend	fafdcd668e3743c1bb461111dcafc2a4_type_spend	f19421c1d4aa40978ebb69ca19b0e20d_type_spend	2906b810c7d4411798c6938adc9daaa5_type_spend	ae264e3637204a6fb9bb56bc8210ddfd_type_sensitivity	4d5c57ea9a6940dd891ad53e9dbe8da0_type_sensitivity	9b98b8c7a33c4b65b9aebfe6a799e6d9_type_sensitivity	0b1e1539f2cc45b7b9fa7c272da2e1d7_type_sensitivity	2298d6c36e964ae4a3e7e9706d1fb8c2_type_sensitivity	fafdcd668e3743c1bb461111dcafc2a4_type_sensitivity	f19421c1d4aa40978ebb69ca19b0e20d_type_sensitivity	2906b810c7d4411798c6938adc9daaa5_type_sensitivity
consumer_id																	
0009655768c64bdeb2e877511632db8f	3.815000	0.000000	0.000	0.000000	0.000	0.000000	12.108750	5.534000	0.00000	-0.381500	-0.381500	-0.763000	-0.190750	-0.545000	0.829375	0.343800	-0.381500
00116118485d4dfda04fdbaba9a87b5c	0.227222	0.000000	0.000	0.000000	0.000	0.000000	0.000000	0.077778	0.00000	-0.022722	-0.022722	-0.045444	-0.011361	-0.032460	-0.022722	-0.029889	-0.022722
0011e0d4e6b944f998e987f904e8c1e5	1.887778	0.000000	0.000	7.720000	5.404	1.704286	0.000000	0.000000	0.00000	-0.188778	-0.188778	1.166444	0.175811	-0.026213	-0.188778	-0.377556	-0.188778
0020c2b971eb4e9188eac86d93036a77	2.838889	0.000000	3.448	0.000000	0.000	0.000000	9.833000	0.000000	0.00000	-0.283889	0.060911	-0.567778	-0.141944	-0.405556	0.699411	-0.567778	-0.283889
0020ccbbb6d84e358d3414a3ff76cffd	7.570000	0.000000	0.000	2.965000	0.000	6.655714	0.000000	10.860000	0.00000	-0.757000	-0.757000	-0.921000	-0.378500	-0.130612	-0.757000	0.658000	-0.757000
```
 
 #### 6.3 Training & Evaluation
 
## Training & Evaluation

We will perform the following steps for our Training (evaluation metrix: with ROC_AUC / Recall / Accuracy):

 1. We will first run a KNN only based on all features
 2. We run a feature importance analysis on features, if necessary, we reduce the dimension using PCA
 3. We apply random forest classifier on other target coupons
 4. Improve with Hyper-parameter tuning (Optional)
 5. Wrap the models to a single function that takes a consumer id and return the best coupon for this consumer
     - I will use the mathematic expectation, fomula:
     
         `expectation = roc_auc * difficulty` 
         
         (the more difficult, the more that consumer will spend)
     
     
# Complete Project

    - You can find the complete project in the `StarbucksML` notebook
    - You can also find the pdf version in `StarbucksMachineLearningLeoLi.pdf`