import xgboost
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve


def read_and_preprocess(file):
    df = pd.read_excel(file, engine='openpyxl')

    df = df.replace('zero', -1)
    df = df.replace('one', 1)

    df = df.replace('LEAVE', -1)
    df = df.replace('STAY', 1)

    label_col = df.pop('LEAVE')

    one_hot_1 = pd.get_dummies(df['REPORTED_SATISFACTION'],
                               prefix='SATISFACTION')
    one_hot_2 = pd.get_dummies(df['REPORTED_USAGE_LEVEL'], prefix='USAGE')
    one_hot_3 = pd.get_dummies(df['CONSIDERING_CHANGE_OF_PLAN'],
                               prefix='CONSIDERING')

    df = df.drop('REPORTED_SATISFACTION', axis=1)
    df = df.drop('REPORTED_USAGE_LEVEL', axis=1)
    df = df.drop('CONSIDERING_CHANGE_OF_PLAN', axis=1)

    df = df.join(one_hot_1)
    df = df.join(one_hot_2)
    df = df.join(one_hot_3)

    df['LEAVE'] = label_col

    return df


def split_train_label(dataframe):
    data = dataframe.iloc[:, :-1]
    label = dataframe.iloc[:, -1]
    return data, label


def split_train_test(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2)
    return X_train, X_test, y_train, y_test


def split_val_test(data, labels):
    X_val, X_test, y_val, y_test = train_test_split(data, labels, test_size=0.5)
    return X_val, X_test, y_val, y_test


def build_xgboost_classifier(train_data, train_label):
    model = xgboost.XGBClassifier(max_depth=5, n_estimators=500,
                                  min_child_weight=5, learning_rate=0.02)
    model.fit(train_data, train_label)
    return model


def get_acc(model, test_data, test_labels):
    y_pred = model.predict(test_data)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


file = '../data/Customer_Churn.xlsx'
preprocessed_data = read_and_preprocess(file)

data, labels = split_train_label(preprocessed_data)
X_train, X_test, y_train, y_test = split_train_test(data, labels)
X_val, X_test, y_val, y_test = split_val_test(X_test, y_test)

classifier = build_xgboost_classifier(X_train, y_train)

acc = get_acc(classifier, X_test, y_test)
print("Accuracy for Gradient Boosted DT: {}".format(acc))

# ------ For Plotting Feature Importance ---------

# xgboost.plot_importance(classifier)
# plt.rcParams['figure.figsize'] = [5, 5]
# plt.savefig('gradient_boost_tree_feat_importance.png')

# ------ For Plotting ROC Curve ---------

# y_score2 = classifier.predict_proba(X_test)[:,1]
#
# false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test,
#                                                                   y_score2)
# print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_score2))
#
# # Plot ROC curves
# plt.subplots(1, figsize=(10, 10))
# plt.title('Receiver Operating Characteristic - Gradient Boosted DT')
# plt.plot(false_positive_rate2, true_positive_rate2)
# plt.plot([0, 1], ls="--")
# plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
