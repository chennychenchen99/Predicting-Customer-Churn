import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_excel('../data/Customer_Churn.xlsx', engine='openpyxl')

df = df.replace('zero', -1)
df = df.replace('one', 1)

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

del(df['LEAVE'])

df2 = df.copy()
cols = list(df.columns)

for col in cols:
    df2[col].replace((df[col] - df[col].mean())/df[col].std(ddof=0))

# estimator = KMeans(n_clusters=3).fit(df2)
# print(estimator.cluster_centers_)

cost = []
for i in range(1, 11):
    KM = KMeans(n_clusters=i, max_iter=500)
    KM.fit(df2)

    # calculates squared error for the clustered points
    cost.append(KM.inertia_)

# plot the cost against K values
plt.plot(range(1, 11), cost, color='g', linewidth='3')
plt.xlabel("Value of K")
plt.ylabel("Sqaured Error (Cost)")
plt.savefig('image.png')
