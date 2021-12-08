import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr


df = pd.read_excel('./Customer_Churn.xlsx', engine='openpyxl')

# General statistics for each attribute
print("{} \n".format(df.COLLEGE.describe()))
print("{} \n".format(df.INCOME.describe()))
print("{} \n".format(df.OVERAGE.describe()))
print("{} \n".format(df.LEFTOVER.describe()))
print("{} \n".format(df.HOUSE.describe()))
print("{} \n".format(df.HANDSET_PRICE.describe()))
print("{} \n".format(df.OVER_15MINS_CALLS_PER_MONTH.describe()))
print("{} \n".format(df.AVERAGE_CALL_DURATION.describe()))
print("{} \n".format(df.REPORTED_SATISFACTION.describe()))
print("{} \n".format(df.REPORTED_USAGE_LEVEL.describe()))
print("{} \n".format(df.CONSIDERING_CHANGE_OF_PLAN.describe()))
print("{} \n".format(df.LEAVE.describe()))

# Plot histogram for multi-class categorical attributes to see distribution
# df.REPORTED_SATISFACTION.hist()
# df.REPORTED_USAGE_LEVEL.hist()
# df.CONSIDERING_CHANGE_OF_PLAN.hist()

# Plot histogram for OVERAGE (Figure 1 in Report)
# df_leave = df[(df.LEAVE == 'LEAVE')]
# df_stay = df[(df.LEAVE == 'STAY')]
# df_leave.HOUSE.hist()
# df_stay.HOUSE.hist()

# Plot pairplot for INCOME and HANDSET_PRICE (Figure 2 in Report)
# df_set = df.iloc[:, [1, 5]]
# df_set_sample = df_set.sample(1000)
# sns.pairplot(df_set_sample)

# Calculate the Pearson's correlation coefficient for INCOME and HANDSET_PRICE
# corr, _ = pearsonr(df.INCOME, df.HANDSET_PRICE)
# print('Pearsons correlation: %.3f' % corr)
