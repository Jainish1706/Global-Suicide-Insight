

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings
from scipy import stats

#Loading data into dataframe

data = pd.read_csv("/content/suicide_data.csv")
data.head()

#Shape of dataframe

data.shape

#Listing the features of the dataset

data.columns

#Information about the dataset

data.info()

data.age.value_counts()

data.generation.value_counts()

#Listing countries

country = data.country.unique()
print("Number of countries:", len(country))
country

"""Visualization and EDA"""

data.hist(bins = 50,figsize = (15,11))

#Correlation heatmap
numerical_data = data.select_dtypes(include=np.number)
plt.figure(figsize=(7,5))
sns.heatmap(numerical_data.corr(), annot=True, cmap='Oranges')
plt.show()

#Gender and suicide count bar plot

plt.figure(figsize=(10,3))
sns.barplot(x='sex', y='suicides_no', data=data)
plt.title('Gender - Suicide Count Bar Plot')
plt.show()

#Age Group - Count Bar Plot Grouped by Gender

plt.figure(figsize=(10,3))
sns.barplot(x = "age", y = "suicides_no", hue = "age", data = data)
plt.title("Age Group - Count Bar Plot Grouped by Gender")
plt.show()

#Generation - Count Bar Plot grouped by Gender

plt.figure(figsize=(9,5))
sns.barplot(x = "generation", y = "suicides_no", hue = "sex", data = data)
plt.title('Generation - Count Bar Plot grouped by Gender')
plt.show()

# Age Group and Suicide count bar plot

plt.figure(figsize=(9,5))
sns.barplot(x=data['age'], y=data['suicides_no'])
plt.xlabel('Age Group')
plt.ylabel('Suicide Count')
plt.title('Age Group - Suicide Count Bar Plot')
plt.show()

#Generation & Suicide Count Bar Plot

plt.figure(figsize=(9,5))
sns.barplot(x=data['generation'], y=data['suicides_no'])
plt.xlabel('Generation')
plt.ylabel('Suicide Count')
plt.title('Generation - Suicide Count Bar Plot')
plt.show()

#Gender & Sucide Count grouped by Age Group bar plot

plt.figure(figsize=(7,7))
sns.barplot(y="sex", x="suicides_no", hue="age", data=data)
plt.title('Gender & Sucide Count grouped by Age Group')
plt.show()

#Gender & Sucide Count grouped by Generation bar plot

plt.figure(figsize=(7,7))
sns.barplot(y="sex", x="suicides_no", hue="generation", data=data)
plt.title('Gender & Sucide Count grouped by Generation')
plt.show()

#Country & Suicide_rate Bar plot

plt.figure(figsize=(15,25))
sns.barplot(x = "suicides/100k pop", y = "country", data = data)
plt.title('Country - Suicide_rate Bar plot')
plt.show()

#Line plpot of year and suicide_rate

data[['year','suicides/100k pop']].groupby(['year']).sum().plot()

#Scatter matrix for checking outlier

plt.figure(figsize=(20,10))
attributes = ['suicides_no', 'population', 'suicides/100k pop','HDI for year',
              ' gdp_for_year','gdp_per_capita']
scatter_matrix(data[attributes], figsize=(20,10))
plt.show()

"""Feature Selection using Variance Threshold, RandomForest, Correlation and ANOVA"""

# Data Cleaning
data[' gdp_for_year'] = data[' gdp_for_year'].replace({',': ''}, regex=True).astype(float)
data = data.drop(columns=['country-year', ' gdp_for_year'])  # Drop redundant columns
data['HDI for year'] = data['HDI for year'].fillna(data['HDI for year'].mean())

# Encoding categorical variables
label_encoders = {}
for column in ['country', 'sex', 'age', 'generation']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Feature and target variables
X = data.drop(columns=['suicides_no'])  # Assuming 'suicides_no' is the target
y = data['suicides_no']

# 1. Variance Threshold - Remove low variance features
variance_selector = VarianceThreshold(threshold=0.01)
X_variance_reduced = variance_selector.fit_transform(X)
low_variance_features = X.columns[~variance_selector.get_support()]
print("Low variance features:", low_variance_features.tolist())

# 2. Correlation Analysis - Identify highly correlated features
correlation_matrix = X.corr()
high_corr_features = correlation_matrix.columns[(correlation_matrix.abs() > 0.8).sum() > 1]
print("Highly correlated features (correlation > 0.8):", high_corr_features.tolist())

# 3. ANOVA - F-Test to find important features
anova_selector = SelectKBest(score_func=f_classif, k=5)
X_anova_selected = anova_selector.fit_transform(X, y)
anova_selected_features = X.columns[anova_selector.get_support()]
print("ANOVA selected features:", anova_selected_features.tolist())

# 4. Feature Importance using Random Forest with Sampling
# Sample the data for memory efficiency
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)  # 10% sample
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_sample, y_sample)
rf_feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Random Forest feature importances:\n", rf_feature_importances)

"""Splitting Data and selecting features"""

data = data.drop(['HDI for year', 'year'], axis = 1)
X = data.drop(columns=['suicides/100k pop'])
y = data['suicides/100k pop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape

"""Applying Models

"""

# Suppress warnings
warnings.filterwarnings("ignore")

# Model storage lists
ML_Model = []
r2_train = []
r2_test = []
rmse_train = []
rmse_test = []

# Define function to store results
def storeResults(model, train_r2, test_r2, train_rmse, test_rmse):
    ML_Model.append(model)
    r2_train.append(round(train_r2, 3))
    r2_test.append(round(test_r2, 3))
    rmse_train.append(round(train_rmse, 3))
    rmse_test.append(round(test_rmse, 3))

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
train_pred = lr_model.predict(X_train)
test_pred = lr_model.predict(X_test)
storeResults('Linear Regression', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# K-Nearest Neighbors Regression
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
train_pred = knn_model.predict(X_train)
test_pred = knn_model.predict(X_test)
storeResults('KNN Regression', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
train_pred = dt_model.predict(X_train)
test_pred = dt_model.predict(X_test)
storeResults('Decision Tree Regression', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# Random Forest Regression
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)
storeResults('Random Forest Regression', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# Gradient Boosting Regression
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
train_pred = gb_model.predict(X_train)
test_pred = gb_model.predict(X_test)
storeResults('Gradient Boosting Regression', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# XGBoost Regressor
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
train_pred = xgb_model.predict(X_train)
test_pred = xgb_model.predict(X_test)
storeResults('XGBoost Regressor', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# Bagging Regressor
bag_model = BaggingRegressor(random_state=42)
bag_model.fit(X_train, y_train)
train_pred = bag_model.predict(X_train)
test_pred = bag_model.predict(X_test)
storeResults('Bagging Regressor', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# MLP Regressor
mlp_model = MLPRegressor(random_state=42, max_iter=500)
mlp_model.fit(X_train, y_train)
train_pred = mlp_model.predict(X_train)
test_pred = mlp_model.predict(X_test)
storeResults('MLP Regressor', r2_score(y_train, train_pred), r2_score(y_test, test_pred),
             mean_squared_error(y_train, train_pred, squared=False),
             mean_squared_error(y_test, test_pred, squared=False))

# Display the results
results_df = pd.DataFrame({
    'Model': ML_Model,
    'Train R2': r2_train,
    'Test R2': r2_test,
    'Train RMSE': rmse_train,
    'Test RMSE': rmse_test
})
print(results_df)

"""**Statistical Testing**

### **Test 1: To check the difference in suicide rates between male and female**
Using independent sample t-test to check the difference in suicide rates between male and female. The hypothesis statements for this test are:

**H0:** There is no difference in the suicide rates among male and female (Null).<br>
**H1:** There is difference in the suicide rates among male and female (Alternate).
"""

#collecting male suicide rate data
male = data['suicides/100k pop'][data['sex'] == 'male']
male

#collecting female suicide rate data
female = data['suicides/100k pop'][data['sex'] == 'female']
female

#calculating p value
ttest,pval = stats.ttest_rel(male, female)

if pval<0.05:
    print("Reject null hypothesis")
else:
    print("Accept null hypothesis")

"""**Test Conclusion:** By performing T-test, the result obtained is to reject the null hypothesis. This basically means that there is different in suicide rates of male & female.

### **Test 2: To find out the dependence of suicide rate on the age.**
Finding out whether there is a dependence of suicide rate on the age using the Chi- Square test. The hypothesis statements for this test are:

**H0:** Suicide rate and age are independent (Null).<br>
**H1:** Suicide rate and age are dependent (Alternate).
"""

#Creating Contingency Table
contingency_table = pd.crosstab(data['suicides/100k pop'], data['age'])

#Significance Level 5%
alpha=0.05

chistat, p, dof, expected = stats.chi2_contingency(contingency_table )

#critical_value
critical_value=stats.chi2.ppf(q=1-alpha,df=dof)
print('critical_value:',critical_value)

print('Significance level: ',alpha)
print('Degree of Freedom: ',dof)
print('chi-square statistic:',chistat)
print('critical_value:',critical_value)
print('p-value:',p)
#Here, pvalue = 0.0 and a low pvalue suggests that your sample provides enough evidence that you can reject  H0  for the entire population.

#compare chi_square_statistic with critical_value and p-value which is the
#probability of getting chi-square>0.09 (chi_square_statistic)
if chistat>=critical_value:
    print("Reject H0,There is a dependency between Age group & Suicide rate.")
else:
    print("Retain H0,There is no relationship between Age group & Suicide rate.")

if p<=alpha:
    print("Reject H0,There is a dependency between Age group & Suicide rate.")
else:
    print("Retain H0,There is no relationship between Age group & Suicide rate.")

"""**Test Conclusion:** By performing Chi- Square test, the result obtained is to reject the null hypothesis. This basically means that there is dependency between Age group & Suicide rate."""