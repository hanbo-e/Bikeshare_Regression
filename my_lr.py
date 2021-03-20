import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read in data
df = pd.read_csv('bike-sharing-demand/train.csv', parse_dates = True, index_col = 0)

# train/test split
X  = df.drop(columns = ['count'])
y = df['count']
Xtr, Xtst, ytr, ytst = train_test_split(X, y, random_state = 42)

# exploratory data analysis

#check for NAs
Xtr.isna().any()

#scatterplots
cols = list(df.columns)
for col in cols:
    df.plot.scatter(x = col, y = 'count')
    plt.show
#based on scatterplots I am interested in the following features:
features = ['windspeed', 'temp', 'season', 'holiday', 'workingday']

#loop through features and get model scores

#accumulator for number of features used
f = []

#store scores in a dataframe
scores = pd.DataFrame(columns=('features','tr_score', 'tst_score', 'coef', 'intercept'))

#add row of data with each iteration
new_row = {'features':[],'tr_score':[], 'tst_score':[], 'coef':[], 'intercept':[]}

for feature in features:
    #add feature to f
    f.append(feature)
    new_row['features'] = f
    #print('appended fs: ', f)
    #print(new_row['features'])
    lin_r = LinearRegression()
    lin_r.fit(Xtr[f], ytr)
    tr_score = lin_r.score(Xtr[f], ytr)
    #print('trscore: ', tr_score)
    #add to new_row
    new_row['tr_score'] = tr_score
    tst_score = lin_r.score(Xtst[f], ytst)
    #print('test score: ', tst_score)
    new_row['tst_score'] = tst_score
    coef = lin_r.coef_
    #print('coef: ', coef)
    new_row['coef'] = coef
    intercept = lin_r.intercept_
    #print('intercept: ', intercept)
    new_row['intercept'] = intercept
    #add new_row to df
    #scores = scores.append(new_row, ignore_index = True)
    my_row = pd.Series(data = new_row, name=feature)
    scores = scores.append(my_row, ignore_index = False)

#use windspeed, temp, and workingday in my model
#generate predictions
dftest = pd.read_csv('bike-sharing-demand/test.csv', index_col = 0, parse_dates = True)
lin_r = LinearRegression()
lin_r.fit(Xtr[['windspeed', 'temp', 'workingday']], ytr)
lin_r.score(Xtr[['windspeed', 'temp', 'workingday']], ytr)
lin_r.score(Xtst[['windspeed', 'temp', 'workingday']], ytst)

##prepare data for kaggle

predictions = lin_r.predict(dftest[['windspeed', 'temp', 'workingday']])
predictions = predictions.astype(int)
temp = pd.Series(predictions)
temp = temp.where(predictions > 0, 0)
#prepare predictions for Kaggle submission
output = pd.DataFrame({'datetime':dftest.index, 'count': temp})
output.to_csv('havaoz_bike_01.csv', index = False, sep = ',')
# feature engineering
#predict
#lin_r.predict([[40]])
# cross-validation
# optimize the model iteratively, select features, try different regressors (e.g. Linear Regression, Random Forest Regressor, SVR)
# calculate a test score when you are done
# submit to Kaggle
