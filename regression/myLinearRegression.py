import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt','r').read()
should_reset = False


def get_shares(reset):
    if reset:
        googl_df = quandl.get('WIKI/GOOGL', api_key=api_key)
        googl_df.to_pickle('googl.pickle')
        return googl_df
    else:
        googl_df = pd.read_pickle('googl.pickle')
        return googl_df


df = get_shares(should_reset)
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# ML Regression

X = np.array(df.drop('label', 1))
y = np.array(df.label)

X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(accuracy)

# With svm
model_svm = svm.SVR()
model_svm.fit(X_train, y_train)

accuracy_svm = model_svm.score(X_test, y_test)
print(accuracy_svm)
