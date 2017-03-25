import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

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

# ML Regression

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df.label)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(accuracy)

# With svm
model_svm = svm.SVR()
model_svm.fit(X_train, y_train)

accuracy_svm = model_svm.score(X_test, y_test)
print('Accuracy:', accuracy_svm)


forecast_set = model.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

# For plot and add data to same df

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
# Seconds
one_day = 24 * 60 * 60
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

