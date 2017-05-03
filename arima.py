from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())

###################### plot the series and co-relation ###################################
series.plot()
pyplot.show()

autocorrelation_plot(series)
pyplot.show()

from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame


##################### fit a random model after getting an idea about the lags ##############################
# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

#################### get an idea about the residuals ###############################################
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())



from sklearn.metrics import mean_squared_error


X = series.values
size = int(len(X) * 0.66)

################### split into train and test #########################################
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
history2 = [x for x in train]
predictions = list()
predictions2 = list()


####################### make rolling forecast by training the model everytime for each prediction ############################### 
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model2 = ARIMA(history2, order=(5,1,0))
	model_fit = model.fit(disp=0)
	model_fit_2 = model2.fit(disp=0)

	output = model_fit.forecast()
	output2 = model_fit_2.forecast()

	yhat = output[0]
	yhat2 = output2[0]

	predictions.append(yhat)
	predictions2.append(yhat2)

	obs = test[t]
	history.append(obs)
	history2.append(obs)

	print('predicted=%f, expected=%f' % (yhat, obs))

###################### calculate MSE ###################################################
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.plot(predictions2, color='black')
pyplot.show()