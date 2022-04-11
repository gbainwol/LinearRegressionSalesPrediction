#Loading the data and importing the necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline


advert = pd.read_csv('Advertising.csv')
advert.head()

#to get info on the shape of our dataframe
advert.info()
#we notice that there is an index column that we want to remove we have to figure out it's exact name
#we use the .columns method to call the columns and once the column names are printed we can copy the column name to put into our function
advert.columns
#use the drop function 
advert.drop(['Unnamed: 0'] , axis= 1, inplace= True)
#displaying the updated data frame, axis is set to 1 for the column(rows are 0
advert.head()

#exploratory data analysis using seaborn distribution plots
import seaborn as sns
sns.distplot(advert.sales)
sns.distplot(advert.newspaper)
sns.distplot(advert.radio)
sns.distplot(advert.TV)

sns.pairplot(advert, x_vars=['TV','radio', 'newspaper'],y_vars='sales', height=7,aspect=0.7,kind='reg')
#notice the strong correlation between the tv sales and adspend 
#weaker correlation for radio
#no correlation between newspaper adspend and sales

advert.TV.corr(advert.sales)
#to view the individual correlation of tv & sales


advert.corr()
#to see the correlation of all columns and sales

#use seaborn heatmap to better see best fit correlation
sns.heatmap(advert.corr(),annot=True)


#Create the simple linear Regression Model
x= advert[['TV']]
x.head()

#checking the type of values in the data and the shape of the dataset we will be passing to the model
print(type(x))
print(x.shape)


y=advert.sales
print(type(y))
print(y.shape)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test =train_test_split(x,y,random_state=1)

#Splitting the data into two categories one to train the model and one to train it's accuracy once trained
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#create the regression model

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(x_train,y_train)

#Use intercept to find Beta0 of linear model and the second value is Beta 1
#Beta1 would be negative if an increase in tv adspend resulted in a decrease in sales 
print(linreg.intercept_)
print(linreg.coef_)
y_pred = linreg.predict(x_test)
y_pred[:5]

#use a evaluation metric designed for comparing continous values 
#evaluation metrics for classification problems such as accuracy arent very useful for regresssion
true =[100,50,30,20]
pred=[90,50,50,30]
#print((10+0+20+10)/4)
#^doing the mean absolute error by hand 
#and also the mean squared error by hand
#print(10**2+0**2+ 20**2+10**2)/4
#for the root mean squared error you need to use the numpy np.sqrt function to take the square root of the mean squared error
#print(np.sqrt((10**2+0**2+ 20**2+10**2)/4))

#below is using sklearn to do it for you
from sklearn import metrics 
print(metrics.mean_absolute_error(true,pred))

print(metrics.mean_squared_error(true,pred))
print(np.sqrt(metrics.mean_squared_error(true,pred)))

#root mean squared error is the most accurate
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))



