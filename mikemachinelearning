import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns",None)
data =pd.DataFrame({
'Temperature': [60, 70, 80, 90, 100],
'Fuel_Price': [3.5, 3.7, 3.6, 3.8, 4.0],
'CPI': [220, 221, 222, 223, 224],
'Unemployment': [5.0, 5.2, 5.1, 5.3, 5.4],
'Weekly_Sales': [20000, 21000, 22000, 23000, 24000]
})
print(data)
y=data['Weekly_Sales']
x=data[['Temperature','Fuel_Price','CPI','Unemployment']]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=47,test_size=0.4)
print(x_train)
model=LinearRegression()
model.fit(x_train,y_train)
y_predicted=model.predict(x_test)
print(y_predicted)
print(mean_squared_error(y_test,y_predicted))
r_score=r2_score(y_test,y_predicted)
print(r_score)
new_df=pd.DataFrame(x_test)
new_df['actual']=y_test.values
new_df['predicted']=y_predicted
new_df['mean_sqrd_error']=(new_df['actual']-new_df['predicted'])**2
new_df['r2_error']=new_df['actual']-new_df['predicted']
print(new_df)
