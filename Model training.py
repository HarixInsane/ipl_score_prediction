import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle

data=pd.read_csv('ipl_data.csv')
data=data[data['overs']>= 6.0]
for c in ['bat_team','bowl_team']:
  data[c]= LabelEncoder().fit_transform(data[c])


c=['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
              'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
              'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
              'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
              'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
              'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
       'runs_last_5', 'wickets_last_5', 'total']
df=pd.DataFrame(data,columns=c)

x=df.drop(['total'],axis=1)
y=df['total']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=200,min_samples_split=2, min_samples_leaf= 1, max_features='log2', max_depth= 30)
model.fit(x_train,y_train)

predtrain=model.predict(x_train)
predtest=model.predict(x_test)

maetrain=mean_absolute_error(y_train,predtrain)
maetest=mean_absolute_error(y_test,predtest)

rmsetrain=mean_squared_error(y_train,predtrain,squared=False)
rmsetest=mean_squared_error(y_test,predtest,squared=False)

r2train=r2_score(y_train,predtrain)
r2test=r2_score(y_test,predtest)

print('Train data: rmse= ',rmsetrain,'mae= ',maetrain,'r2= ',r2train)
print('Test data: rmse= ',rmsetest,'mae= ',maetest,'r2= ',r2test)

import pickle
filename = "ipl_predictor.pkl"
pickle.dump(forest, open(filename, "wb"))
