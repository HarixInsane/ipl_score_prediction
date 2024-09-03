import math
import numpy as np
import pickle
import streamlit as st

filename='ipl_predictor.pkl'
model = pickle.load(open(filename,'rb'))

st.markdown("<h1 style='text-align: center; color: black;'> IPL Score Predictor</h1>", unsafe_allow_html=True)


batteam= st.selectbox('Select the Batting Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))

prediction = []
if batteam == 'Chennai Super Kings':
    prediction = prediction + [1,0,0,0,0,0,0,0]
elif batteam == 'Delhi Daredevils':
    prediction = prediction + [0,1,0,0,0,0,0,0]
elif batteam == 'Kings XI Punjab':
    prediction = prediction + [0,0,1,0,0,0,0,0]
elif batteam == 'Kolkata Knight Riders':
    prediction = prediction + [0,0,0,1,0,0,0,0]
elif batteam == 'Mumbai Indians':
    prediction = prediction + [0,0,0,0,1,0,0,0]
elif batteam == 'Rajasthan Royals':
    prediction = prediction + [0,0,0,0,0,1,0,0]
elif batteam == 'Royal Challengers Bangalore':
    prediction = prediction + [0,0,0,0,0,0,1,0]
elif batteam == 'Sunrisers Hyderabad':
    prediction = prediction + [0,0,0,0,0,0,0,1]

bowlteam = st.selectbox('Select the Bowling Team ',('Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab','Kolkata Knight Riders','Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad'))
if bowlteam==batteam:
    st.error('Bowling and Batting teams should be different')
if bowlteam == 'Chennai Super Kings':
    prediction = prediction + [1,0,0,0,0,0,0,0]
elif bowlteam == 'Delhi Daredevils':
    prediction = prediction + [0,1,0,0,0,0,0,0]
elif bowlteam == 'Kings XI Punjab':
    prediction = prediction + [0,0,1,0,0,0,0,0]
elif bowlteam == 'Kolkata Knight Riders':
    prediction = prediction + [0,0,0,1,0,0,0,0]
elif bowlteam == 'Mumbai Indians':
    prediction = prediction + [0,0,0,0,1,0,0,0]
elif bowlteam == 'Rajasthan Royals':
    prediction = prediction + [0,0,0,0,0,1,0,0]
elif bowlteam == 'Royal Challengers Bangalore':
    prediction = prediction + [0,0,0,0,0,0,1,0]
elif bowlteam == 'Sunrisers Hyderabad':
    prediction = prediction + [0,0,0,0,0,0,0,1]
  
col1,col2 = st.columns(2)

with col1:
    overs = st.number_input('Enter the Current Over',min_value=5.1,max_value=19.5,value=5.1,step=0.1)
    if overs-math.floor(overs)>0.6:
        st.error('Please enter valid over input as one over only contains 6 balls')
with col2:
    runs = st.number_input('Enter Current runs',min_value=0,max_value=354,step=1,format='%i')


wickets=st.slider('Enter Wickets fallen till now',0,9)
wickets=int(wickets)

col3, col4 = st.columns(2)

with col3:
    runs_in_prev_5 = st.number_input('Runs scored in the last 5 overs',min_value=0,max_value=runs,step=1,format='%i')

with col4:
    wickets_in_prev_5 = st.number_input('Wickets taken in the last 5 overs',min_value=0,max_value=wickets,step=1,format='%i')

prediction=prediction+[runs, wickets, overs, runs_in_prev_5,wickets_in_prev_5]
prediction=np.array([prediction])
predict=model.predict(prediction)


if st.button('Predict Score'):
    prediction=int(round(predict[0]))
    x=f'PREDICTED MATCH SCORE : {prediction-5} to {prediction+5}' 
    st.success(x)
   
