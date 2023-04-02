import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

teams = [
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
    'Delhi Capitals',
    'Chennai Super Kings',
    'Gujarat Titans',
    'Lucknow Super Giants',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Mumbai Indians'
]

cities = ['Hyderabad', 'Sharjah', 'Navi Mumbai', 'Mumbai', 'Kimberley',
          'Centurion', 'Jaipur', 'Bangalore', 'Chandigarh', 'Kolkata',
          'Bengaluru', 'Chennai', 'Ahmedabad', 'Abu Dhabi', 'Durban',
          'Indore', 'Visakhapatnam', 'Pune', 'Raipur', 'Cuttack', 'Delhi',
          'Dubai', 'Port Elizabeth', 'Johannesburg', 'East London',
          'Cape Town', 'Bloemfontein', 'Nagpur', 'Dharamsala', 'Ranchi']

# pipe = pickle.load(open('pipe.pkl', 'rb'))
final_df = pickle.load(open('final_df.pkl', 'rb'))
############################
X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'), ['BattingTeam', 'BowlingTeam', 'City'])],
    remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))])

pipe.fit(X_train, y_train)
pipe.predict_proba(X_test)
############################

# st.title('IPL Win Predictor')
st.markdown("<h1 style='text-align: center; color: red;'>IPL Win Predictor</h1>", unsafe_allow_html=True)

st.write("---")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

ani = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_26bG2B.json")

st_lottie(ani, height=200)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

st.write("---")

col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3:
    center_button = st.button('Predict Probability')

if center_button:
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'BattingTeam': [batting_team], 'BowlingTeam': [bowling_team], 'City': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets],
                             'target': [target], 'current_run_rate': [crr], 'required_run_rate': [rrr]})
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.markdown("<h5 style='text-align: left; color: red;'>Chances of winning: </h5>", unsafe_allow_html=True)

    st.header(batting_team + " - " + str(round(win * 100)) + "%")
    st.header(bowling_team + " - " + str(round(loss * 100)) + "%")
st.write("---")

