import numpy as np
import pandas as pd

matches = pd.read_csv('IPL_Matches_2008_2022.csv')
balls = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')

# from each ball we found the
total_score = balls.groupby(['ID', 'innings']).sum()['total_run'].reset_index()

# taking the total score of 1st inning only
total_score = total_score[total_score['innings'] == 1]

total_score['target'] = total_score['total_run'] + 1

match_df = matches.merge(total_score[['ID', 'target']], on='ID')

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

# combining the same teams which has different names
match_df['Team1'] = match_df['Team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['Team2'] = match_df['Team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['Team1'] = match_df['Team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
match_df['Team2'] = match_df['Team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')

match_df['Team1'] = match_df['Team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['Team2'] = match_df['Team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

# only keeping those values of team which are playing in the team1 or team2 or in winning_team
match_df = match_df[match_df['Team1'].isin(teams)]
match_df = match_df[match_df['Team2'].isin(teams)]
match_df = match_df[match_df['WinningTeam'].isin(teams)]

balls_df = match_df.merge(balls, on='ID')

balls_df = balls_df[balls_df['innings'] == 2]

balls_df['current_score'] = balls_df.groupby('ID')['total_run'].cumsum()

# runs left
balls_df['runs_left'] = np.where(balls_df['target'] - balls_df['current_score'] >= 0,
                                 balls_df['target'] - balls_df['current_score'], 0)

# balls left
balls_df['balls_left'] = np.where(120 - balls_df['overs'] * 6 - balls_df['ballnumber'] >= 0,
                                  120 - balls_df['overs'] * 6 - balls_df['ballnumber'], 0)

# calculating the current run rate
# crr = runs/over
balls_df['current_run_rate'] = (balls_df['current_score'] * 6) / (120 - balls_df['balls_left'])

# calculating the required run rate to win the match
# rrr = runs_left/over_left
balls_df['required_run_rate'] = np.where(balls_df['balls_left'] > 0, balls_df['runs_left'] * 6 / balls_df['balls_left'],
                                         0)


# creating function for converting the winning and losing as 1 and 0
def result(row):
    return 1 if row['BattingTeam'] == row['WinningTeam'] else 0


# 1 as win and 0 as lose
balls_df['result'] = balls_df.apply(result, axis=1)

index1 = balls_df[balls_df['Team2'] == balls_df['BattingTeam']]['Team1'].index
index2 = balls_df[balls_df['Team1'] == balls_df['BattingTeam']]['Team2'].index

balls_df.loc[index1, 'BowlingTeam'] = balls_df.loc[index1, 'Team1']
balls_df.loc[index2, 'BowlingTeam'] = balls_df.loc[index2, 'Team2']

final_df = balls_df[['BattingTeam', 'BowlingTeam', 'City', 'runs_left', 'balls_left', 'wickets_left', 'target', 'current_run_rate', 'required_run_rate', 'result']]

final_df = final_df.sample(final_df.shape[0])

print(final_df)
