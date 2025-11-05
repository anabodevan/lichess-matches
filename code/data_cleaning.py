import pandas as pd
import numpy as np

df = pd.read_csv('data/games.csv')
df.head(1)
df.shape
df.info()

print(df.isnull().sum())

df.duplicated().sum()
df.drop_duplicates(inplace=True)

df[df['turns'] <= 2]['turns'].sum()
df = df[df['turns'] > 2]

df.shape
df = df[['turns', 'victory_status', 'winner', 'white_rating', 'black_rating', 'opening_name', 'opening_ply', 'increment_code']]
df.head()

df['opening_name'].unique()
df['opening_name'].str.split('|', expand=True).head(1)
df.loc['opening_name'] = df['opening_name'].str.split('|', n=1).str[0]

df.to_csv('data/games_clean.csv', index=False)