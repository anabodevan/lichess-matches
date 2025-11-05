# # # Loading Libraries and Data # # # 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('data/games_clean.csv')
df.head()
df.describe()

sns.set_style("whitegrid")

# # # Explanatory Data Analysis # # # 

# # Descriptive Analysis # # 

print(df[['turns', 'white_rating', 'black_rating', 'opening_ply']].describe())

df['rating_difference'] = df['white_rating'] - df['black_rating'] # rating difference 
print(df['rating_difference'].describe())

numerical_cols = ['turns', 'white_rating', 'black_rating', 'opening_ply', 'rating_difference']
print(df[numerical_cols].describe())

# Correlation Matrix of Numerical Features
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(9, 7))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            cbar=True, linewidths=.5, linecolor='black')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()

plt.savefig('output/correlation_matrix_heatmap.png')
plt.close()

# # Bivariate Analysis # # 

plt.figure(figsize=(14, 5))

# Victory status distribution
plt.subplot(1, 2, 1)
sns.countplot(y='victory_status', data=df, order=df['victory_status'].value_counts().index, palette="viridis")
plt.title('Distribution of Victory Status')
plt.xlabel('Count') 

# Winner distribution
plt.subplot(1, 2, 2)
sns.countplot(x='winner', data=df, order=df['winner'].value_counts().index, palette="magma")
plt.title('Distribution of Winner')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('output/victory_winner_distributions.png')
plt.close()

# Rating difference by winner 
plt.figure(figsize=(8, 6))
sns.histplot(data=df[df['winner'] != 'draw'], x='rating_difference', hue='winner', kde=True, bins=50)
plt.title('Distribution of Rating Difference by Winner')
plt.xlabel('Rating Difference (White - Black)')
plt.legend(title='Winner', labels=['Black', 'White'])

plt.savefig('output/rating_difference_by_winner_hist.png')
plt.close()

# Game Lenght by Victory Status 
plt.figure(figsize=(10, 6))
sns.boxplot(x='victory_status', y='turns', data=df, order=df['victory_status'].value_counts().index, palette="Pastel1")
plt.title('Game Length (Turns) by Victory Status')
plt.xlabel('Victory Status')
plt.ylabel('Number of Turns')
plt.ylim(0, 150)

plt.savefig('output/turns_by_victory_status_boxplot.png')
plt.close()

# # Openings Analysis # # 

top_openings = df['opening_name'].value_counts().nlargest(10).index.tolist() # define top 10 openings count
df_top_openings = df[df['opening_name'].isin(top_openings)]

opening_win_counts = df_top_openings.groupby('opening_name')['winner'].value_counts().unstack(fill_value=0)
total_games = opening_win_counts.sum(axis=1)
opening_win_rates = opening_win_counts.divide(total_games, axis=0) * 100
opening_win_rates = opening_win_rates.sort_values(by='white', ascending=False) # calculate openings win rate 

print(opening_win_rates[['white', 'black', 'draw']].round(2))

# Winning rates for top 10 openings 
plt.figure(figsize=(12, 8))
opening_win_rates[['white', 'black', 'draw']].plot(
    kind='barh',
    stacked=True,
    figsize=(12, 8),
    colormap='RdYlBu',
    ax=plt.gca()
)
plt.title('Winning Rates for Top 10 Most Frequent Openings')
plt.xlabel('Percentage (%)')
plt.ylabel('Opening Name')
plt.legend(title='Winner', loc='lower right')
plt.tight_layout()

plt.savefig('output/top_10_openings_win_rates.png')
plt.close()
