# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df1 = pd.read_csv(r'"C:\Users\ab665\OneDrive\Desktop\Data Analysis\Netflix\netflix_titles.csv"')

# Explore Info
df1.isnull().sum()  # Checking for null values

# Fill missing values
df1['country'].fillna('USA', inplace=True)
df1['director'].fillna('No Director', inplace=True)
df1['cast'].fillna('No Cast', inplace=True)
df1['rating'].fillna('Not Specified', inplace=True)

# Remove rows with missing values
df1 = df1.dropna()

# Data Exploration
sns.countplot(x='type', data=df1)
plt.title('Number of Films and TV Shows')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='rating', data=df1)
plt.title('Distribution of Ratings')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='rating', data=df1, hue='type')
plt.title('Distribution of Ratings by Type')
plt.show()

# Oldest films available on Netflix
oldest_films = df1[df1['type'] == 'Movie'].sort_values("release_year", ascending=True)
oldest_films = oldest_films[oldest_films['duration'] != ""]
oldest_films[['title', "release_year"]][:15]

# Stand-Up Comedy Shows on Netflix
tag = "Stand-Up Comedy"
stand_up_comedy = df1[df1["listed_in"].str.lower().str.contains(tag.lower())]
stand_up_comedy = stand_up_comedy[stand_up_comedy["country"] == "United States"][["title", "country", "release_year"]].head(11)

# Children's TV Shows on Netflix
tag = "Children's TV"
children_tv_shows = df1[df1["listed_in"].str.lower().str.contains(tag.lower())]
children_tv_shows = children_tv_shows[children_tv_shows["country"] == "United States"][["title", "country", "release_year"]].head(11)

# Count of Countries
df1_countries = pd.DataFrame(df1['country'].value_counts().reset_index().values, columns=["country", "count"])
df1_countries.head()

# Count of Release Years
date = pd.DataFrame(df1['release_year'].value_counts().reset_index().values, columns=["Year", "Count"])
date.head()

plt.figure(figsize=(12, 6))
df1[df1["type"] == "Movie"]["release_year"].value_counts()[:20].plot(kind="bar", color="red")
plt.title("Frequency of Movies Released in Different Years on Netflix")
plt.show()

plt.figure(figsize=(12, 6))
df1[df1["type"] == "TV Show"]["release_year"].value_counts()[:20].plot(kind="bar", color="blue")
plt.title("Frequency of TV Shows Released in Different Years on Netflix")
plt.show()

plt.figure(figsize=(12, 6))
df1[df1["type"] == "Movie"]["listed_in"].value_counts()[:11].plot(kind="barh", color="black")
plt.title("Top 11 Categories of Movies")
plt.show()

plt.figure(figsize=(12, 6))
df1[df1["type"] == "TV Show"]["listed_in"].value_counts()[:11].plot(kind="barh", color="brown")
plt.title("Top 11 Categories of TV Shows")
plt.show()

# Create Year Added column based on Date Added
df1['year_added'] = pd.to_datetime(df1['date_added']).dt.year

# Temporary InfoFrames for Plots
netflix_total = df1['year_added'].value_counts().to_frame().reset_index().rename(columns={"index": "year", "year_added": "count"})
netflix_films = df1[df1['type'] == 'Movie']['year_added'].value_counts().to_frame().reset_index().rename(columns={"index": "year", "year_added": "count"})
netflix_tv_shows = df1[df1['type'] == 'TV Show']['year_added'].value_counts().to_frame().reset_index().rename(columns={"index": "year", "year_added": "count"})

fig, ax = plt.subplots(figsize=(13, 7))
plt.title("Frequency of Content Added by Netflix (2018 - 2020)")
plt.xlabel("Year")
plt.ylabel("Number Added")
ax.set_xticks(np.arange(2018, 2022, 1))
sns.set_style("dark")
sns.lineplot(data=netflix_total, x="year", y="count", color="black")
sns.lineplot(data=netflix_films, x="year", y="count", color="red")
sns.lineplot(data=netflix_tv_shows, x="year", y="count", color="blue")
plt.legend(['Total', 'Films', "TV Shows"])
plt.grid()
plt.show()

# Create a temporary InfoFrame for Genres
category = df1.set_index('title')['listed_in'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
category_df = pd.DataFrame()
category_df['genre'] = category
years = df1.set_index('title')['year_added']
description = df1.set_index('title')['description']
merged_data = pd.merge(category_df, years, left_index=True, right_index=True)
temp = pd.merge(merged_data, description, left_index=True, right_index=True)

plt.figure(figsize=(11, 11))
sns.countplot(y='genre', data=temp, order=temp['genre'].value_counts().iloc[:20].index)
plt.title('Top 20 Categories Added by Netflix (2018 - 2020)')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.grid()
plt.show()

# Count of Directors
df1['director'].value_counts()

# Count of Ratings
new_info = df1.groupby('rating').size().rename_axis('Rating').reset_index(name='Count')
new_info = new_info.sort_values(by='Count', ascending=True)
new_info = new_info.tail(5)

# Count of Directors
film_directors = df1['director'].str.split(',', expand=True).stack()
film_directors = pd.DataFrame(film_directors)
film_directors.columns = ['director']
directors = film_directors.groupby(['director']).size().reset_index(name='counts')
directors = directors.sort_values(by='counts', ascending=False)
directors = directors[directors['director'] != 'No Director']
directors = directors.head(5)

# Count of Actors
film_actors = df1['cast'].str.split(',', expand=True).stack()
film_actors = pd.DataFrame(film_actors)
film_actors.columns = ['cast']
actors = film_actors.groupby(['cast']).size().reset_index(name='counts')
actors = actors.sort_values(by='counts', ascending=False)
actors = actors[actors['cast'] != 'No Cast']
actors = actors.head(5)

# Filter and process data for climate-related TV Shows

# Create a DataFrame containing the specified features
features = ['title', 'duration', 'type']
climate_shows = df1[features].copy()  # Create a copy to avoid warnings

# Clean and convert the 'duration' column
climate_shows['no_of_climates'] = climate_shows['duration'].str.replace(' Climate', '').str.replace('s', '')
climate_shows['no_of_climates'] = climate_shows['no_of_climates'].str.extract('(\d+)').astype(float).astype(int)

# Filter the DataFrame for TV Shows
climate_shows = climate_shows[climate_shows['type'] == 'TV Show']

# Drop the 'duration' column
climate_shows = climate_shows.drop('duration', axis=1)

# Sort the DataFrame by 'no_of_climates' in descending order
climate_shows = climate_shows.sort_values('no_of_climates', ascending=False)

# Get the top 5 TV Shows with the most 'no_of_climates'
top_5_climate_shows = climate_shows.head(5)

# Display the result
print(top_5_climate_shows)

