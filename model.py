import numpy as np
import pandas as pd


ratings = pd.read_csv('https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv')
movies = pd.read_csv('https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv')

n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

#Visualizing Datasets

"""print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")

print(f"Mean global rating: {round(ratings['rating'].mean(),2)}.")
mean_ratings = ratings.groupby('userId')['rating'].mean()
print(f"Mean rating per user: {round(mean_ratings.mean(),2)}.")

movie_ratings = ratings.merge(movies, on='movieId')
print(movie_ratings['title'].value_counts()[0:10])


movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
from collections import Counter

genre_frequency = Counter(g for genres in movies['genres'] for g in genres)

print(f"There are {len(genre_frequency)} genres.")

genre_frequency
print("The 5 most common genres: \n", genre_frequency.most_common(5))"""

#The Butter - Data Preprocessing

from scipy.sparse import csr_matrix

def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe containing 3 columns (userId, movieId, rating)
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))
    
    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

#print(X.shape)

#Matrix Sparsity Calculation
"""
n_total = X.shape[0]*X.shape[1]
n_ratings = X.nnz
sparsity = n_ratings/n_total
print(f"Matrix sparsity: {round(sparsity*100,2)}%")
"""

#Item-item recommendations / k-Nearest Neighbors

from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    
    Output: returns list of k similar movie ID's
    """
    X = X.T
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    # use k+1 since kNN output includes the movieId of interest
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

similar_movies = find_similar_movies(1, X, movie_mapper, movie_inv_mapper, k=10)

movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 1

similar_movies = find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
movie_title = movie_titles[movie_id]

#Getting the movie titles
"""
print(f"Because you watched {movie_title}:")
for i in similar_movies:
    print(movie_titles[i])
"""

#Fixing the cold start problem 

n_movies = movies['movieId'].nunique()
#print(f"There are {n_movies} unique movies in our movies dataset.")
genres = set(g for G  in movies['genres'] for g in G)

for g in genres:
    movies[g] = movies.genres.transform(lambda x: int(g in x))
    
movie_genres = movies.drop(columns=['movieId', 'title','genres'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(movie_genres, movie_genres)
#print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")

from fuzzywuzzy import process

#function that allows us to get the index of a movie based on title
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

title = movie_finder('juminji')
title
movie_idx = dict(zip(movies['title'], list(movies.index)))
idx = movie_idx[title]

#print(f"Movie index for Jumanji: {idx}")

n_recommendations=10
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:(n_recommendations+1)]
#print(sim_scores)

similar_movies = [i[0] for i in sim_scores]
"""
print(f"Because you watched {title}:")
print(movies['title'].iloc[similar_movies])
"""
import re

def parse_movie_list(movie_series_list):
    movies = []
    # Extract the Series from the list
    if movie_series_list:
        movie_series = movie_series_list[0]
        # Convert Series to list
        movie_list = movie_series.tolist()
        for movie in movie_list:
            match = re.match(r"^(.+?)\s\((\d{4})\)$", movie)
            if match:
                name, year = match.groups()
                movies.append({"name": name, "year": int(year)})
    return movies



def get_content_based_recommendations(title_string, n_recommendations=5):
    recommended_movies = []
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    #print(f"Because you watched {title}:")
    #print(movies['title'].iloc[similar_movies])
    recommended_movies.append(movies['title'].iloc[similar_movies])
    return recommended_movies

parsed_movies = parse_movie_list(get_content_based_recommendations('toy story', 5))
#print(parsed_movies)
#print(get_content_based_recommendations('toy story', 5))




#Advanced compression technique
"""
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=20, n_iter=10)
Q = svd.fit_transform(X.T)
Q.shape
movie_id = 1
similar_movies = find_similar_movies(movie_id, Q.T, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}:")
for i in similar_movies:
    print(movie_titles[i])
"""

#Getting poster image for movie
import requests

def get_movie_details(movie_name, year):
    TMDB_API_KEY = '7563749e71a07ba0a2bdf8a95d8c1b7b'  # Replace with your actual TMDb API key
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}&year={year}"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            return {
                'title': movie['title'],
                'year': movie['release_date'].split('-')[0],
                'overview': movie['overview'],
                'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie['poster_path'] else None
            }
    return None




