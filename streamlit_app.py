import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from model import get_content_based_recommendations, parse_movie_list, get_movie_details

"""
# Movie Recommender System based on Machine Learning

"""

movies = pickle.load(open('artifacts/movies.pkl', 'rb'))
movie_list = movies['title'].values
selection = st.selectbox(
    'Type or select a movie to get a recommendation',
    movie_list
)
# Add custom CSS for padding
st.markdown(
    """
    <style>
    .movie-container {
        margin-right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button('Show recommendation'):
    movies = parse_movie_list(get_content_based_recommendations(selection, 5))
    st.write("Here are your movie recommendations:")

    # Create a grid layout
    cols = st.columns(5)  # Adjust the number of columns based on your layout preference

    for i, movie in enumerate(movies):
        details = get_movie_details(movie['name'], movie['year'])
        if details:
            with cols[i % 5]:  # Loop through columns
                st.subheader(details['title'])
                if details['poster_path']:
                    st.image(details['poster_path'], width=150)
                else:
                    st.write("No poster available")
                
                # Add horizontal spacing between movies
                st.write("")  # Or use st.markdown("<br>", unsafe_allow_html=True) for a blank line
        else:
            with cols[i % 5]:
                st.write(f"Details not found for {movie['name']} ({movie['year']})")