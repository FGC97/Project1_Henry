# Libraries 

import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast


app = FastAPI()

df_movies = pd.read_csv('dataset/df_movies.csv', sep = ',')
df_movies_ml = pd.read_csv('dataset/df_movies_ml.csv', sep = ',')


#Presentation:
#We create a simple endpoint as a presentation with our name.


@app.get("/")
def Presentation():
    return {"Fredy González": "Welcome to my first Henry project. To explore the API documentation, please visit https://project1-henry.onrender.com/docs."}

# Endpoint 1

@app.get("/peliculas_idioma/{language}")
def peliculas_idioma(language: str):
    
    """The peliculas_idioma function takes a language (string) as input, validates its format, and returns a dictionary. The dictionary contains either the count of movies in the specified language as the value with the language as the key (formatted with capitalization), or an error message if the language is invalid or no movies are found."""
    
    try:
        if not isinstance(language, str):
            raise TypeError("Please provide a valid language name. It must be a string.")

        # Get the list of unique languages in the DataFrame
        languages_list = list(df_movies['original_language'].unique())

        # Check if the provided language is in the list of languages
        if language in languages_list:
            # Count the number of movies released in the provided language
            movies_count = df_movies[df_movies['original_language'] == language].shape[0]
            return {language: f'{movies_count} movies'}
        else:
            # Raise a ValueError if no movies are found in the provided language
            raise ValueError
    except TypeError as e:
        # Return a dictionary with a TypeError message if an invalid language type is provided
        return {'TypeError': str(e)}
    except ValueError:
        # Return a dictionary with a NotFound message if no movies are found in the provided language
        return {'NotFound': f'No movie released in the language {language} was found'}
    
    
# Endpoint 2

@app.get("/peliculas_duracion/{movie}")    
def peliculas_duracion(movie: str):
    
    """The peliculas_duracion function takes a movie name as input and returns a dictionary. The dictionary contains the movie name as the key (formatted with capitalization) and a list of strings as the value. Each string in the list represents the duration and release year of a movie. If no movie is found, it returns a string message indicating that no movie was found with the given name."""
    
    try:
        if isinstance(movie, str):  # Check if the input is a string
            movie = movie.strip()
            movie = movie.lower()  # Convert the movie name to lowercase
            duplicates = df_movies[df_movies['title'].str.lower() == movie]
            # Select rows where the lowercase movie title matches the lowercase input movie name

            if not duplicates.empty:
                results = []
                for index, row in duplicates.sort_values('release_year').iterrows():
                    duration = row['runtime']  # Get the movie duration
                    year = row['release_year']  # Get the movie release year
                    movie_info = f"Duración: {duration} min, Año: {year}"
                    results.append(movie_info)
                    # Append the formatted string with movie duration and year to the results list
                return {movie.title(): results}  # Return a dictionary with the movie name as key and the list of movie information as value
            else:
                selection = df_movies[df_movies['title'].str.lower() == movie]
                # Select rows where the lowercase movie title matches the lowercase input movie name (without duplicates)
                if not selection.empty:
                    duration = selection['runtime'].iloc[0]  # Get the duration of the first selected movie
                    year = selection['release_year'].iloc[0]  # Get the release year of the first selected movie
                    movie_info = f"Duración: {duration} min, Año: {year}"
                    return {movie.title(): [movie_info]}
                    # Return a dictionary with the movie name as key and a list with the movie information as value
            return f"No '{movie}' movie found, please provide a valid movie name as a string."
            # Return a message indicating that no movie was found with the given name
        else:
            return f"Invalid input: '{movie}', please provide a valid movie name as a string."
    except ValueError:
        return {'NotFound': f'No movie released in the language {movie} was found'}
    
    
    

# Endpoint 3

@app.get("/franquicia/{franquicia}")
def franquicia(franquicia: str):
    
    """The franquicia function takes a franchise name (string) as input, validates its format, and returns a dictionary. The dictionary contains either the information about the specified franchise, including the number of movies, total revenue, and average revenue per movie, or an error message if the franchise is invalid or not found. The franchise name is case-insensitive, and leading/trailing spaces are removed before processing."""
    
    try:
        if not isinstance(franquicia, str):
            raise TypeError("Please provide a valid franchise name. It must be a string!")

        # Remove leading and trailing spaces from the franchise name and Convert the franchise name to lowercase
        franchise = franquicia.strip().lower()

        # selection = df_movies[df_movies['belongs_to_collection'].str.lower().str.contains(franchise, na=False)]  # El contains hace que falle la respuesta

        selection = df_movies[df_movies['belongs_to_collection'].str.lower().str.fullmatch(franchise.lower(), na=False)]
        # Select rows where the 'belongs_to_collection' column fullmatch the franchise name (case-insensitive)

        if not selection.empty:
            movies = selection.shape[0]  # Get the number of movies in the franchise
            total_revenue = selection['revenue'].sum()  # Calculate the total revenue of the movies
            average_revenue = selection['revenue'].mean()  # Calculate the average revenue per movie

            return {franchise.title(): {'Number of movies': movies, 'Total revenue': total_revenue, 'Average revenue': average_revenue}}
            # Return the formatted string with franchise information
        else:
            return {'Error': {'No franchise found': franchise}}
            # Return a message indicating that no franchise was found with the given name
    except TypeError as e:
        return {'Error': {'TypeError': str(e)}}
    except ValueError:
        return {'No franchise found': {franchise: 'Franchise not found'}}
    
# Endpoint 4

@app.get("/peliculas_pais/{pais}")
def peliculas_pais(pais: str):
    
    """The peliculas_pais function takes a country name (string) as input, validates its format, and returns a dictionary. The dictionary contains either the count of movies produced in the specified country or an appropriate error message if the country name is invalid or no movies are found. The country name is case-insensitive, and leading/trailing spaces are removed before processing."""
    
    try:
        if not isinstance(pais, str):
            raise TypeError("please provide a valid country name. It must be a string!")

        # Remove leading and trailing spaces from the input country name and convert it to lowercase
        country = pais.strip().title()

        # Filter the DataFrame to get rows where 'production_countries' column contains the country
        matching_movies = df_movies[df_movies['production_countries'].str.contains(country, case=False, na=False)]

        # Count the number of matching movies
        countries_count = matching_movies.shape[0]

        if countries_count > 0:
            # Return a dictionary with the movie count and country name
            return {country: f"{countries_count} movie{'s' if countries_count > 1 else ''}"}
        else:
            # If no movies were found, return a message indicating that
            return {'No movies were produced in': country}
    except TypeError as e:
        return {'Error': f"Invalid input: {pais}, {str(e)}"}
    except Exception as e:
        return {'Error': f"Error retrieving movie data for {country}: {str(e)}"}
    
    

# Endpoint 5

@app.get("/productoras_exitosas/{productora}")
def productoras_exitosas(productora: str):
    
    """The productoras_exitosas function takes a production company name (string) as input, validates its format, and returns a dictionary. The dictionary contains information about the specified production company, including the total revenue generated by its movies and the number of movies it has produced. If no successful production company is found with the given name, an error message is returned. The production company name is case-insensitive, and leading/trailing spaces are removed before processing."""
    
    try:
        # Check if the input is a valid string
        if not isinstance(productora, str):
            raise TypeError("Please provide a valid production company name. It must be a string.")

        # Remove leading and trailing spaces from the input production company name
        productora = productora.strip()

        # Convert the production company name to lowercase
        productora = productora.lower()

        # Filter the DataFrame to get rows where 'production_companies' column contains the lowercase production company name
        matching_movies = df_movies[df_movies['production_companies'].str.lower().str.contains(productora, na=False)]

        if not matching_movies.empty:
            # Calculate the total revenue and number of movies for the matching production company
            total_revenue = matching_movies['revenue'].sum()
            movies_count = matching_movies.shape[0]

            # Return a dictionary with the production company, total revenue, and number of movies
            return {
                'Production Company': productora.title(),
                'Total Revenue': total_revenue,
                'Number of movies': movies_count
            }
        else:
            # If no matching production company is found, return an error message
            return {'Error': f"No successful production company found: '{productora.title()}'"}

    except TypeError as e:
        # Return an error message for invalid input type
        return {'Error': f"Invalid input: '{productora}', {str(e)}"}

    except Exception as e:
        # Return an error message for other exceptions
        return {'Error': f"No successful production company found: '{productora}', {str(e)}"}
    


# Endpoint 6

@app.get("/get_director/{director_name}")
def get_director(director_name: str):
    
    """The get_director function takes a director name (string) as input, validates its format, and returns a dictionary. The dictionary contains information about the specified director, including their success based on the total revenue generated by their movies, the total number of movies they have directed, and the details of each movie."""
    
    try:
        # Check if a string is provided as input
        if not isinstance(director_name, str):
            raise ValueError("TypeError: Please provide a valid director name. It must be a string!")

        # Normalize the input director name
        director_name = director_name.strip().title()

        # Filter the DataFrame to select rows where the director name matches
        matching_directors = df_movies[df_movies["director"].str.title().apply(lambda directors: director_name in directors)]

        # Check if the director was found in the dataset
        if matching_directors.empty:
            return {"Error": f"Director '{director_name}' not found"}

        # Calculate the total number of movies for the director
        total_movies = len(matching_directors)

        # Calculate the success of the director based on the average return
        director_success = matching_directors['revenue'].sum()

        # Create a list of dictionaries containing movie details
        director_movies = []
        for index, row in matching_directors.iterrows():
            movie_info = {
                'Title': row['title'],
                'Release date': row['release_date'],
                'Return': row['return'],
                'Budget': row['budget'],
                'Revenue': row['revenue']
            }
            director_movies.append(movie_info)

        # Return a dictionary with the director's success, total movies, and movie details
        return {"Director": director_name, "Success (Revenue)": director_success, "Total Movies": total_movies, "Movies": director_movies}

    except ValueError as e:
        return str(e)

# Endpoint 7

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    
    """The recomendacion function takes a movie title as input and returns a dictionary containing recommendations for similar movies. The function first verifies if the input title is a string and normalizes it. It then checks if the title exists in the movie dataset and retrieves the matching movies. For each matching movie, the function filters movies with the same genres and calculates the TF-IDF similarity between the input movie and the filtered movies based on their title and overview. It selects the top 5 most similar movies and creates a dictionary of recommendations, including their genres and vote average. Finally, the function returns a dictionary containing recommendations for each matching movie based on the input title."""
    
    # Verify if the title is a string
    if not isinstance(titulo, str):
        return {"Error": f"'{titulo}' is not a string"}

    # Normalize the title
    titulo = titulo.strip().title()

    # Check if the title is present in the DataFrame
    matching_movies = df_movies_ml[df_movies_ml['title'].str.title() == titulo]
    if matching_movies.empty:
        return {"Error": f"The movie {titulo} was not found"}

    recommendations_dict = {}

    # Iterate over the matching movies
    for _, movie in matching_movies.iterrows():
        movie_title = movie['title']
        movie_anio = movie['release_year']
        movie_id = movie['id']
        movie_genre = ast.literal_eval(movie["genres"])

        # Check if the movie has at least one genre
        if len(movie_genre) == 0:
            return {"Error": f"No recommendations found for the movie {titulo}"}

        # Filter movies with the same genres as the current movie
        filtered_movies = df_movies_ml[df_movies_ml['genres'].apply(lambda x: len(ast.literal_eval(x)) == len(movie_genre) and ast.literal_eval(x) == movie_genre)]

        # Reset the index of the filtered DataFrame
        filtered_movies = filtered_movies.reset_index(drop=True)

        # Create a TF-IDF vectorizer for movie features (title and overview)
        tfidf = TfidfVectorizer(stop_words='english')

        # Combine the features (title and overview) into a single field
        filtered_movies['combined_features'] = filtered_movies['title'] + ' ' + filtered_movies['overview'].fillna('')

        # Calculate the TF-IDF matrix of the features
        tfidf_matrix = tfidf.fit_transform(filtered_movies['combined_features'])

        # Get the index corresponding to the movie ID of the input movie
        movie_index = filtered_movies[filtered_movies['id'] == movie_id].index[0]

        # Calculate the cosine similarity between the input movie and the filtered movies
        similarities = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()

        # Get the indices of the top 5 most similar movies (excluding the input movie)
        top_indices = similarities.argsort()[-6:-1][::-1]

        recommendations = {}
        # Create a dictionary of recommendations for the current movie
        for idx in top_indices:
            rec_movie_title = filtered_movies.iloc[idx]['title']
            rec_movie_genres = filtered_movies.iloc[idx]['genres']
            rec_movie_vote_average = filtered_movies.iloc[idx]['vote_average']
            recommendations[rec_movie_title] = {'genres': rec_movie_genres, 'vote_average': rec_movie_vote_average}

        # Add the recommendations for the current movie to the final recommendations dictionary
        recommendations_dict[f"{movie_title}, year {movie_anio}"] = recommendations

    # Return the final dictionary containing recommendations for each movie matching the input title
    return recommendations_dict
