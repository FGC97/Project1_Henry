# Libraries 

import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.utils.extmath           import randomized_svd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import linear_kernel


app = FastAPI()



#Presentation:
#We create a simple endpoint as a presentation with our name.

@app.get("/")
def presentation():
    return {'Owner':'Fredy Gonzalez'}

df_movies = pd.read_csv('dataset/df_movies.csv')

# Endpoint 1

@app.get("/peliculas_idioma/{language}")
def peliculas_idioma(language: str):
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

        # Calculate the success of the director based on the average return
        director_success = matching_directors['return'].mean()

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

        # Return a dictionary with the director's success and movie details
        return {"Director": director_name, "Success": director_success, "Movies": director_movies}

    except ValueError as e:
        return str(e)


# Endpoint 7

#@app.get("/recomendacion/{titulo}")
