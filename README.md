<img
style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 100%;
           height=400px"
    src="src/henry.PNG" 
    alt="MLOps">


<h1 style="text-align: center;">Henry Project Nº1 Data Science</h1>
<h1 style="text-align: center;">Machine Learning Operations (MLOps)</h1>
<h1 style="text-align: center;">Fredy Gonzalez</h1>

<img
style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 500px;
           height=400px"
    src="src/mlopsf.png" 
    alt="MLOps">


# Introduction and context

### Welcome to the first individual project of the labs stage of the Henry Bootcamp! In this occasion, I am going to perform as an MLOps Engineer.  

### I was tasked with developing an API using the **FastAPI** framework to make the company's movie datasets available. I was asked to create a minimum viable product ***(MVP)*** that included 6 functions for the API endpoints and another function for a recommendation system using machine learning.

<img
style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 900px;
           height=700px"
    src="src/MLOps_Process.png" 
    alt="MLOps">
<figcaption>
MLOps process.
</figcaption>






1. **`presentation()`**: This function serves as a simple endpoint to present the owner's name.
2. **`peliculas_idioma(language: str)`**: This function takes a language as input and returns the number of movies produced in that language.
3. **`peliculas_duracion(movie: str)`**: This function takes a movie name as input and returns the duration and release year of the movie.
4. **`franquicia(franquicia: str)`**: This function takes a franchise name as input and returns information about the franchise, such as the number of movies, total revenue, and average revenue.
5. **`peliculas_pais(pais: str)`**: This function takes a country name as input and returns the number of movies produced in that country.
6. **`productoras_exitosas(productora: str)`**: This function takes a production company name as input and returns information about the success of the production company, such as total revenue and the number of movies.
7. **`get_director(director_name: str)`**: This function takes a director name as input and returns information about the director's success, total number of movies, and details of each movie.