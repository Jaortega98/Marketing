-- Se eliminan usuarios con mas de 1000 peliculas vistas

DELETE FROM ratings
WHERE userId IN (SELECT userId FROM (SELECT userId, COUNT(*) AS watched_movies FROM ratings
GROUP BY userId
HAVING watched_movies > 1000));




-- Tabla con toda la informacion cruzada

DROP TABLE IF EXISTS full_info;

CREATE TABLE full_info AS SELECT userId, ratings.movieId, title, genres, rating, timestamp FROM ratings
LEFT JOIN movies
ON ratings.movieId = movies.movieId;