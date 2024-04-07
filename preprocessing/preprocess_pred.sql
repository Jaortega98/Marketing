DROP TABLE IF EXISTS suggestions;
    
CREATE TABLE suggestions AS SELECT uid AS userId, title, genres, est FROM predictions t1
JOIN movies t2
ON iid = movieId;
