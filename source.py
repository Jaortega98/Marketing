import sqlite3
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ipywidgets as ipyw       
from ipywidgets import widgets, interactive_output, interact, interactive, fixed, widget
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import KNNWithMeans, KNNBasic, KNNWithZScore, KNNBaseline


def create_df(query, dates=False):

    conn = sqlite3.connect("db_movies")

    if dates:
        df = pd.read_sql(query, conn, parse_dates="timestamp")
    else:
        df = pd.read_sql(query, conn)

    conn.close()

    return df


def preprocess(path="preprocess.sql"):

    conn = sqlite3.connect("db_movies")

    cur = conn.cursor()

    with open(path, "r") as f:
        cur.executescript(f.read())
    
    query = "SELECT * FROM full_info"

    df = create_df(query, dates=True)

    df["released_yr"] = df["title"].str.extract("\((\d{4})\)").dropna()

    df.to_sql("full_info", conn, if_exists="replace", index=False)

    query_2 = "SELECT * FROM movies"

    df = create_df(query_2)

    dup = df[df.duplicated(subset="title", keep=False)].index.tolist()
    dup_2 = df[df.duplicated(subset="title", keep=False)].index.tolist()

    ind_drop = []

    for i in dup:
        dup_2.remove(i)
        for j in dup_2:
            if df.loc[i, "title"] == df.loc[j, "title"]:
                if len(df.loc[i, "genres"]) < len(df.loc[j, "genres"]):
                    ind_drop.append(i)
                else:
                    ind_drop.append(j)
            

    df.drop(ind_drop, inplace=True)

    df.to_sql("movies", conn, if_exists="replace", index=False)

    conn.close()



def distributions():

    query_1 = "SELECT userId, COUNT(*) AS watched_movies FROM ratings GROUP BY userId ORDER BY watched_movies DESC"
    query_2 = "SELECT movieId, COUNT(*) AS total_ratings FROM ratings GROUP BY movieId ORDER BY total_ratings"
    df_user = create_df(query_1)
    df_mov = create_df(query_2)


    fig = make_subplots(rows=1, cols=2, column_titles=("<b>Distribucion Cantidad de peliculas vistas<br>por usuario</b>",
                                                    "<b>Distribucion Cantidad de calificaciones<br>por pelicula</b>"))

    fig.add_trace(go.Histogram(x=df_user["watched_movies"], nbinsx=60,
                            hovertemplate="<b>Peliculas vistas</b>=%{x}<br><b>Usuarios</b>=%{y}<extra></extra>", 
                            marker={"color":"rgba(41, 223, 137, 0.4)"}), col=1, row=1)

    fig.add_trace(go.Histogram(x=df_mov["total_ratings"], nbinsx=50,
                            hovertemplate="<b>Veces Calificada</b>=%{x}<br><b>Peliculas</b>=%{y}<extra></extra>",
                            marker={"color":"rgba(109, 47, 79, 0.8)"}), col=2, row=1)

    fig.update_layout(plot_bgcolor="white", yaxis={"title":"<b>Usuarios</b>"}, xaxis={"title":"<b>Peliculas vistas</b>"}, 
                    yaxis2={"title":"<b>Peliculas</b>"}, xaxis2={"title":"<b>Veces Calificada</b>"}, legend={"visible":False})
    fig.show()


def pieplot():

    query_1 = "SELECT COUNT(*) AS 'mas de nueve' FROM (SELECT movieId, COUNT(*) AS t_rated FROM ratings GROUP BY movieId HAVING t_rated > 9)"
    query_2 = "SELECT COUNT(*) AS 'cero a nueve' FROM (SELECT movieId, COUNT(*) AS t_rated FROM ratings GROUP BY movieId HAVING t_rated BETWEEN 1 AND 9)"
    
    df = pd.melt(pd.concat([create_df(query_1), create_df(query_2)], axis=1), var_name="veces vista", value_name="peliculas")
    
    fig = px.pie(df, names="veces vista", values="peliculas", width=800, height=430, title="<b>Proporcion  de peliculas<br>por veces vista</b>",
                color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.7, hover_data={"veces vista":False})
    
    fig.update_layout(title={"x":0.5})
    fig.update_traces(marker={"line":{"color":"black", "width":1.25}})
    
    fig.show()


def filter_popularity(type, n, yr, m_ten):

    if type == "Mejor Calificada":

        if m_ten:

            query = f'SELECT movieId, title AS Titulo, ROUND(AVG(rating), 3) AS "Promedio calificacion", COUNT(*) AS "Veces vista" FROM full_info GROUP BY movieId HAVING "Veces vista" > 9 ORDER BY "Promedio calificacion" DESC LIMIT {n}'
    
            df = create_df(query)
        
            display(df.iloc[:, 1:])
        else:
    
            query = f'SELECT movieId, title AS Titulo, ROUND(AVG(rating), 3) AS "Promedio calificacion", COUNT(*) AS "Veces vista" FROM full_info GROUP BY movieId ORDER BY "Promedio calificacion" DESC LIMIT {n}'
        
            df = create_df(query)
        
            display(df.iloc[:, 1:])
    
    elif type == "Mas Vista":
        
        query = f'SELECT movieId, title AS Titulo, ROUND(AVG(rating), 3) AS "Promedio calificacion", COUNT(*) AS "Veces vista" FROM full_info GROUP BY movieId ORDER BY "Veces vista" DESC LIMIT {n}'
    
        df = create_df(query)
    
        display(df.iloc[:, 1:])
    
    elif type == "Mejor Calificada x Año de lanzamiento":

        if m_ten:

          query = f'SELECT released_yr AS "Año de Lanzamiento", title AS Titulo, ROUND(AVG(rating), 3) AS "Promedio calificacion", COUNT(*) "Veces vista" FROM full_info GROUP BY "Año de Lanzamiento", Titulo HAVING "Año de Lanzamiento" = {yr}  AND "Veces vista" > 9 ORDER BY "Año de Lanzamiento" DESC, "Promedio calificacion" DESC'

          df = create_df(query)
    
          display(df.iloc[:n])
          
        else:
    
    
          query = f'SELECT released_yr AS "Año de Lanzamiento", title AS Titulo, ROUND(AVG(rating), 3) AS "Promedio calificacion", COUNT(*) "Veces vista" FROM full_info GROUP BY "Año de Lanzamiento", Titulo HAVING "Año de Lanzamiento" = {yr} ORDER BY "Año de Lanzamiento" DESC, "Promedio calificacion" DESC'
      
          df = create_df(query)
      
          display(df.iloc[:n])

def popularity_recommendations(max_titles=60):

    wid_1 = widgets.Dropdown(options=["Mejor Calificada", "Mas Vista", "Mejor Calificada x Año de lanzamiento"])
    wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)
    wid_3 = widgets.Dropdown(options=create_df("SELECT DISTINCT released_yr FROM full_info").sort_values(by="released_yr").values.flatten().tolist(),
                               disabled=True)
    wid_4 = widgets.Dropdown(options={"Si":True, "No":False}, value=False)

    def update_wid3(change):

        if change.new == "Mejor Calificada x Año de lanzamiento":
            wid_3.disabled = False
        else:
            wid_3.disabled = True

    def update_wid4(change):

        if change.new in ["Mejor Calificada", "Mejor Calificada x Año de lanzamiento"]:
            wid_4.disabled = False
        else:
            wid_4.disabled = True


    ui_1 = widgets.VBox([ipyw.HTML("<b>Tipo de filtrado</b>"), wid_1, ipyw.HTML("<b>Cantidad de sugerencias</b>"),
                         wid_2, ipyw.HTML("<b>Año de lanzamiento</b>"), wid_3])

    ui_2 = widgets.VBox([ipyw.HTML("<b>Vista mas de 10 veces</b>"), wid_4])

    ui_T = widgets.HBox([ui_1, ui_2])

    output = interactive_output(filter_popularity, {"type":wid_1, "n":wid_2, "yr":wid_3, "m_ten":wid_4})

    wid_1.observe(update_wid3, names="value")
    wid_1.observe(update_wid4, names="value")

    display(ui_T, output)

def filter_content(title, n):
    
    query = "SELECT movieId, title AS Titulo, genres AS Generos FROM movies;"

    movies = create_df(query)

    movies["Año de Lanzamiento"] = movies["Titulo"].str.extract("\((\d{4})\)").fillna(2010).astype(int)

    escaler = MinMaxScaler()

    index = movies[movies["Titulo"] ==  title].index.values.tolist()[0]

    to_corr = pd.concat([movies["Año de Lanzamiento"], movies['Generos'].str.get_dummies()], axis=1)

    to_corr = pd.DataFrame(escaler.fit_transform(to_corr), columns=to_corr.columns, index=to_corr.index)


    movies["similitud"] = to_corr.corrwith(to_corr.loc[index], axis=1)


    display(movies[movies["Titulo"] != title].sort_values(by="similitud", ascending=False).iloc[:n,1:])

def content_recommendations(max_titles=60):

    wid_1 = widgets.Dropdown(options=create_df("SELECT DISTINCT title FROM movies ORDER BY title").values.flatten().tolist())
    wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)
    

    ui_1 = widgets.VBox([ipyw.HTML("<b>Titulo de la pelicula</b>"), wid_1, ipyw.HTML("<b>Cantidad de sugerencias</b>"),wid_2])

    output = interactive_output(filter_content, {"title":wid_1, "n":wid_2})

    display(ui_1, output)


def filter_knn(title, n_neighbors=10):

      query = "SELECT movieId, title AS Titulo, genres AS Generos FROM movies;"

      movies = create_df(query)

      movies["Año de Lanzamiento"] = movies["Titulo"].str.extract("\((\d{4})\)").fillna(2010).astype(int)
    
      to_corr = pd.concat([movies["Año de Lanzamiento"], movies['Generos'].str.get_dummies()], axis=1)
    
      model = make_pipeline(MinMaxScaler(),
                            NearestNeighbors(n_neighbors=n_neighbors, metric="cosine"))
    
      model.fit(to_corr)
    
    
      dis, id_dis = model.named_steps["nearestneighbors"].kneighbors(to_corr)
    
      dis = pd.DataFrame(dis)
    
      id_list = pd.DataFrame(id_dis)
    
    
    
      movies_id = movies[movies['Titulo'] == title].index[0]
    
    
    
      inx = id_list.loc[movies_id]
    
    
      display(movies.iloc[inx,1:])


def knn_recommendations(max_titles=60):

    wid_1 = widgets.Dropdown(options=create_df("SELECT DISTINCT title FROM movies ORDER BY title").values.flatten().tolist())
    wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)


    ui_1 = widgets.VBox([ipyw.HTML("<b>Titulo de la pelicula</b>"), wid_1, ipyw.HTML("<b>Cantidad de sugerencias</b>"),wid_2])

    output = interactive_output(filter_knn, {"title":wid_1, "n_neighbors":wid_2})

    display(ui_1, output)

def filter_user_knn(user, n_neighbors):

    query = "SELECT movieId, title AS Titulo, genres AS Generos FROM movies;"
    
    query_s = f"SELECT DISTINCT movieId FROM full_info WHERE userId = {user}"
    
    movies = create_df(query)
    
    movies["Año de Lanzamiento"] = movies["Titulo"].str.extract("\((\d{4})\)").fillna(2010).astype(int)
    
    to_corr = pd.concat([movies["Año de Lanzamiento"], movies['Generos'].str.get_dummies()], axis=1)
    
    seen_movies = create_df(query_s)
    
    mid_seen = seen_movies.values.flatten()
    ind_seen = movies[movies["movieId"].isin(mid_seen)].index
    ind_unseen = movies[~movies["movieId"].isin(ind_seen)].index
    
    centroid = to_corr.loc[ind_seen].mean().to_frame().T
    
    
    
    model = make_pipeline(MinMaxScaler(),
                          NearestNeighbors(n_neighbors=n_neighbors, metric="cosine"))
    
    model.fit(to_corr.loc[ind_unseen])
    
    
    dis, id_dis = model.named_steps["nearestneighbors"].kneighbors(centroid)
    
    dis = pd.DataFrame(dis)
    
    id_dis = pd.DataFrame(id_dis)
    
    display(movies.iloc[id_dis.values.flatten(), 1:])

def knn_user_recommendations(max_titles=60):

    wid_1 = widgets.Dropdown(options=create_df("SELECT DISTINCT userId FROM ratings").values.flatten().tolist())
    wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)


    ui_1 = widgets.VBox([ipyw.HTML("<b>Id del usuario</b>"), wid_1, ipyw.HTML("<b>Cantidad de sugerencias</b>"),wid_2])

    output = interactive_output(filter_user_knn, {"user":wid_1, "n_neighbors":wid_2})

    display(ui_1, output)

def create_pred(cv=5, cv_grid=3, path="preprocess_pred.sql"):

    query = 'SELECT * FROM ratings'

    df = create_df(query)
    
    min = df["rating"].min()
    max = df["rating"].max()
    
    reader = Reader(rating_scale=(min, max))
    
    data = Dataset.load_from_df(df.iloc[:, :-1], reader)
    
    models = [KNNWithMeans(), KNNBasic(), KNNWithZScore(), KNNBaseline()]
    
    
    model_results = pd.DataFrame()
    
    for model in models:
    
      CVscores = cross_validate(model, data, cv=cv, measures=["MAE", "RMSE"], n_jobs=-1)
      df = pd.DataFrame(CVscores, index=[model.__class__.__name__]*cv).iloc[:, :2].rename(columns={"test_mae":"MAE", "test_rmse":"RSME"})
      model_results = pd.concat([model_results, df])
    
    best_rsme = model_results.sort_values(by="RSME", ascending=False).index[0]
    
    model_chose = [i for i in models if i.__class__.__name__ == best_rsme][0].__class__
    
    
    
    param_grid = {"sim_options":{"name":["msd", "cosine"],
                                 "min_support":[7],
                                  "user_based":[False, True]}}
    
    grid_search = GridSearchCV(model_chose, param_grid=param_grid, cv=cv_grid, measures=["rmse"], n_jobs=-1)
    
    grid_search.fit(data)
    
    best_model = grid_search.best_estimator["rmse"]
    
    
    train = data.build_full_trainset()
    test = train.build_anti_testset()
    
    best_model.fit(train)
    
    predictions = best_model.test(test)
    pred_df = pd.DataFrame(predictions)
    
    conn = sqlite3.connect("db_movies")
    cur = conn.cursor()
    
    pred_df.iloc[:, :-1].to_sql("predictions", conn, if_exists="replace", index=False)

    with open(path, "r") as f:
        cur.executescript(f.read())
    
    conn.close()


def filter_colab(user_id, n):

    query = f'SELECT title AS Titulo, genres AS Generos, CAST(ROUND((est/(SELECT MAX(rating) FROM ratings))*100, 1) AS TEXT) AS Estimacion FROM suggestions WHERE userId = {user_id} ORDER BY Estimacion DESC LIMIT {n}'
    
    df = create_df(query)

    df["Estimacion"] = (df["Estimacion"].astype(float)/100)

    df["Estimacion"] = df["Estimacion"].apply('{:.1%}'.format)

    display(df)

def colab_recommendations(max_titles=60):

    wid_1 = widgets.Dropdown(options=create_df("SELECT DISTINCT userId FROM ratings").values.flatten().tolist())
    wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)


    ui_1 = widgets.VBox([ipyw.HTML("<b>Id del usuario</b>"), wid_1, ipyw.HTML("<b>Cantidad de sugerencias</b>"),wid_2])

    output = interactive_output(filter_colab, {"user_id":wid_1, "n":wid_2})

    display(ui_1, output)