import sqlite3
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ipywidgets as ipyw               
from ipywidgets import widgets, interactive_output, interact, interactive, fixed, widget
from IPython.display import display

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

    df["rel_yr"] = df["title"].str.extract("\((\d{4})\)")

    df.to_sql("full_info", conn, if_exists="replace", index=False)

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


def filter_popularity(type, n, yr):

    if type == "Mejor Calificada":
    
        query = f"SELECT movieId, title, ROUND(AVG(rating), 3) AS avg_ratings, COUNT(*) AS total_ratings FROM full_info GROUP BY movieId ORDER BY avg_ratings DESC LIMIT {n}"
    
        df = create_df(query)
    
        display(df)
    
    elif type == "Mas Vista":
        
        query = f"SELECT movieId, title, ROUND(AVG(rating), 3) AS avg_ratings, COUNT(*) AS total_ratings FROM full_info GROUP BY movieId ORDER BY total_ratings DESC LIMIT {n}"
    
        df = create_df(query)
    
        display(df)
    
    elif type == "Mejor Calificada x Año de lanzamiento":
    
        query = f"SELECT rel_yr, title, ROUND(AVG(rating), 3) AS avg_ratings, COUNT(*) total_ratings FROM full_info GROUP BY rel_yr, title HAVING rel_yr = {yr} ORDER BY rel_yr DESC, avg_ratings DESC"
    
        df = create_df(query)
    
        display(df)


def popularity_recommendations():

    wid_1 = widgets.Dropdown(options=["Mejor Calificada", "Mas Vista", "Mejor Calificada x Año de lanzamiento"], 
                         description="Tipo de Filtrado")
    wid_2 = widgets.BoundedIntText(value=10, min=10, max=60, 
                                description="Numero de recomendaciones")
    wid_3 = widgets.Dropdown(options=create_df("SELECT DISTINCT rel_yr FROM full_info").sort_values(by="rel_yr").values.flatten().tolist(), 
                            description="Año")

    ui_1 = widgets.VBox([wid_1, wid_2, wid_3])

    output = interactive_output(filter_popularity, {"type":wid_1, "n":wid_2, "yr":wid_3})

    display(ui_1, output)
