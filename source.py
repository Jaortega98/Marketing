import sqlite3
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ipywidgets import widgets, interactive_output, HTML
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from surprise import Reader, Dataset
from surprise.model_selection import  cross_validate, GridSearchCV
from surprise import KNNWithMeans, KNNBasic, KNNWithZScore, KNNBaseline


class Exploration:

    """Para la exploracion y procesamiento de la base de datos"""

    def __init__(self, path_db="db_movies"):

        """init

        Parametros
        -----------
        path_db : str, opcional 
            ruta a la base de datos

        """

        self.db = path_db  


    def create_df(self, query, dates=False):

        """Devuelve un data frame de una consulta SQL

        Parametros
        -----------
        query : str
            Consulta tipo SQL
        dates : bool, opcional
            Hacer o no parse de fechas

        Returns
        --------
            df : DataFrame
              Resultado de la consulta en formato pd.DataFrame  
        
        """

        # Conectar a la base de datps
        conn=sqlite3.connect(self.db)

        # Condicional para hacer parser de las fechas
        if dates:
            # Leer la consulta
            df = pd.read_sql(query, conn, parse_dates="timestamp")
        else:
            df = pd.read_sql(query, conn)

        # Cerrar la conexion
        conn.close()

        return df
    

    def preprocess(self, path="preprocessing/preprocess.sql"):

        """Preprocesa las bases de datos con las consultas especificadas

        Parametros
        -----------
            path : str, default=preprocessing/preprocess.sql
               ruta al archivo .sql con las consultas de preprocesamiento 
        
        """

        # Conectar a la base de datos
        conn = sqlite3.connect(self.db)

        # Cursor para las realizar las consultas
        cur = conn.cursor()

        # Abrir archivo con consultas
        with open(path, "r") as f:

            # Ejecutar consultas
            cur.executescript(f.read())
        
        # Consulta para modificar tablas creadas con pandas
        query = "SELECT * FROM full_info"

        # Crear Data Frame
        df = self.create_df(query, dates=True)

        # Extraer años de lanzamiento de las peliculas en una nueva columna
        df["released_yr"] = df["title"].str.extract("\((\d{4})\)").dropna()

        # Modificar la tabla en la base de datos
        df.to_sql("full_info", conn, if_exists="replace", index=False)

        # Consulta para eliminar valores duplicados de las bases existentes
        query_2 = "SELECT * FROM movies"

        # Crear Data Frame
        df = self.create_df(query_2)

        # Listas con los indices de las peliculas duplicadas en la base de datos
        dup = df[df.duplicated(subset="title", keep=False)].index.tolist()
        dup_2 = df[df.duplicated(subset="title", keep=False)].index.tolist()

        # Lista vacia para los indices a eliminar
        ind_drop = []   

        # Ciclo para selecionar indices a eliminar
        for i in dup:
            dup_2.remove(i)
            for j in dup_2:
                if df.loc[i, "title"] == df.loc[j, "title"]:
                    # Escoger peliculas con menos generos disponibles para ser eliminadas
                    if len(df.loc[i, "genres"]) < len(df.loc[j, "genres"]):
                        ind_drop.append(i)
                    else:
                        ind_drop.append(j)
                
        # Eliminar indices de los valores duplicados
        df.drop(ind_drop, inplace=True)

        # Modificar tabla en la base de datos
        df.to_sql("movies", conn, if_exists="replace", index=False)

        # Cerrar la conexion con la base de datos
        conn.close()


    def distributions(self):
        
        """Realiza la representación gráfica de las distribuciones estadísticas tanto de la frecuencia de visualización de las películas en la base de datos como de la cantidad de películas vistas por usuario

        """
        # Consulta con la frecuencia de visualizacion de las peliculas
        query_1 = "SELECT userId, COUNT(*) AS watched_movies FROM ratings GROUP BY userId ORDER BY watched_movies DESC"
        
        # Consulta con la cantidad de peliculas vistas por usuario
        query_2 = "SELECT movieId, COUNT(*) AS total_ratings FROM ratings GROUP BY movieId ORDER BY total_ratings"
        df_user = self.create_df(query_1)
        df_mov = self.create_df(query_2)

        # Subplots para poner las dos graficas de distribucion
        fig = make_subplots(rows=1, cols=2, column_titles=("<b>Distribucion Cantidad de peliculas vistas<br>por usuario</b>",
                                                        "<b>Distribucion Cantidad de calificaciones<br>por pelicula</b>"))
        
        # Grafico de distribucion de la cantidad de peliculas vistas por usuario
        fig.add_trace(go.Histogram(x=df_user["watched_movies"], nbinsx=60,
                                hovertemplate="<b>Peliculas vistas</b>=%{x}<br><b>Usuarios</b>=%{y}<extra></extra>", 
                                marker={"color":"rgba(41, 223, 137, 0.4)"}), col=1, row=1)

        # Grafico de distribucion de la frecuencia de visualizacion de las peliculas
        fig.add_trace(go.Histogram(x=df_mov["total_ratings"], nbinsx=50,
                                hovertemplate="<b>Veces Calificada</b>=%{x}<br><b>Peliculas</b>=%{y}<extra></extra>",
                                marker={"color":"rgba(109, 47, 79, 0.8)"}), col=2, row=1)

        # Ajustes del layout
        fig.update_layout(plot_bgcolor="white", yaxis={"title":"<b>Usuarios</b>"}, xaxis={"title":"<b>Peliculas vistas</b>"}, 
                        yaxis2={"title":"<b>Peliculas</b>"}, xaxis2={"title":"<b>Veces Calificada</b>"}, legend={"visible":False})
        
        # Mostrar figura
        fig.show()

    
    def pieplot(self):

        """Realiza la representación gráfica de la proporcion de peliculas vistas de 0-9 veces y mas de 10 veces en la base de datos

        """
        # Consulta con la cantidad de peliculas vistas mas de 9 veces
        query_1 = "SELECT COUNT(*) AS 'mas de nueve' FROM (SELECT movieId, COUNT(*) AS t_rated FROM ratings GROUP BY movieId HAVING t_rated > 9)"

        # Consulta con la cantidad de peliculas vistas menos de 9 veces
        query_2 = "SELECT COUNT(*) AS 'cero a nueve' FROM (SELECT movieId, COUNT(*) AS t_rated FROM ratings GROUP BY movieId HAVING t_rated BETWEEN 1 AND 9)"

        # Creacion del Data Frame a graficar
        df = pd.melt(pd.concat([self.create_df(query_1), self.create_df(query_2)], axis=1), var_name="veces vista", value_name="peliculas")
        
        # Creacion de la figura
        fig = px.pie(df, names="veces vista", values="peliculas", width=800, height=430, title="<b>Proporcion  de peliculas<br>por veces vista</b>",
                    color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.7, hover_data={"veces vista":False})
        
        # Ajustes del layout
        fig.update_layout(title={"x":0.5})

        # Ajuste de detalles de la grafica
        fig.update_traces(marker={"line":{"color":"black", "width":1.25}})

        # Mostrar Figura
        fig.show()


class SystemRecomendation:

    """"Para la creacion de las recomendaciones y sus filtros"""

    def __init__(self, cdf=Exploration().create_df, path_db="db_movies"):

        """init

        Parametros
        -----------
        cdf : own function, opcional
            Funcion para crear los Data Frames de las consultas
        path_db : str, opcional 
            ruta a la base de datos

        """
        self.create_df = cdf
        self.db = path_db


    def filter_popularity(self, type, n, yr, m_ten):

        """Muestra la peliculas de mayor popularidad entre los usuarios

        Parametros
        -----------
            type : str
                Tipo de filtro que se va a aplicar. Puede ser: Mejor Calificada, Mas Vista o Mejor Calificada x Año de lanzamiento
            n : int
                Cantidad de sugerencias a mostrar
            yr : int
                Año a filtrar las peliculas, solo funciona cuando type = Mejor Calificada x Año de lanzamiento
            m_ten : bool, opcional
                Si se quieren ver solo peliculas con mas de diez visualizaciones

        """
        # Filtro para el tipo de consulta
        if type == "Mejor Calificada":

            # Filtro para peliculas con mas de 10 visualizaciones
            if m_ten:
                # Cosulta con peliculas de mejores calificaciones
                query = f'SELECT movieId, title AS Titulo, ROUND(AVG(rating), 1) AS "Promedio calificacion", COUNT(*) AS "Veces vista" FROM full_info GROUP BY movieId HAVING "Veces vista" > 9 ORDER BY "Promedio calificacion" DESC LIMIT {n}'
                # Crear Data Frame con la consulta
                df = self.create_df(query)
                # Mostrar Data Frame
                display(df.iloc[:, 1:])
            else:
                # Cosulta con peliculas de mejores calificaciones 
                query = f'SELECT movieId, title AS Titulo, ROUND(AVG(rating), 1) AS "Promedio calificacion", COUNT(*) AS "Veces vista" FROM full_info GROUP BY movieId ORDER BY "Promedio calificacion" DESC LIMIT {n}'
                # Crear Data Frame con la consulta
                df = self.create_df(query)
                # Mostrar Data Frame
                display(df.iloc[:, 1:])

        elif type == "Mas Vista":
            
            # Cosulta con peliculas mas vistas por los usuarios
            query = f'SELECT movieId, title AS Titulo, ROUND(AVG(rating), 1) AS "Promedio calificacion", COUNT(*) AS "Veces vista" FROM full_info GROUP BY movieId ORDER BY "Veces vista" DESC LIMIT {n}'
            # Crear Data Frame con la consulta
            df = self.create_df(query)
            # Mostrar Data Frame
            display(df.iloc[:, 1:])
        
        elif type == "Mejor Calificada x Año de lanzamiento":

            # Filtro para peliculas con mas de 10 visualizaciones
            if m_ten:
                # Cosulta con peliculas de mejores calificaciones 
                query = f'SELECT released_yr AS "Año de Lanzamiento", title AS Titulo, ROUND(AVG(rating), 1) AS "Promedio calificacion", COUNT(*) "Veces vista" FROM full_info GROUP BY "Año de Lanzamiento", Titulo HAVING "Año de Lanzamiento" = {yr}  AND "Veces vista" > 9 ORDER BY "Año de Lanzamiento" DESC, "Promedio calificacion" DESC'
                # Crear Data Frame con la consulta
                df = self.create_df(query)
                # Mostrar Data Frame
                display(df.iloc[:n])           
            else:
                # Cosulta con peliculas de mejores calificaciones 
                query = f'SELECT released_yr AS "Año de Lanzamiento", title AS Titulo, ROUND(AVG(rating), 1) AS "Promedio calificacion", COUNT(*) "Veces vista" FROM full_info GROUP BY "Año de Lanzamiento", Titulo HAVING "Año de Lanzamiento" = {yr} ORDER BY "Año de Lanzamiento" DESC, "Promedio calificacion" DESC'
                # Crear Data Frame con la consulta
                df = self.create_df(query)
                # Mostrar Data Frame
                display(df.iloc[:n])


    def popularity_recommendations(self, max_titles=60):
        
        """Crea widgets interactivos para filtrar las peliculas por popularidad

        Parametros
        -----------
            max_titles: int, default=60
                Cantidad maxima de sugerencias a mostrar

        """

        # Widget para el tipo de filtro a utilizar (type)
        wid_1 = widgets.Dropdown(options=["Mejor Calificada", "Mas Vista", "Mejor Calificada x Año de lanzamiento"])
        # Widget para la cantidad de valores a mostrar (n)
        wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)
        # Widget para en año a mostrar (yr)
        wid_3 = widgets.Dropdown(options=self.create_df("SELECT DISTINCT released_yr FROM full_info").sort_values(by="released_yr").values.flatten().tolist(),
                                disabled=True)
        # Widget para mostrar peliculas vistas mas de 10 veces (m_ten)
        wid_4 = widgets.Dropdown(options={"Si":True, "No":False}, value=False)

        # Funcion para la activacion y desactivacion de wid_3 (yr) segun los valores en wid_1 (type)
        def update_wid3(change):
            if change.new == "Mejor Calificada x Año de lanzamiento":
                wid_3.disabled = False
            else:
                wid_3.disabled = True

        # Funcion para la activacion y desactivacion de wid_4 (m_ten) segun los valores en wid_1 (type)
        def update_wid4(change):
            if change.new in ["Mejor Calificada", "Mejor Calificada x Año de lanzamiento"]:
                wid_4.disabled = False
            else:
                wid_4.disabled = True

        # Ubicar widgets en columnas verticales
        ui_1 = widgets.VBox([HTML("<b>Tipo de filtrado</b>"), wid_1, HTML("<b>Cantidad de sugerencias</b>"),
                            wid_2, HTML("<b>Año de lanzamiento</b>"), wid_3])
        # Ubicar widgets en columnas verticales
        ui_2 = widgets.VBox([HTML("<b>Vista mas de 10 veces</b>"), wid_4])
        # Ubicar las columnas verticales en una horizontal
        ui_T = widgets.HBox([ui_1, ui_2])

        # Hacer widgets interactivos con la funcion filter_popularity
        output = interactive_output(self.filter_popularity, {"type":wid_1, "n":wid_2, "yr":wid_3, "m_ten":wid_4})

        # Activar o desactivar widgets 4 y 3
        wid_1.observe(update_wid3, names="value")
        wid_1.observe(update_wid4, names="value")

        # Mostrar resultados
        display(ui_T, output)


    def filter_content(self, title, n, fillna_yr=2010):

        """Muestra las peliculas mas correlacionadas segun su contenido y año de lanzamiento 

        Parametros
        -----------
            title : str
                Titulo de la pelicula a observar
            n : int
                Cantidad de sugerencias a mostrar
            fillna_yr : int, default=2010
                Año para completar el valor de la columna de año de lanzamiento de las peliculas que no lo tengan en su titulo

        """
        # Cosulta de todas las peliculas con titulo, Id y genero
        query = "SELECT movieId, title AS Titulo, genres AS Generos FROM movies;"
        # Crear Data Frame con la consulta 
        movies = self.create_df(query)
        # Crear columna con año de lanzamiento
        movies["Año de Lanzamiento"] = movies["Titulo"].str.extract("\((\d{4})\)").fillna(fillna_yr).astype(int)

        # Instnciar escalador para procesar los datos 
        escaler = MinMaxScaler()
        # Index en el Data Frame del titulo elegido
        index = movies[movies["Titulo"] ==  title].index.values.tolist()[0]
        # Data Frame con varibles dummies de los generos de cada pelicula y su año de lanzamiento
        to_corr = pd.concat([movies["Año de Lanzamiento"], movies['Generos'].str.get_dummies()], axis=1)
        # Data Frame con los datos preprocesados
        to_corr = pd.DataFrame(escaler.fit_transform(to_corr), columns=to_corr.columns, index=to_corr.index)

        # Columna con el nivel de similitud de las peliculas con el titulo seleccionado
        movies["Similitud"] = to_corr.corrwith(to_corr.loc[index], axis=1)

        # Data Frame a mostrar, organizado de acuerdo al nivel de similitud
        df =  movies[movies["Titulo"] != title].sort_values(by="Similitud", ascending=False).iloc[:n,1:]
        # Convertir formato de columna a porcentaje
        df["Similitud"] = df["Similitud"].apply('{:.2%}'.format)
        
        # Mostrar Data Frame
        display(df)


    def content_recommendations(self, max_titles=60):

        """Crea widgets interactivos para filtrar las peliculas por similitud de contenido

        Parametros
        -----------
            max_titles: int, default=60
                Cantidad maxima de sugerencias a mostrar

        """
        # Widget para el titulo de la pelicula (title)
        wid_1 = widgets.Dropdown(options=self.create_df("SELECT DISTINCT title FROM movies ORDER BY title").values.flatten().tolist())
        # Widget para la cantidad de valores a mostrar (n)
        wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)
        
        # Ubicar widgets en una columna vertical
        ui_1 = widgets.VBox([HTML("<b>Titulo de la pelicula</b>"), wid_1, HTML("<b>Cantidad de sugerencias</b>"),wid_2])

        # Hacer widgets interactivos con la funcion filter_content
        output = interactive_output(self.filter_content, {"title":wid_1, "n":wid_2})

        # Mostrar resultados
        display(ui_1, output)


    def filter_knn(self, title, n_neighbors=10, fillna_yr=2010):

        """Muestra las peliculas mas similares en su contenido y año de lanzamiento usando el algoritmo KNN

        Parametros
        -----------
            title : str
                Titulo de la pelicula a observar
            n_neighbors : int
                Cantidad de sugerencias a encontrar con el algoritmo
            fillna_yr : int, default=2010
                Año para completar el valor de la columna de año de lanzamiento de las peliculas que no lo tengan en su titulo

        """
        # Cosulta de todas las peliculas con titulo, Id y genero
        query = "SELECT movieId, title AS Titulo, genres AS Generos FROM movies"
        # Crear Data Frame con la consulta 
        movies = self.create_df(query)
        # Crear columna con año de lanzamiento
        movies["Año de Lanzamiento"] = movies["Titulo"].str.extract("\((\d{4})\)").fillna(fillna_yr).astype(int)
        # Data Frame con varibles dummies de los generos de cada pelicula y su año de lanzamiento
        to_corr = pd.concat([movies["Año de Lanzamiento"], movies['Generos'].str.get_dummies()], axis=1)

        # Pipeline con el procesador de datos y el algoritmo KNN
        model = make_pipeline(MinMaxScaler(),
                                NearestNeighbors(n_neighbors=n_neighbors, metric="cosine"))

        # Entrenar modelo
        model.fit(to_corr)

        # Arrays con las distacias mas cercanas y los indices las peliculas correspondientes a estas en el Data Frame para cada pelicula 
        dis, id_dis = model.named_steps["nearestneighbors"].kneighbors(to_corr)
        
        # Data Frame con las distancias
        dis = pd.DataFrame(dis)
        # Data Frame con los indices de cada distancia
        id_dis = pd.DataFrame(id_dis)
        
        # Index en el Data Frame de peliculas del titulo seleccionado
        movies_id = movies[movies['Titulo'] == title].index[0]
        # Indices de las peliculas con distancias mas cercanas al titulo escogido
        inx = id_dis.loc[movies_id]
        
        # Mostrar Data Frame 
        display(movies.iloc[inx,1:])


    def knn_recommendations(self, max_titles=60):

        """Crea widgets interactivos para filtrar las peliculas por similitud de contenido usando KNN

        Parametros
        -----------
            max_titles: int, default=60
                Cantidad maxima de sugerencias a mostrar

        """
        # Widget para el titulo de la pelicula (title)
        wid_1 = widgets.Dropdown(options=self.create_df("SELECT DISTINCT title FROM movies ORDER BY title").values.flatten().tolist())
        # Widget para la cantidad de valores a mostrar (n_neighbors)
        wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)

        # Ubicar widgets en una columna vertical
        ui_1 = widgets.VBox([HTML("<b>Titulo de la pelicula</b>"), wid_1, HTML("<b>Cantidad de sugerencias</b>"),wid_2])

        # Hacer widgets interactivos con la funcion filter_knn
        output = interactive_output(self.filter_knn, {"title":wid_1, "n_neighbors":wid_2})

        # Mostrar resultados
        display(ui_1, output)

    
    def filter_user_knn(self, user, n_neighbors, fillna_yr=2010):

        """Muestra las peliculas mas similares segun el contenido visto por un usuario

        Parametros
        -----------
            user : int
                Id del usuario
            n_neighbors : int
                Cantidad de sugerencias a encontrar con el algoritmo
            fillna_yr : int, default=2010
                Año para completar el valor de la columna de año de lanzamiento de las peliculas que no lo tengan en su titulo

        """
        # Cosulta de todas las peliculas con titulo, Id y genero
        query = "SELECT movieId, title AS Titulo, genres AS Generos FROM movies"
        # Cosulta de todos los Id's de las peliculas vistas por el usuario escogido
        query_s = f"SELECT DISTINCT movieId FROM full_info WHERE userId = {user}"
        # Crear Data Frame de peliculas
        movies = self.create_df(query)
        # Crear columna con año de lanzamiento
        movies["Año de Lanzamiento"] = movies["Titulo"].str.extract("\((\d{4})\)").fillna(fillna_yr).astype(int)
        # Data Frame con varibles dummies de los generos de cada pelicula y su año de lanzamiento
        to_corr = pd.concat([movies["Año de Lanzamiento"], movies['Generos'].str.get_dummies()], axis=1)
        # Crear Data Frame con Id's de peliculas vistas por el usuario
        seen_movies = self.create_df(query_s)
        
        # Convertir Id's en array
        mid_seen = seen_movies.values.flatten()
        # Obtener index de las peliculas vistas por el usuario en el Data Frame de peliculas
        ind_seen = movies[movies["movieId"].isin(mid_seen)].index
        # Obtener index de las peliculas no vistas por el usuario en el Data Frame de peliculas
        ind_unseen = movies[~movies["movieId"].isin(ind_seen)].index
        # Data Frame con promedio de las caracteristicas de las peliculas vistas por el usuario
        centroid = to_corr.loc[ind_seen].mean().to_frame().T
        
        # Pipeline con el procesador de datos y el algoritmo KNN
        model = make_pipeline(MinMaxScaler(),
                            NearestNeighbors(n_neighbors=n_neighbors, metric="cosine"))
        
        # Entrenar modelo con Data Frame de las pelicualas no vistas
        model.fit(to_corr.loc[ind_unseen])
        
        # Arrays con las distacias mas cercanas al vector de caracteristicas promedio y los indices de las peliculas correspondientes a estas en el Data Frame
        dis, id_dis = model.named_steps["nearestneighbors"].kneighbors(centroid)
        
        # Data Frame con las distancias
        dis = pd.DataFrame(dis)
        # Data Frame con los indices de cada distancia
        id_dis = pd.DataFrame(id_dis)

        # Mostrar Data Frame con peliculas mas cercanas
        display(movies.iloc[id_dis.values.flatten(), 1:])


    def knn_user_recommendations(self, max_titles=60):

        """Crea widgets interactivos para filtrar las peliculas mas recomendadas a cada usuario segun el contenido visto por este

        Parametros
        -----------
            max_titles: int, default=60
                Cantidad maxima de sugerencias a mostrar

        """
        # Widget para el Id del usuario (user)
        wid_1 = widgets.Dropdown(options=self.create_df("SELECT DISTINCT userId FROM ratings").values.flatten().tolist())
        # Widget para la cantidad de valores a mostrar (n)
        wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)

        # Ubicar widgets en una columna vertical
        ui_1 = widgets.VBox([HTML("<b>Id del usuario</b>"), wid_1, HTML("<b>Cantidad de sugerencias</b>"),wid_2])

        # Hacer widgets interactivos con la funcion filter_user_knn
        output = interactive_output(self.filter_user_knn, {"user":wid_1, "n_neighbors":wid_2})

        # Mostrar resultados
        display(ui_1, output)


    def create_pred(self, cv=5, cv_grid=3, path="preprocessing/preprocess_pred.sql"):

        """Crea predicciones de calificacion a las peliculas no vistas por el usuario teniendo en cuanta la calificacion dada a las peliculas vistas

        Parametros
        -----------
            cv : int, default=5
               Numero de folds para el cross validation 
            cv_grid : int, default=3
               Numero de folds para el cross validation del entrenamiento del modelo seleccionado 
            path : str, default=preprocessing/preprocess_pred.sql
               Ruta al archivo .sql con las consultas de creacion de la tabla de predicciones

        """
        # Cosulta de todos los ratings registrados por los usuarios
        query = 'SELECT * FROM ratings'
        # Crear Data Frame de calificaciones
        df = self.create_df(query)
        
        # Valor Minimo de calificaciones
        min = df["rating"].min()
        # Valor Maximo de calificaciones
        max = df["rating"].max()
        
        # Lector de datos para la libreria surprise
        reader = Reader(rating_scale=(min, max))
        # Leer datos en formato aceptado por surprise
        data = Dataset.load_from_df(df.iloc[:, :-1], reader)
        # Lista con posibles modelos a aplicar
        models = [KNNWithMeans(), KNNBasic(), KNNWithZScore(), KNNBaseline()]
        
        # Data Frame vacio para ubicar los resultados de los diferentes modelos
        model_results = pd.DataFrame()
        
        # Ciclo para entrenar los diferentes modelos
        for model in models:

            # Cross validation con 'MAE' y 'RMSE'
            CVscores = cross_validate(model, data, cv=cv, measures=["MAE", "RMSE"], n_jobs=-1)
            # Rellenar Data Frame con resultados, index corresponde al nombre de cada algoritmo
            df = pd.DataFrame(CVscores, index=[model.__class__.__name__]*cv).iloc[:, :2].rename(columns={"test_mae":"MAE", "test_rmse":"RSME"})
            # Concatenar los resultados de cada modelo
            model_results = pd.concat([model_results, df])
        
        # Obtener el nombre del mejor modelo segun RSME
        best_rsme = model_results.sort_values(by="RSME", ascending=False).index[0]
        
        # Escoger el mejor modelo
        model_chose = [i for i in models if i.__class__.__name__ == best_rsme][0].__class__
        
        # Hiperparametros para entrenar el modelo escogido
        param_grid = {"sim_options":{"name":["msd", "cosine"],
                                    "min_support":[7],
                                    "user_based":[False, True]}}
        
        # Cross valitadion para afinar hiperparametros del modelo escogido 
        grid_search = GridSearchCV(model_chose, param_grid=param_grid, cv=cv_grid, measures=["rmse"], n_jobs=-1)
        
        # Entrenar el modelo
        grid_search.fit(data)

        # Obtener el modelo con los mejores hipeparametros
        best_model = grid_search.best_estimator["rmse"]
        
        # Crear datos de entrenamiento del algoritmo
        train = data.build_full_trainset()
        # Crear datos de evaluacion del algoritmo
        test = train.build_anti_testset()
        
        # Entrenar el modelo
        best_model.fit(train)
        
        # Crear array de predicciones
        predictions = best_model.test(test)
        # Convertir a Data Frame
        pred_df = pd.DataFrame(predictions)
        
        # Conectar a la base de datos
        conn = sqlite3.connect(self.db)
        # Cursor para realizar consultas 
        cur = conn.cursor()
        
        # Enviar enviar la tabla de predicciones con las columnas de interes a la base de datos
        pred_df.iloc[:, :-1].to_sql("predictions", conn, if_exists="replace", index=False)

        # Abrir el archivo con las consultas
        with open(path, "r") as f:
            # Ejecutar consultas
            cur.executescript(f.read())
        
        # Cerrar la conexion a la base de datos
        conn.close()


    def filter_colab(self, user, n):

        """Muestra las peliculas con la prediccion de calificacion mas alta segun el contenido visto por un usuario

        Parametros
        -----------
            user : int
                Id del usuario
            n : int
                Cantidad de sugerencias a mostrar

        """
        # Cosulta de todos las predicciones de calificacion mas alta para cada usuario
        query = f'SELECT title AS Titulo, genres AS Generos, CAST(ROUND((est/(SELECT MAX(rating) FROM ratings))*100, 1) AS TEXT) AS Estimacion FROM suggestions WHERE userId = {user} ORDER BY Estimacion DESC LIMIT {n}'
        # Crear Data Frame de predicciones
        df = self.create_df(query)

        # Convertir estimacion a tipo float
        df["Estimacion"] = (df["Estimacion"].astype(float)/100)
        # Covertir formato de columna a porcentaje
        df["Estimacion"] = df["Estimacion"].apply('{:.1%}'.format)

        # Mostrar Data Frame con predicciones mas altas
        display(df)  


    def colab_recommendations(self, max_titles=60):

        """Crea widgets interactivos para filtrar las peliculas mas recomendadas a cada usuario segun la prediccion de calificacion obtenida mediante el contenido visto por este

        Parametros
        -----------
            max_titles: int, default=60
                Cantidad maxima de sugerencias a mostrar

        """

        # Widget para el Id del usuario (user)
        wid_1 = widgets.Dropdown(options=self.create_df("SELECT DISTINCT userId FROM ratings").values.flatten().tolist())
        # Widget para la cantidad de valores a mostrar (n)
        wid_2 = widgets.BoundedIntText(value=10, min=10, max=max_titles)

        # Ubicar widgets en una columna vertical
        ui_1 = widgets.VBox([HTML("<b>Id del usuario</b>"), wid_1, HTML("<b>Cantidad de sugerencias</b>"),wid_2])

        # Hacer widgets interactivos con la funcion filter_colab
        output = interactive_output(self.filter_colab, {"user":wid_1, "n":wid_2})

        # Mostrar resultados
        display(ui_1, output)


    def filter_system(self, type_sr):

        """Despliega el tipo de sistema de recomendacion escogido

            Parametros
            -----------
                type_sr: str
                    Tipo de systema de recomendacion a desplegar. Puede tomar los valores de: Basado en Popularidad, Basado en Contenido, Basado en Contenido (KNN), Basado en Contenido visto por usuario, Colaborativo

        """

        # Condicional para filtrar el tipo de sistema escogido
        if type_sr == "Basado en Popularidad":
            # Agregar titulo
            display(HTML('<br><h1 style="text-align: center; font-size:27px;">Sistema de recomendacion basado en popularidad</h1><br>'))
            # Llamar funcion
            self.popularity_recommendations(150)
            
        elif type_sr == "Basado en Contenido":
            # Agregar titulo
            display(HTML('<br><h1 style="text-align: center; font-size:27px;">Sistema de recomendacion basado en contenido</h1><br>'))
            # Llamar funcion
            self.content_recommendations(150)
        
        elif type_sr == "Basado en Contenido (KNN)":
            # Agregar titulo
            display(HTML('<br><h1 style="text-align: center; font-size:27px;">Sistema de recomendacion basado en contenido (KNN)</h1><br>'))
            # Llamar funcion
            self.knn_recommendations(150)
        
        elif type_sr == "Basado en Contenido visto por usuario":
            # Agregar titulo
            display(HTML('<br><h1 style="text-align: center; font-size:27px;">Sistema de recomendacion basado en contenido visto por el usuario</h1><br>'))
            # Llamar funcion
            self.knn_user_recommendations(150)
        
        elif type_sr == "Colaborativo":
            # Agregar titulo
            display(HTML('<br><h1 style="text-align: center; font-size:27px;">Sistema de recomendacion Colaborativo</h1><br>'))
            # Llamar funcion
            self.colab_recommendations(150)


    def display_system(self):

        """Crea un widgets interactivos para filtrar los tipos de sistemas de recomendacion y realizar predicciones para los filtros colaborativos

        """
        # Widget para el tipo de sistema de recomendacion
        wid_1 = widgets.Dropdown(options=["Basado en Popularidad", "Basado en Contenido", "Basado en Contenido (KNN)", 
                                        "Basado en Contenido visto por usuario", "Filtros Colaborativos"], value=None)
        # Widget para ejecutar la funcion y realizar las predicciones para los filtros colaborativos
        wid_2 = widgets.Button(description="Realizar Predicciones")

        # Funcion auxiliar
        def make_pred(wid_2):
            # Ejecutar funcion de predicciones
            self.create_pred()

        # Relacionar funcion al widget 
        wid_2.on_click(make_pred)

        # Ubicar widgets en una columna vertical
        ui_1 = widgets.VBox([HTML("<h2>Sistema de recomendacion</h2><br>"), wid_1], layout=dict(margin="0px 80px 0px 200px"))
        # Ubicar widgets en una columna vertical
        ui_2 = widgets.VBox([HTML("<h2>Filtros Colaborativos</h2><br>"), wid_2], layout=dict(margin="0px 0px 0px 0px"))

        # Ubicar widgets en columnas horizontales
        ui_T = widgets.HBox([ui_1, ui_2])

        # Hacer widgets interactivos con la funcion filter_system
        output = interactive_output(self.filter_system, {"type_sr":wid_1})

        # Mostrar resultados
        display(ui_T, output)