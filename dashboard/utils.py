import pandas as pd
import arff
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.conf import settings # IMPORTANTE PARA RUTAS
from pandas.plotting import scatter_matrix 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib
# CORRECCIÓN: Evita que Render intente abrir una ventana gráfica (que no existe)
matplotlib.use('Agg')

# --- CLASE AUXILIAR PARA ARCHIVO 09 ---
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.dropna()

# --- CARGA DE DATOS (CORREGIDO PARA PRODUCCIÓN) ---
def load_kdd_data():
    # CORRECCIÓN: Usa la ruta dinámica del proyecto, no una de tu disco local
    path = os.path.join(settings.BASE_DIR, 'KDDTrain+.arff')
    
    if not os.path.exists(path): 
        print(f"Archivo no encontrado en: {path}")
        return None
        
    with open(path, 'r') as f:
        dataset = arff.load(f)
    
    attributes = [attr[0] for attr in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=attributes)
    
    # CORRECCIÓN PARA PLAN FREE (512MB RAM): 
    # Render Free matará el proceso si intentas procesar las 125,000 filas.
    # Limitamos a 2500 para que las gráficas y tablas carguen sin error.
    return df.head(2500)

def get_base64_graph():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    # CORRECCIÓN: Formato estándar para enviar imágenes a HTML sin archivos físicos
    string = base64.b64encode(buf.read()).decode('utf-8')
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    plt.close('all') 
    return uri

# --- ARCHIVO 06: HISTOGRAMAS, CORRELACIÓN Y PROTOCOLOS ---
def generate_visualizations_archivo_06():
    df = load_kdd_data()
    if df is None: return []
    imgs = []
    
    # Histogramas
    df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(10, 8))
    imgs.append(get_base64_graph())
    
    # Matriz correlación
    df_c = df.copy()
    # Convertimos la clase a numérico solo para la correlación
    df_c['class'] = LabelEncoder().fit_transform(df_c['class'].astype(str))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(df_c.corr(numeric_only=True), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    imgs.append(get_base64_graph())
    
    # Scatter Matrix (Muestra pequeña para no colapsar la RAM)
    cols = ['same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate']
    scatter_matrix(df_c[cols].head(100), figsize=(10, 8))
    imgs.append(get_base64_graph())

    # Barras Protocolo
    plt.figure(figsize=(8, 6))
    df['protocol_type'].value_counts().plot(kind='bar', grid=True)
    imgs.append(get_base64_graph())
    
    return imgs

# --- ARCHIVO 07: BARRAS DE PROTOCOLOS ---
def generate_visualizations_archivo_07():
    df = load_kdd_data()
    if df is None: return []
    imgs = []
    counts = df['protocol_type'].value_counts()
    colors = ['blue', 'orange', 'green', 'red']
    
    for i in range(4):
        plt.figure(figsize=(8, 6))
        if i == 1: counts.sort_index(ascending=False).plot(kind='bar', grid=True, color=colors[i])
        elif i == 2: counts.sort_values(ascending=True).plot(kind='bar', grid=True, color=colors[i])
        elif i == 3: counts.reindex(['tcp', 'udp', 'icmp']).plot(kind='bar', grid=True, color=colors[i])
        else: counts.plot(kind='bar', grid=True, color=colors[0])
        imgs.append(get_base64_graph())
    return imgs

# --- ARCHIVO 08: PARTICIONADO Y ESCALADO ---
def generate_data_processing_08():
    df = load_kdd_data()
    if df is None: return {}
    
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df['protocol_type'])
    X_train = train_set.drop("class", axis=1)
    
    X_train_num = X_train.select_dtypes(exclude=['object'])
    imputer = SimpleImputer(strategy="median")
    X_train_clean_all = pd.DataFrame(imputer.fit_transform(X_train_num), columns=X_train_num.columns)
    
    robust_scaler = RobustScaler()
    X_train_scaled_all = pd.DataFrame(robust_scaler.fit_transform(X_train_clean_all), columns=X_train_clean_all.columns)

    X_train_with_nulls = X_train.copy()
    # Muestra segura para nulos
    idx = X_train_with_nulls.sample(min(10, len(X_train_with_nulls))).index
    X_train_with_nulls.loc[idx, "src_bytes"] = np.nan
    filas_nulas = X_train_with_nulls[X_train_with_nulls.isnull().any(axis=1)].head(10)

    return {
        "train_len": len(train_set), 
        "test_len": len(test_set),
        "nulos": filas_nulas.to_html(classes='table table-danger table-sm'),
        "X_train_clean": X_train_clean_all.head(5).to_html(classes='table table-success table-sm'),
        "X_train_scaled": X_train_scaled_all.head(10).to_html(classes='table table-warning table-sm')
    }

# --- ARCHIVO 05: NLP ---
def generate_email_processing_05():
    mail = {
        'subject': ['gener', 'ciali', 'brand', 'qualiti'],
        'body': ['do', 'feel', 'pressur', 'perform', 'rise', 'occas', 'tri', 'viagra', 'anxieti', 'thing', 'past', 'back', 'old', 'self'],
        'content_type': 'multipart/alternative'
    }
    
    prep_email = [" ".join(mail['subject']) + " " + " ".join(mail['body'])]
    vectorizer = CountVectorizer()
    x_cv = vectorizer.fit_transform(prep_email)
    
    tokens = mail['subject'] + mail['body']
    prep_email_oh = [[w] for w in tokens]
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_oh = enc.fit_transform(prep_email_oh)
    
    return {
        "mail_dict": mail,
        "prep_email_text": prep_email,
        "cv_features": vectorizer.get_feature_names_out(),
        "oh_features": enc.get_feature_names_out(),
        "oh_values": X_oh
    }

# --- ARCHIVO 09: PIPELINE AVANZADO ---
def generate_pipeline_processing_09():
    df = load_kdd_data()
    if df is None: return {}
    train_set, _ = train_test_split(df, test_size=0.4, random_state=42)
    X_train = train_set.drop("class", axis=1).head(500).copy()
    
    X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"] < 800), "src_bytes"] = np.nan
    table_1 = X_train[X_train.isnull().any(axis=1)].head(10).to_html(classes='table table-sm table-secondary')

    delete_nan = DeleteNanRows()
    X_train_prep_rows = delete_nan.fit_transform(X_train)
    table_2 = X_train_prep_rows.head(10).to_html(classes='table table-sm table-info')

    num_attribs = list(X_train.select_dtypes(exclude=['object']).columns)
    cat_attribs = ["protocol_type", "service", "flag"]
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs)])
    
    X_train_prep = full_pipeline.fit_transform(X_train)
    column_names = full_pipeline.get_feature_names_out()
    X_train_prep_df = pd.DataFrame(X_train_prep.toarray() if hasattr(X_train_prep, 'toarray') else X_train_prep, columns=column_names, index=X_train.index)
    table_3 = X_train_prep_df.head(10).to_html(classes='table table-sm table-primary')

    return {"table_1": table_1, "table_2": table_2, "table_3": table_3}
