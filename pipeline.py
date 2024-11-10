import pandas as pd
import numpy as np
import gensim
from gensim import models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input,Concatenate

def carga_inf():
    datos = pd.read_pickle("ensayos_processed_docs.pkl")
    datos_features = pd.read_pickle("ensayos_features.pkl")
    embeddings = np.load("ensayos_embeddings.npy")
    doc_embedding = pd.read_csv("ensayos_embedding_fasttext.csv")
    return datos, embeddings, doc_embedding, datos_features

def calc_topicos(datos, indict):
    topicos = []
    for y in range(datos.shape[0]):
        if len(indict[y]) > 0:
            valid_sublist = [sublist for sublist in indict[y] if len(sublist) > 1]
            if len(valid_sublist) > 0:
                max_index = np.argmax([sublist[1] for sublist in valid_sublist])
                topicos.append(valid_sublist[max_index][0])
            else:
                topicos.append(None)
        else:
            topicos.append(None)

    return topicos

def bw_corpus(datos):
    procc_docs = datos["processed_docs"]
    dictionary = gensim.corpora.Dictionary(procc_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=500)
    bow_corpus = [dictionary.doc2bow(doc) for doc in procc_docs]
    return dictionary, bow_corpus

def modelo_lda(bow_corpus, dictionary):
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
    ind_without_tfidf = lda_model[bow_corpus]    
    return ind_without_tfidf

def modelo_tfidf(bow_corpus, dictionary):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=2, workers=4)
    ind_with_tfidf = lda_model_tfidf[bow_corpus]
    return ind_with_tfidf

def modelo_fast(doc_embedding):
    pca = PCA(n_components=30, random_state=0)
    pcs = pca.fit_transform(doc_embedding.values)
    scaler = StandardScaler()
    scaled_pcs = scaler.fit_transform(pcs)

    km = KMeans(n_clusters=8, random_state=0)
    km.fit_transform(scaled_pcs)
    cluster_labels = km.labels_
    return cluster_labels

def resultados(datos, embeddings, doc_embedding):
    data = datos.get(["essay_id","essay_set"])
    dictionary, bow_corpus = bw_corpus(datos)

    ind_without_tfidf = modelo_lda(bow_corpus, dictionary)
    ind_with_tfidf = modelo_tfidf(bow_corpus, dictionary)

    kmeans = KMeans(n_clusters=8, random_state=0).fit(embeddings)
    km = modelo_fast(doc_embedding)

    data["topic_wo"] = calc_topicos(data, ind_without_tfidf)
    data["topic_tfidf"] = calc_topicos(data, ind_with_tfidf)
    data['bert'] = kmeans.labels_
    data["FAST"] = km
    return data

def proc_resultado(data):
    data["topic_wo"] = data["topic_wo"] + 1
    data["topic_tfidf"] = data["topic_tfidf"] + 1
    data["bert"] = data["bert"] + 1
    data["FAST"] = data["FAST"] + 1

    matriz_wo = pd.crosstab(data["essay_set"],data["topic_wo"])
    matriz_tfidf = pd.crosstab(data["essay_set"],data["topic_tfidf"])
    matriz_bert = pd.crosstab(data["essay_set"],data["bert"])
    matriz_fast = pd.crosstab(data["essay_set"],data["FAST"])

    tablas_cruzadas = [matriz_wo, matriz_tfidf, matriz_bert, matriz_fast]
    resultados_totales = []
    
    for matriz in tablas_cruzadas:
        categorias_usadas = set()
        asignaciones = {} 
        
        for essay_set, row in matriz.iterrows():
            categorias_ordenadas = row.sort_values(ascending=False).index
            
            for categoria in categorias_ordenadas:
                if categoria not in categorias_usadas:
                    asignaciones[essay_set] = categoria
                    categorias_usadas.add(categoria)
                    break
        resultado = pd.DataFrame(list(asignaciones.items()), columns=['essay_set', 'Categoria'])
        resultados_totales.append(resultado)

    resultado = pd.merge(resultados_totales[0], resultados_totales[1], on = "essay_set", suffixes=('_wo', '_tfidf'))
    resultado = pd.merge(resultado, resultados_totales[2], on = "essay_set")
    resultado = pd.merge(resultado, resultados_totales[3], on = "essay_set", suffixes=('_bert', '_FAST'))

    listado = [1,2,3,4,5,6,7,8]
    asignacion_wo = dict(zip(resultado["Categoria_wo"].values, listado))
    asignacion_tfidf = dict(zip(resultado["Categoria_tfidf"].values, listado))
    asignacion_bert = dict(zip(resultado["Categoria_bert"].values, listado))
    asignacion_FAST = dict(zip(resultado["Categoria_FAST"].values, listado))

    data_copia = data.copy()
    data_copia["topic_wo"] = data_copia["topic_wo"].replace(asignacion_wo)
    data_copia["topic_tfidf"] = data_copia["topic_tfidf"].replace(asignacion_tfidf)
    data_copia["bert"] = data_copia["bert"].replace(asignacion_bert)
    data_copia["FAST"] = data_copia["FAST"].replace(asignacion_FAST)
    return data_copia

def valores_roc(data):
    data_copia = proc_resultado(data)
    data_copia1 = data_copia.copy()
    modelos = ("topic_wo","topic_tfidf","bert","FAST")
    temas = data_copia["essay_set"].unique()
    y_true = []
    y_test = []
    for tema in temas:
      data_copia1.loc[data_copia["essay_set"]==tema, "essay_set"] = 1
      data_copia1.loc[data_copia["essay_set"]!=tema, "essay_set"] = 0
      
      y_true.append(list(data_copia1["essay_set"]))
      yest = []
      for modelo in modelos:
        data_copia1.loc[data_copia[modelo]==tema, modelo] = 1
        data_copia1.loc[data_copia[modelo]!=tema, modelo] = 0
        
        yest.append(list(data_copia1[modelo]))
      y_test.append(yest)

    valores_ROC = pd.DataFrame({"TEMAS":["Tema 1","Tema 2","Tema 3","Tema 4","Tema 5","Tema 6","Tema 7","Tema 8"]})
    for modelo in range(len(y_test[0])):
        roc = []
        for tema in range(len(y_test)):
            roc.append(roc_auc_score(y_true[tema], y_test[tema][modelo]).round(2))
        valores_ROC[f"MODELO {modelo}"] = roc

    valores_ROC = valores_ROC.rename(columns = {'MODELO 0':'SIN TF-IDF', 'MODELO 1':'CON TF-IDF',
                                                'MODELO 2':'MODELO BERT','MODELO 3':'FAST TEXT'})
    return valores_ROC

def acc_global(data):
    data_copia = proc_resultado(data)
    matriz1_wo = pd.crosstab(data_copia["essay_set"],data_copia["topic_wo"]).to_numpy()
    matriz1_tfidf = pd.crosstab(data_copia["essay_set"],data_copia["topic_tfidf"]).to_numpy()
    matriz1_bert = pd.crosstab(data_copia["essay_set"],data_copia["bert"]).to_numpy()
    matriz1_fast = pd.crosstab(data_copia["essay_set"],data_copia["FAST"]).to_numpy()

    acc_wo = np.diag(matriz1_wo).sum() * 100 / np.sum(matriz1_wo)
    acc_tfidf = np.diag(matriz1_tfidf).sum() * 100 / np.sum(matriz1_tfidf)
    acc_bert = np.diag(matriz1_bert).sum() * 100 / np.sum(matriz1_bert)
    acc_fast = np.diag(matriz1_fast).sum() * 100 / np.sum(matriz1_fast)

    accuracy_global = pd.DataFrame({"MODELO": ["SIN TF-IDF","CON TF-IDF","MODELO BERT","FAST TEXT"],
                                    "ACCURACY":[acc_wo.round(2), acc_tfidf.round(2),
                                                acc_bert.round(2), acc_fast.round(2)]})
    return accuracy_global

def datos_red(datos_features, embedding):
    datos_features["essay_set"] = datos_features["essay_set"] - 1

    X_provisional = datos_features.get(["essay_id","token_count"]) 
    y_provisional = datos_features.get(["essay_set"]).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_provisional,y_provisional,test_size = 0.10,stratify = y_provisional)
    index_train = X_train.index
    index_test = X_test.index

    X = datos_features.drop(columns=['essay_id', 'essay_set'])
    y = datos_features["essay_set"]

    features_train = X.iloc[index_train,]
    embeddings_train = embedding.iloc[index_train,]
    y_train = y.iloc[index_train,] 
    
    features_test = X.iloc[index_test,]
    embeddings_test = embedding.iloc[index_test,]
    y_test = y.iloc[index_test,]

    scaler = MinMaxScaler((-1.0,1.0))
    features_train_scaled = pd.DataFrame(scaler.fit_transform(features_train))
    features_train_scaled.columns = features_train.columns
    features_test_scaled = pd.DataFrame(scaler.fit_transform(features_test))
    features_test_scaled.columns = features_test.columns
    return features_train_scaled, features_test_scaled, embeddings_train, y_train, embeddings_test, y_test

def FCNN(datos, embedding):
    features_train_scaled, features_test_scaled, embeddings_train, y_train, embeddings_test, y_test = datos_red(datos, embedding)
    clases = len(y_train.unique())
    x1 = Input(shape=(embeddings_train.shape[1],), name='Input_Embedding')
    x2 = Input(shape=(features_train_scaled.shape[1],), name='Input_Features')
    x = Concatenate(name='Concatenar')([x1, x2])
    x = Dense(64, activation='elu', name='Capa_Densa_1')(x)
    x = Dense(32, activation='elu', name='Capa_Densa_2')(x)
    x = Dense(16, activation='elu', name='Capa_Densa_3')(x)
    x = Dense(clases, activation='softmax', name='Output')(x)
    model = Model(inputs=[x1, x2], outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x = [embeddings_train,features_train_scaled], y = y_train,
                        validation_data = ([embeddings_test,features_test_scaled], y_test),
                        epochs=100,batch_size=32,verbose=1)
    return history

def final():
    datos, embeddings, doc_embedding, datos_features = carga_inf()
    data = resultados(datos, embeddings, doc_embedding)
    accuracy_global = acc_global(data)
    valores_ROC = valores_roc(data)

    mejor_modelo = accuracy_global["MODELO"][np.argmax(accuracy_global["ACCURACY"])]
    mejor_acc = accuracy_global["ACCURACY"][np.argmax(accuracy_global["ACCURACY"])]

    if mejor_modelo == "FAST TEXT":
        embedding = doc_embedding
    elif mejor_modelo == "MODELO BERT":
        embedding = pd.DataFrame(embeddings)
    
    history = FCNN(datos_features, embedding)
    return history, mejor_modelo, mejor_acc, valores_ROC
