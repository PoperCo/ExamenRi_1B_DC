"""Motor de recuperación NFCorpus. Archivo independiente para el servicio web."""

import csv
import math
import re
import unicodedata
from collections import Counter
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


# Las negaciones cambian el sentido de consultas biomédicas: no deben eliminarse.
NEGACIONES = {"no", "not", "nor", "never", "without"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGACIONES
STEMMER = SnowballStemmer("english")


def tokenizar(texto):
    """Normaliza, separa palabras, quita stopwords y aplica stemming inglés."""
    tokens = []
    # NFKD conserva letras acentuadas como su letra base antes del filtrado.
    texto_normalizado = unicodedata.normalize("NFKD", texto.casefold())
    caracteres = []

    for caracter in texto_normalizado:
        if not unicodedata.combining(caracter):
            caracteres.append(caracter)

    texto_normalizado = "".join(caracteres)
    palabras = re.findall(r"[a-z0-9]+", texto_normalizado)

    for palabra in palabras:
        if palabra not in STOPWORDS:
            raiz = STEMMER.stem(palabra)
            tokens.append(raiz)

    return tokens

def leer_corpus(ruta):
    """Lee todos los documentos; conserva el identificador original."""
    import pyarrow.parquet as parquet

    tabla = parquet.read_table(ruta, columns=["_id", "title", "text"])
    filas = tabla.to_pylist()
    documentos = []

    for fila in filas:
        documento = {
            "id": str(fila["_id"]),
            "title": fila["title"] or "",
            "text": fila["text"] or "",
        }
        documentos.append(documento)

    return documentos


def leer_consultas(ruta):
    """Lee las consultas; NFCorpus guarda el texto útil en la columna text."""
    import pyarrow.parquet as parquet

    tabla = parquet.read_table(ruta, columns=["_id", "text"])
    consultas = {}

    for fila in tabla.to_pylist():
        consultas[str(fila["_id"])] = fila["text"] or ""

    return consultas


def leer_qrels(ruta):
    """Devuelve relevancias agrupadas por ID de consulta y documento."""
    qrels = {}

    with open(ruta, "r", encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo, delimiter="\t")

        for fila in lector:
            id_consulta = fila["query-id"]
            id_documento = fila["corpus-id"]
            relevancia = int(fila["score"])

            if id_consulta not in qrels:
                qrels[id_consulta] = {}

            qrels[id_consulta][id_documento] = relevancia

    return qrels

def crear_indice(documentos):
    """Construye vocabulario, índice invertido y matriz TF-IDF normalizada."""
    textos = []
    indice_invertido = {}
    longitudes = []

    for numero_documento, documento in enumerate(documentos):
        # El título y el cuerpo forman UNA unidad: los qrels usan el ID original.
        texto = documento["title"] + " " + documento["text"]
        textos.append(texto)
        frecuencias = Counter(tokenizar(texto))
        longitudes.append(sum(frecuencias.values()))

        for termino, frecuencia in frecuencias.items():
            if termino not in indice_invertido:
                indice_invertido[termino] = {}

            indice_invertido[termino][numero_documento] = frecuencia

    # L2 hace que el producto punto de dos vectores sea similitud coseno.
    vectorizador = TfidfVectorizer(analyzer=tokenizar, norm="l2", dtype=np.float32)
    matriz_tfidf = vectorizador.fit_transform(textos)

    sistema = {
        "documentos": documentos,
        "indice_invertido": indice_invertido,
        "longitudes": longitudes,
        "longitud_media": sum(longitudes) / len(longitudes),
        "vectorizador": vectorizador,
        "matriz_tfidf": matriz_tfidf,
        "k1": 1.2,
        "b": 0.75,
    }
    return sistema

def buscar_bm25(sistema, consulta, top_k):
    """Acumula BM25 solo sobre listas de documentos con términos de consulta."""
    tokens_consulta = tokenizar(consulta)
    puntuaciones = {}
    total_documentos = len(sistema["documentos"])

    # Counter cuenta repetición de términos de consulta sin una sintaxis compacta.
    for termino, frecuencia_consulta in Counter(tokens_consulta).items():
        apariciones = sistema["indice_invertido"].get(termino, {})
        frecuencia_documental = len(apariciones)

        if frecuencia_documental == 0:
            continue

        idf = math.log(1 + (total_documentos - frecuencia_documental + 0.5)
                       / (frecuencia_documental + 0.5))

        for numero_documento, frecuencia_documento in apariciones.items():
            longitud = sistema["longitudes"][numero_documento]
            ajuste = sistema["k1"] * (
                1 - sistema["b"] + sistema["b"] * longitud / sistema["longitud_media"]
            )
            aporte = idf * frecuencia_documento * (sistema["k1"] + 1)
            aporte = aporte / (frecuencia_documento + ajuste)
            aporte = aporte * frecuencia_consulta

            if numero_documento not in puntuaciones:
                puntuaciones[numero_documento] = 0.0

            puntuaciones[numero_documento] += aporte

    ranking = sorted(
        puntuaciones.items(),
        key=lambda par: (-par[1], sistema["documentos"][par[0]]["id"]),
    )
    return ranking[:top_k]


def buscar_tfidf(sistema, consulta, top_k):
    """Transforma una consulta libre y compara con todos los documentos."""
    vector_consulta = sistema["vectorizador"].transform([consulta])
    puntuaciones = (sistema["matriz_tfidf"] @ vector_consulta.T).toarray().ravel()
    ranking = []

    for numero_documento, puntuacion in enumerate(puntuaciones):
        if puntuacion > 0:
            ranking.append((numero_documento, float(puntuacion)))

    ranking.sort(
        key=lambda par: (-par[1], sistema["documentos"][par[0]]["id"])
    )
    return ranking[:top_k]


def buscar(sistema, consulta, modelo, top_k):
    """Entrega IDs originales, puntuaciones y texto para el servicio JSON."""
    if modelo == "bm25":
        ranking = buscar_bm25(sistema, consulta, top_k)
    else:
        ranking = buscar_tfidf(sistema, consulta, top_k)

    resultados = []

    for posicion, par in enumerate(ranking, start=1):
        numero_documento, puntuacion = par
        documento = sistema["documentos"][numero_documento]
        resultado = {
            "rank": posicion,
            "document_id": documento["id"],
            "score": float(puntuacion),
            "title": documento["title"],
            "snippet": documento["text"][:240],
        }
        resultados.append(resultado)

    return resultados

def metricas_consulta(resultados, relevancias, k=10):
    """Calcula P/R binarios y NDCG con grados de relevancia originales."""
    recuperados = resultados[:k]
    positivos = 0
    total_relevantes = 0
    dcg = 0.0

    for grado in relevancias.values():
        if grado > 0:
            total_relevantes += 1

    for posicion, resultado in enumerate(recuperados, start=1):
        grado = relevancias.get(resultado["document_id"], 0)

        if grado > 0:
            positivos += 1

        dcg += (2 ** grado - 1) / math.log2(posicion + 1)

    grados_ideales = sorted(relevancias.values(), reverse=True)[:k]
    idcg = 0.0

    for posicion, grado in enumerate(grados_ideales, start=1):
        idcg += (2 ** grado - 1) / math.log2(posicion + 1)

    precision = positivos / k
    recall = positivos / total_relevantes if total_relevantes > 0 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"precision@10": precision, "recall@10": recall, "ndcg@10": ndcg}


def evaluar(sistema, consultas, qrels, modelo):
    """Evalúa cada consulta de test y promedia las métricas sin saltarse ninguna."""
    filas = []

    for id_consulta, relevancias in qrels.items():
        texto = consultas[id_consulta]
        resultados = buscar(sistema, texto, modelo, 10)
        metricas = metricas_consulta(resultados, relevancias)
        fila = {"query_id": id_consulta}
        fila.update(metricas)
        filas.append(fila)

    promedio = {}

    for nombre in ("precision@10", "recall@10", "ndcg@10"):
        suma = 0.0

        for fila in filas:
            suma += fila[nombre]

        promedio[nombre] = suma / len(filas)

    return filas, promedio
