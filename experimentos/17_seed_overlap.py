#!/usr/bin/env python3
"""17. Solapamiento entre listas top-K al variar el producto-semilla.

Por qué existe este script. La sección 4.2.1 de la tesis afirmaba un
solapamiento medio del 76,5 % sobre cuarenta semillas escogidas al azar. Esa
cifra no procedía de ningún artefacto del repositorio: el cuaderno 13 solo
produce cinco semillas, y esas cinco están elegidas a mano con expresiones
regulares sobre la descripción, no al azar. Este script hace de verdad el
experimento que el texto describía.

Qué mide. Se fija el mismo contexto representativo del cuaderno 13 (RUC de mayor
frecuencia, ciudad y ruta modales, geolocalización mediana) y solo se varía el
producto-semilla. Si el modelo recuperase complementos condicionados al artículo
consultado, las listas deberían diferir entre sí. El solapamiento medio entre
pares mide cuánto se parecen.

Dos decisiones de muestreo que conviene dejar por escrito:

1. Las semillas se sortean entre los productos que aparecen como COD_PROD en el
   conjunto de entrenamiento, no sobre el catálogo entero. Un producto sin
   interacciones tiene su embedding de identidad sin entrenar, de modo que
   incluirlo mediría el arranque en frío y no lo que aquí interesa. La variante
   sobre el catálogo completo se calcula igualmente como comprobación.

2. El intervalo se obtiene remuestreando SEMILLAS, no pares. Los C(n,2) pares no
   son independientes: salen de n listas. Tratarlos como independientes daría un
   intervalo falsamente estrecho, que es justo el defecto que la sección 3.1.5
   declara al hablar del criterio de no solapamiento.

Salidas:
    experimentos/seed_overlap.csv          una fila por semilla, con su top-10
    experimentos/seed_overlap_summary.json las cifras que cita la tesis
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import polars as pl
import tensorflow as tf

N_SEMILLAS = 40
K = 10
REPLICAS = 1000
SEMILLA_ALEATORIA = 42

np.random.seed(SEMILLA_ALEATORIA)
tf.random.set_seed(SEMILLA_ALEATORIA)

train_df = pl.read_parquet("data_processed/retrieval_train.parquet")
products_df = (pl.read_parquet("data_processed/products_catalog.parquet")
                 .unique(subset=["id_producto"]))
products_df = products_df.with_columns([
    pl.col("marca").fill_null("SIN_MARCA"),
    pl.col("familia1").fill_null("SIN_CATEGORIA"),
    pl.col("familia2").fill_null("SIN_SUBCATEGORIA"),
    pl.col("precio").fill_null(0.0),
    pl.col("peso_unitario").fill_null(0.0),
    pl.col("descripción corta").fill_null("(sin descripción)").alias("desc")])

catalog_arr = np.array(products_df["id_producto"].to_list())
desc_map = dict(products_df.select(["id_producto", "desc"]).iter_rows())
fam_map = dict(products_df.select(["id_producto", "familia1"]).iter_rows())

vocab_ruc = sorted(train_df["RUC"].unique().to_list())
vocab_ciudad = sorted(train_df["CIUDAD"].unique().to_list())
vocab_ruta = sorted(train_df["RUTA"].unique().to_list())
vocab_products = sorted(products_df["id_producto"].unique().to_list())
vocab_marca = sorted(products_df["marca"].unique().to_list())
vocab_familia1 = sorted(products_df["familia1"].unique().to_list())
vocab_familia2 = sorted(products_df["familia2"].unique().to_list())


class QueryTower(tf.keras.Model):
    """Idéntica a la de los cuadernos 06, 07 y 13. No se altera ni un valor."""

    def __init__(self, vocab_ruc, vocab_ciudad, vocab_ruta, vocab_products,
                 embedding_dim=128, dropout_rate=0.2):
        super().__init__()
        self.ruc_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_ruc, mask_token=None)
        self.ruc_embedding = tf.keras.layers.Embedding(len(vocab_ruc) + 1, 64, name="ruc_emb")
        self.ciudad_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_ciudad, mask_token=None)
        self.ciudad_embedding = tf.keras.layers.Embedding(len(vocab_ciudad) + 1, 16, name="ciudad_emb")
        self.ruta_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_ruta, mask_token=None)
        self.ruta_embedding = tf.keras.layers.Embedding(len(vocab_ruta) + 1, 32, name="ruta_emb")
        self.product_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_products, mask_token=None)
        self.product_embedding = tf.keras.layers.Embedding(len(vocab_products) + 1, 64, name="product_emb")
        self.geo_normalization = tf.keras.layers.Normalization(axis=-1)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(embedding_dim, name="query_projection")])

    def call(self, inputs):
        ruc_emb = self.ruc_embedding(self.ruc_lookup(inputs["RUC"]))
        ciudad_emb = self.ciudad_embedding(self.ciudad_lookup(inputs["CIUDAD"]))
        ruta_emb = self.ruta_embedding(self.ruta_lookup(inputs["RUTA"]))
        product_emb = self.product_embedding(self.product_lookup(inputs["COD_PROD"]))
        lat = tf.expand_dims(inputs["LATITUD"], axis=-1)
        lon = tf.expand_dims(inputs["LONGITUD"], axis=-1)
        geo_norm = self.geo_normalization(tf.concat([lat, lon], axis=-1))
        return tf.math.l2_normalize(
            self.mlp(tf.concat([ruc_emb, ciudad_emb, ruta_emb, product_emb, geo_norm], axis=-1)),
            axis=-1)


class CandidateTower(tf.keras.Model):
    def __init__(self, vocab_products, vocab_marca, vocab_familia1, vocab_familia2,
                 embedding_dim=128, dropout_rate=0.2):
        super().__init__()
        self.product_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_products, mask_token=None)
        self.product_embedding = tf.keras.layers.Embedding(len(vocab_products) + 1, 64, name="candidate_product_emb")
        self.marca_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_marca, mask_token=None)
        self.marca_embedding = tf.keras.layers.Embedding(len(vocab_marca) + 1, 16, name="candidate_marca_emb")
        self.fam1_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_familia1, mask_token=None)
        self.fam1_embedding = tf.keras.layers.Embedding(len(vocab_familia1) + 1, 16, name="candidate_fam1_emb")
        self.fam2_lookup = tf.keras.layers.StringLookup(vocabulary=vocab_familia2, mask_token=None)
        self.fam2_embedding = tf.keras.layers.Embedding(len(vocab_familia2) + 1, 16, name="candidate_fam2_emb")
        self.continuous_normalization = tf.keras.layers.Normalization(axis=-1)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(embedding_dim, name="candidate_projection")])

    def call(self, inputs):
        prod_emb = self.product_embedding(self.product_lookup(inputs["id_producto"]))
        marca_emb = self.marca_embedding(self.marca_lookup(inputs["marca"]))
        fam1_emb = self.fam1_embedding(self.fam1_lookup(inputs["familia1"]))
        fam2_emb = self.fam2_embedding(self.fam2_lookup(inputs["familia2"]))
        precio = tf.expand_dims(inputs["precio"], axis=-1)
        peso = tf.expand_dims(inputs["peso_unitario"], axis=-1)
        cont_norm = self.continuous_normalization(tf.concat([precio, peso], axis=-1))
        return tf.math.l2_normalize(
            self.mlp(tf.concat([prod_emb, marca_emb, fam1_emb, fam2_emb, cont_norm], axis=-1)),
            axis=-1)


def adapt_and_init(qt, ct):
    geo_train = train_df.select(["LATITUD", "LONGITUD"]).sample(n=100_000, seed=42).to_numpy().astype(np.float32)
    qt.geo_normalization.adapt(geo_train)
    ct.continuous_normalization.adapt(
        products_df.select(["precio", "peso_unitario"]).to_numpy().astype(np.float32))
    qt({"RUC": tf.constant([vocab_ruc[0]]), "CIUDAD": tf.constant([vocab_ciudad[0]]),
        "RUTA": tf.constant([vocab_ruta[0]]), "LATITUD": tf.constant([0.0], dtype=tf.float32),
        "LONGITUD": tf.constant([0.0], dtype=tf.float32),
        "COD_PROD": tf.constant([vocab_products[0]])})
    ct({"id_producto": tf.constant([vocab_products[0]]), "marca": tf.constant([vocab_marca[0]]),
        "familia1": tf.constant([vocab_familia1[0]]), "familia2": tf.constant([vocab_familia2[0]]),
        "precio": tf.constant([0.0], dtype=tf.float32),
        "peso_unitario": tf.constant([0.0], dtype=tf.float32)})


def candidate_embeddings(ct, pdf):
    ds = tf.data.Dataset.from_tensor_slices({
        "id_producto": pdf["id_producto"].to_numpy(), "marca": pdf["marca"].to_numpy(),
        "familia1": pdf["familia1"].to_numpy(), "familia2": pdf["familia2"].to_numpy(),
        "precio": pdf["precio"].to_numpy().astype(np.float32),
        "peso_unitario": pdf["peso_unitario"].to_numpy().astype(np.float32)}).batch(1024)
    return tf.concat([ct(b) for b in ds], axis=0)


def resumen(listas: dict[str, list[str]], etiqueta: str) -> dict:
    """Media de solapamiento por pares, con intervalo remuestreando semillas."""
    nombres = list(listas)
    pares = {(a, b): len(set(listas[a]) & set(listas[b])) / K
             for a, b in itertools.combinations(nombres, 2)}
    media = float(np.mean(list(pares.values())))

    rng = np.random.default_rng(SEMILLA_ALEATORIA)
    replicas = []
    for _ in range(REPLICAS):
        muestra = rng.choice(nombres, size=len(nombres), replace=True)
        vals = [pares[(a, b)] if (a, b) in pares else pares[(b, a)]
                for a, b in itertools.combinations(muestra, 2) if a != b]
        if vals:
            replicas.append(np.mean(vals))
    bajo, alto = np.percentile(replicas, [2.5, 97.5])

    union = set().union(*listas.values())
    print(f"\n== {etiqueta} ==")
    print(f"   semillas         : {len(nombres)}")
    print(f"   pares            : {len(pares)}")
    print(f"   solapamiento     : {100*media:.1f} %  IC 95 % [{100*bajo:.1f}; {100*alto:.1f}]")
    print(f"   productos únicos : {len(union)} en la unión de las listas")
    return {"semillas": len(nombres), "pares": len(pares),
            "solapamiento_medio_pct": round(100 * media, 1),
            "ic95_pct": [round(100 * bajo, 1), round(100 * alto, 1)],
            "productos_distintos_union": len(union),
            "replicas_bootstrap": REPLICAS}


def main() -> None:
    print("TF", tf.__version__, "| Polars", pl.__version__)
    qt = QueryTower(vocab_ruc, vocab_ciudad, vocab_ruta, vocab_products)
    ct = CandidateTower(vocab_products, vocab_marca, vocab_familia1, vocab_familia2)
    adapt_and_init(qt, ct)
    qt.load_weights("models/query_tower.weights.h5")
    ct.load_weights("models/candidate_tower.weights.h5")
    cand_emb = candidate_embeddings(ct, products_df).numpy()
    print("Embeddings de catálogo:", cand_emb.shape)

    ctx_ciudad = train_df["CIUDAD"].mode().to_list()[0]
    ctx_ruta = train_df["RUTA"].mode().to_list()[0]
    ctx_lat = float(train_df["LATITUD"].median())
    ctx_lon = float(train_df["LONGITUD"].median())
    ctx_ruc = train_df["RUC"].value_counts(sort=True)["RUC"].to_list()[0]
    print(f"Contexto fijo -> RUC={ctx_ruc}, CIUDAD={ctx_ciudad}, RUTA={ctx_ruta}, "
          f"LAT={ctx_lat:.2f}, LON={ctx_lon:.2f}")

    def recomendar(seed_id: str) -> list[str]:
        q = {"RUC": tf.constant([ctx_ruc]), "CIUDAD": tf.constant([ctx_ciudad]),
             "RUTA": tf.constant([ctx_ruta]),
             "LATITUD": tf.constant([ctx_lat], dtype=tf.float32),
             "LONGITUD": tf.constant([ctx_lon], dtype=tf.float32),
             "COD_PROD": tf.constant([seed_id])}
        orden = np.argsort(-(cand_emb @ qt(q).numpy()[0]))
        return [catalog_arr[j] for j in orden if catalog_arr[j] != seed_id][:K]

    observados = set(train_df["COD_PROD"].unique().to_list()) & set(vocab_products)
    poblaciones = {
        "semillas observadas en entrenamiento": sorted(observados),
        "catálogo completo": vocab_products,
    }

    rng = np.random.default_rng(SEMILLA_ALEATORIA)
    salida, filas = {}, []
    for etiqueta, poblacion in poblaciones.items():
        elegidas = rng.choice(np.array(poblacion), size=N_SEMILLAS, replace=False)
        listas = {sid: recomendar(sid) for sid in elegidas}
        salida[etiqueta] = resumen(listas, etiqueta)
        for sid, recs in listas.items():
            filas.append({"poblacion": etiqueta, "semilla_id": sid,
                          "semilla_desc": desc_map.get(sid, "?"),
                          "semilla_familia": fam_map.get(sid, "?"),
                          "top_k_desc": " | ".join(desc_map.get(r, "?") for r in recs)})

    pl.DataFrame(filas).write_csv("experimentos/seed_overlap.csv")
    salida["parametros"] = {"K": K, "n_semillas": N_SEMILLAS,
                            "semilla_aleatoria": SEMILLA_ALEATORIA,
                            "contexto": {"RUC": ctx_ruc, "CIUDAD": ctx_ciudad,
                                         "RUTA": ctx_ruta,
                                         "LATITUD": round(ctx_lat, 4),
                                         "LONGITUD": round(ctx_lon, 4)}}
    with open("experimentos/seed_overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print("\nGuardado experimentos/seed_overlap.csv y seed_overlap_summary.json")


if __name__ == "__main__":
    main()
