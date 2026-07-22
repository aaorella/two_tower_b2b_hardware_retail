"""
18. Selección de hiperparámetros del recomendador Two Towers.

Búsqueda secuencial por grupos de hiperparámetros sobre el 25 % de los pares de
entrenamiento, que es el punto de rendimientos decrecientes de la curva de datos
(cuaderno 09). La configuración seleccionada se entrena después sobre la totalidad
de los pares en el cuaderno 06.

MÉTRICA DE SELECCIÓN. Recall@100 bajo similitud coseno sobre el catálogo completo
(20.683 candidatos). Se elige coseno porque es la función de recuperación que
utiliza el índice FAISS en servido (cuaderno 11): evaluar con una similitud
distinta de la que se sirve mediría un sistema que no existe en producción.

NÚMERO DE ÉPOCAS. No es un valor fijado a priori sino el resultado de una parada
temprana sobre esa misma métrica, con paciencia de 2 épocas y restauración de los
pesos de la mejor época. La pérdida in-batch no se usa como criterio: depende de
la composición del lote y no es una señal limpia de generalización.

FASES.
  A. Muestreo de negativos: tamaño de lote x enmascarado de falsos negativos
     (remove_accidental_hits) x dropout. Con negativos in-batch el tamaño de lote
     determina cuántos negativos recibe cada positivo, y el enmascarado evita que
     un ítem que aparece como positivo para otra consulta del mismo lote se use
     como negativo contra sí mismo.
  B. Parametrización de la similitud: normalización L2 de la salida de ambas
     torres y temperatura del softmax. Con embeddings unitarios los logits quedan
     acotados en [-1, 1], de modo que la temperatura deja de ser opcional; se
     incluye el caso sin temperatura como control.

REPRODUCIBILIDAD. Semilla 42 en todas las fuentes de aleatoriedad, vocabularios
ordenados y barajado del pipeline con semilla fija. El script es reanudable: los
CSV se escriben tras cada configuración y las ya evaluadas se omiten.

NOTA DE ENTORNO. TensorFlow debe importarse antes que Polars y Pandas. Ambos
distribuyen su propia copia de abseil y PyArrow exporta los mismos símbolos; si
Arrow se carga primero, la espera de absl::Notification de TensorFlow se resuelve
contra el semáforo de Arrow y el entrenamiento se bloquea de forma indefinida, sin
consumo de CPU y sin mensaje de error.
"""
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import polars as pl
import pandas as pd
import os, time, gc, json, itertools, resource

# ---------------------------------------------------------------- parámetros
SEL_FRAC = 0.25          # fracción de pares usada para seleccionar
MAX_EPOCHS = 8           # techo; el valor efectivo lo fija la parada temprana
PATIENCE = 2
K_LIST = [10, 50, 100]
N_VAL = 2000             # consultas de validación (mismo protocolo que 07 y 12)
SEED = 42

GRID_A = {"batch_size": [1024, 4096], "remove_accidental_hits": [False, True],
          "dropout": [0.2, 0.0]}
GRID_B = {"temperature": [None, 0.2, 0.1, 0.05]}

PHASE_A_CSV = "experimentos/hparam_phaseA.csv"
PHASE_B_CSV = "experimentos/hparam_phaseB.csv"
EPOCHS_CSV = "experimentos/hparam_epochs.csv"
SELECTED_JSON = "experimentos/hparam_selected.json"

_d = os.getcwd()
while not os.path.isdir("data_processed") and os.path.dirname(_d) != _d:
    os.chdir(".."); _d = os.getcwd()
assert os.path.isdir("data_processed"), f"No se encontró data_processed/ desde {os.getcwd()}"

np.random.seed(SEED); tf.random.set_seed(SEED)

def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if r > 10 ** 9 else r / (1024 ** 2)

# -------------------------------------------------------------------- datos
train_df = pl.read_parquet("data_processed/retrieval_train.parquet")
val_df = pl.read_parquet("data_processed/retrieval_val.parquet")
products_df = pl.read_parquet("data_processed/products_catalog.parquet").unique(subset=["id_producto"])
products_df = products_df.with_columns([
    pl.col("marca").fill_null("SIN_MARCA"), pl.col("familia1").fill_null("SIN_CATEGORIA"),
    pl.col("familia2").fill_null("SIN_SUBCATEGORIA"), pl.col("precio").fill_null(0.0),
    pl.col("peso_unitario").fill_null(0.0)])

vocab_ruc = sorted(train_df["RUC"].unique().to_list())
vocab_ciudad = sorted(train_df["CIUDAD"].unique().to_list())
vocab_ruta = sorted(train_df["RUTA"].unique().to_list())
vocab_products = sorted(products_df["id_producto"].unique().to_list())
vocab_marca = sorted(products_df["marca"].unique().to_list())
vocab_familia1 = sorted(products_df["familia1"].unique().to_list())
vocab_familia2 = sorted(products_df["familia2"].unique().to_list())

ren = {c: f"cand_{c}" if c != "id_producto" else c for c in products_df.columns}
products_renamed = products_df.rename(ren)

def join_cand(df):
    return df.join(products_renamed, left_on="COD_PROD_2", right_on="id_producto", how="left").rename({
        "cand_marca": "marca_2", "cand_familia1": "familia1_2", "cand_familia2": "familia2_2",
        "cand_precio": "precio_2", "cand_peso_unitario": "peso_unitario_2"}).with_columns([
        pl.col("marca_2").fill_null("SIN_MARCA"), pl.col("familia1_2").fill_null("SIN_CATEGORIA"),
        pl.col("familia2_2").fill_null("SIN_SUBCATEGORIA"), pl.col("precio_2").fill_null(0.0),
        pl.col("peso_unitario_2").fill_null(0.0)])

# Tabla de corrección LogQ: la probabilidad de muestreo se estima siempre sobre el
# conjunto de entrenamiento completo, con independencia de la fracción que se use
# para ajustar los pesos.
total_pairs = train_df.height
counts = train_df.group_by("COD_PROD_2").agg(pl.len().alias("count")).with_columns(
    (pl.col("count") / total_pairs).alias("prob"))
prob_dict = {r["COD_PROD_2"]: r["prob"] for r in counts.iter_rows(named=True)}
logq_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(vocab_products, dtype=tf.string),
        tf.constant([prob_dict.get(p, 1e-8) for p in vocab_products], dtype=tf.float32)),
    default_value=1e-8)
del counts, prob_dict; gc.collect()

val_sample = join_cand(val_df).sample(n=min(N_VAL, val_df.height), seed=SEED)
targets = val_sample["COD_PROD_2"].to_numpy()
catalog_ids = products_df["id_producto"].to_list()
id2c = {p: i for i, p in enumerate(catalog_ids)}
cont_train = products_df.select(["precio", "peso_unitario"]).to_numpy().astype(np.float32)
print(f"Pares de entrenamiento: {total_pairs:,} | catálogo: {len(catalog_ids):,} | "
      f"validación: {val_sample.height} consultas", flush=True)

# -------------------------------------------------------------- arquitectura
class QueryTower(tf.keras.Model):
    def __init__(self, vocab_ruc, vocab_ciudad, vocab_ruta, vocab_products,
                 embedding_dim=128, dropout_rate=0.2, normalize=False):
        super().__init__()
        self.normalize = normalize
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
        out = self.mlp(tf.concat([ruc_emb, ciudad_emb, ruta_emb, product_emb, geo_norm], axis=-1))
        return tf.math.l2_normalize(out, axis=-1) if self.normalize else out


class CandidateTower(tf.keras.Model):
    def __init__(self, vocab_products, vocab_marca, vocab_familia1, vocab_familia2,
                 embedding_dim=128, dropout_rate=0.2, normalize=False):
        super().__init__()
        self.normalize = normalize
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
        out = self.mlp(tf.concat([prod_emb, marca_emb, fam1_emb, fam2_emb, cont_norm], axis=-1))
        return tf.math.l2_normalize(out, axis=-1) if self.normalize else out


class HardwareTwoTowers(tfrs.Model):
    def __init__(self, query_tower, candidate_tower, logq_table,
                 remove_accidental_hits=False, temperature=None):
        super().__init__()
        self.query_tower = query_tower
        self.candidate_tower = candidate_tower
        self.logq_table = logq_table
        self.remove_accidental_hits = remove_accidental_hits
        self.task = tfrs.tasks.Retrieval(remove_accidental_hits=remove_accidental_hits,
                                         temperature=temperature)

    def compute_loss(self, features, training=False):
        qe = self.query_tower({k: features[k] for k in
                               ["RUC", "CIUDAD", "RUTA", "LATITUD", "LONGITUD", "COD_PROD"]})
        ce = self.candidate_tower({k: features[k] for k in
                                   ["id_producto", "marca", "familia1", "familia2",
                                    "precio", "peso_unitario"]})
        kwargs = {"candidate_sampling_probability": self.logq_table.lookup(features["id_producto"])}
        if self.remove_accidental_hits:
            kwargs["candidate_ids"] = features["id_producto"]
        return self.task(qe, ce, **kwargs)


def make_tf_dataset(df, batch_size, shuffle=False):
    inputs = {
        "RUC": df["RUC"].to_numpy(), "CIUDAD": df["CIUDAD"].to_numpy(), "RUTA": df["RUTA"].to_numpy(),
        "LATITUD": df["LATITUD"].to_numpy().astype(np.float32),
        "LONGITUD": df["LONGITUD"].to_numpy().astype(np.float32),
        "COD_PROD": df["COD_PROD"].to_numpy(), "id_producto": df["COD_PROD_2"].to_numpy(),
        "marca": df["marca_2"].to_numpy(), "familia1": df["familia1_2"].to_numpy(),
        "familia2": df["familia2_2"].to_numpy(),
        "precio": df["precio_2"].to_numpy().astype(np.float32),
        "peso_unitario": df["peso_unitario_2"].to_numpy().astype(np.float32)}
    ds = tf.data.Dataset.from_tensor_slices(inputs)
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------- métricas de recuperación reales
_cand_inputs = {
    "id_producto": products_df["id_producto"].to_numpy(), "marca": products_df["marca"].to_numpy(),
    "familia1": products_df["familia1"].to_numpy(), "familia2": products_df["familia2"].to_numpy(),
    "precio": products_df["precio"].to_numpy().astype(np.float32),
    "peso_unitario": products_df["peso_unitario"].to_numpy().astype(np.float32)}
_query_inputs = {
    "RUC": val_sample["RUC"].to_numpy(), "CIUDAD": val_sample["CIUDAD"].to_numpy(),
    "RUTA": val_sample["RUTA"].to_numpy(),
    "LATITUD": val_sample["LATITUD"].to_numpy().astype(np.float32),
    "LONGITUD": val_sample["LONGITUD"].to_numpy().astype(np.float32),
    "COD_PROD": val_sample["COD_PROD"].to_numpy()}


def _score(q, c):
    sim = q @ c.T
    rec = {k: 0.0 for k in K_LIST}
    mrr = 0.0
    n = len(targets)
    for i in range(n):
        j = id2c.get(targets[i])
        if j is None:
            continue
        rank = 1 + int(np.sum(sim[i] > sim[i, j]))
        mrr += 1.0 / rank
        for k in K_LIST:
            if rank <= k:
                rec[k] += 1.0
    return {f"recall{k}": rec[k] / n for k in K_LIST} | {"mrr": mrr / n}


def retrieval_metrics(query_tower, candidate_tower):
    """Recall@K y MRR contra el catálogo completo, bajo coseno y bajo producto punto.

    Se reportan ambas para dejar constancia de que coinciden cuando las torres
    emiten vectores unitarios: la similitud optimizada y la servida son la misma.
    """
    cds = tf.data.Dataset.from_tensor_slices(_cand_inputs).batch(1024)
    ce = tf.concat([candidate_tower(b, training=False) for b in cds], axis=0).numpy()
    qds = tf.data.Dataset.from_tensor_slices(_query_inputs).batch(1024)
    qe = tf.concat([query_tower(b, training=False) for b in qds], axis=0).numpy()
    dot = _score(qe, ce)
    qn = qe / np.maximum(np.linalg.norm(qe, axis=1, keepdims=True), 1e-12)
    cn = ce / np.maximum(np.linalg.norm(ce, axis=1, keepdims=True), 1e-12)
    return dot, _score(qn, cn), ce


class CosineEarlyStopping(tf.keras.callbacks.Callback):
    """Parada temprana sobre Recall@100 coseno, la métrica del sistema servido."""

    def __init__(self, query_tower, candidate_tower, tag, patience=PATIENCE):
        super().__init__()
        self.query_tower = query_tower
        self.candidate_tower = candidate_tower
        self.tag = tag
        self.patience = patience
        self.best = -1.0
        self.wait = 0
        self.best_epoch = -1
        self.best_weights = None
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        dot, cos, ce = retrieval_metrics(self.query_tower, self.candidate_tower)
        norms = np.linalg.norm(ce, axis=1)
        self.history.append({
            "epoca": epoch + 1,
            **{f"cos_{k}": v for k, v in cos.items()},
            **{f"dot_{k}": v for k, v in dot.items()},
            "norma_media": float(norms.mean()), "norma_std": float(norms.std()),
            "loss": float(logs.get("total_loss", logs.get("loss", np.nan))) if logs else np.nan})
        print(f"    [{self.tag}] época {epoch+1}: R@100(cos)={cos['recall100']*100:5.2f}%  "
              f"R@100(dot)={dot['recall100']*100:5.2f}%  MRR={cos['mrr']:.4f}", flush=True)
        if cos["recall100"] > self.best:
            self.best = cos["recall100"]
            self.best_epoch = epoch + 1
            self.wait = 0
            self.best_weights = (self.query_tower.get_weights(), self.candidate_tower.get_weights())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"    [{self.tag}] parada temprana en la época {epoch+1}; "
                      f"mejor época {self.best_epoch}", flush=True)
                self.model.stop_training = True

    def restore(self):
        if self.best_weights is not None:
            self.query_tower.set_weights(self.best_weights[0])
            self.candidate_tower.set_weights(self.best_weights[1])


def evaluate(train_data, cfg, tag):
    """Entrena una configuración con parada temprana y devuelve su mejor época."""
    tf.random.set_seed(SEED)
    qt = QueryTower(vocab_ruc, vocab_ciudad, vocab_ruta, vocab_products,
                    dropout_rate=cfg["dropout"], normalize=cfg["normalize"])
    ct = CandidateTower(vocab_products, vocab_marca, vocab_familia1, vocab_familia2,
                        dropout_rate=cfg["dropout"], normalize=cfg["normalize"])
    qt.geo_normalization.adapt(train_data.select(["LATITUD", "LONGITUD"]).to_numpy().astype(np.float32))
    ct.continuous_normalization.adapt(cont_train)
    model = HardwareTwoTowers(qt, ct, logq_table,
                              remove_accidental_hits=cfg["remove_accidental_hits"],
                              temperature=cfg["temperature"])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    cb = CosineEarlyStopping(qt, ct, tag)
    t0 = time.time()
    model.fit(make_tf_dataset(train_data, cfg["batch_size"], shuffle=True),
              epochs=MAX_EPOCHS, verbose=0, callbacks=[cb])
    cb.restore()
    secs = time.time() - t0
    best = cb.history[cb.best_epoch - 1]
    row = {**cfg, "mejor_epoca": cb.best_epoch, "epocas_corridas": len(cb.history),
           "cos_recall10": best["cos_recall10"], "cos_recall50": best["cos_recall50"],
           "cos_recall100": best["cos_recall100"], "cos_mrr": best["cos_mrr"],
           "dot_recall100": best["dot_recall100"],
           "minutos": round(secs / 60, 1), "rss_pico_GB": round(rss_gb(), 2)}
    del qt, ct, model
    gc.collect(); tf.keras.backend.clear_session()
    return row, cb.history


def _config_key(cfg):
    """Clave canónica de una configuración, estable entre memoria y CSV.

    Necesaria porque al releer un CSV los enteros llegan como numpy.int64, los
    booleanos como numpy.bool_ y los None como NaN; comparar sin normalizar haría
    que la reanudación no reconociera configuraciones ya evaluadas.
    """
    def norm(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "None"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):g}"
        return str(v)
    return "|".join(f"{k}={norm(cfg[k])}" for k in sorted(cfg))


def run_phase(train_data, configs, csv_path, phase_name):
    """Evalúa una lista de configuraciones; reanudable desde el CSV."""
    rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []
    epoch_rows = pd.read_csv(EPOCHS_CSV).to_dict("records") if os.path.exists(EPOCHS_CSV) else []
    seen = {_config_key({k: r[k] for k in configs[0]}) for r in rows}
    for cfg in configs:
        key = _config_key(cfg)
        if key in seen:
            print(f"  {cfg} ya evaluada — se omite", flush=True)
            continue
        tag = "/".join(f"{k[:3]}={v}" for k, v in cfg.items())
        print(f"\n  --- {tag} ---", flush=True)
        row, history = evaluate(train_data, cfg, tag)
        rows.append(row)
        epoch_rows.extend({**cfg, "fase": phase_name, **h} for h in history)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        pd.DataFrame(epoch_rows).to_csv(EPOCHS_CSV, index=False)
        print(f"  {tag}: R@100(cos)={row['cos_recall100']*100:.2f}% en la época "
              f"{row['mejor_epoca']} [{row['minutos']} min]", flush=True)
    return pd.DataFrame(rows).sort_values("cos_recall100", ascending=False)


if __name__ == "__main__":
    train_sel = join_cand(train_df.sample(fraction=SEL_FRAC, seed=SEED))
    print(f"\nSelección sobre {train_sel.height:,} pares ({SEL_FRAC*100:.0f} %)", flush=True)

    # --- Fase A: muestreo de negativos -------------------------------------
    print("\n=== FASE A — muestreo de negativos ===", flush=True)
    configs_a = [{"batch_size": b, "remove_accidental_hits": r, "dropout": d,
                  "normalize": False, "temperature": None}
                 for b, r, d in itertools.product(GRID_A["batch_size"],
                                                  GRID_A["remove_accidental_hits"],
                                                  GRID_A["dropout"])]
    res_a = run_phase(train_sel, configs_a, PHASE_A_CSV, "A")
    print("\n" + res_a.to_string(index=False), flush=True)
    best_a = res_a.iloc[0]
    base = {"batch_size": int(best_a["batch_size"]),
            "remove_accidental_hits": bool(best_a["remove_accidental_hits"]),
            "dropout": float(best_a["dropout"])}
    print(f"\nFase A -> {base}", flush=True)

    # --- Fase B: parametrización de la similitud ---------------------------
    print("\n=== FASE B — parametrización de la similitud ===", flush=True)
    configs_b = [{**base, "normalize": True, "temperature": t} for t in GRID_B["temperature"]]
    res_b = run_phase(train_sel, configs_b, PHASE_B_CSV, "B")
    print("\n" + res_b.to_string(index=False), flush=True)
    best_b = res_b.iloc[0]

    selected = {**base, "normalize": True,
                "temperature": None if pd.isna(best_b["temperature"]) else float(best_b["temperature"]),
                "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seed": SEED,
                "seleccion_frac": SEL_FRAC,
                "metrica_seleccion": "Recall@100 coseno",
                "cos_recall100_seleccion": float(best_b["cos_recall100"]),
                "mejor_epoca_seleccion": int(best_b["mejor_epoca"])}
    with open(SELECTED_JSON, "w") as fh:
        json.dump(selected, fh, indent=2, ensure_ascii=False)
    print(f"\nCONFIGURACIÓN SELECCIONADA -> {SELECTED_JSON}", flush=True)
    print(json.dumps(selected, indent=2, ensure_ascii=False), flush=True)
