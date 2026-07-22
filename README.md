# Sistema de recomendación Two-Towers para distribución ferretera B2B

Recomendador de recuperación de candidatos (*retrieval*) basado en la arquitectura **Two-Towers**, entrenado sobre transacciones de una distribuidora ferretera ecuatoriana. Trabajo de fin de máster (TFM).

El modelo aprende a proyectar clientes (contexto de compra) y productos (SKUs) a un mismo espacio vectorial denso, de modo que la similitud coseno entre ambos recupera el Top-K de productos relevantes con latencias de milisegundos mediante búsqueda ANN.

---

## Resultados principales

Evaluación offline sobre el conjunto de validación. Intervalos de confianza al 95 % por *bootstrap* (ver [experimentos/12_extended_metrics_and_ci.ipynb](experimentos/12_extended_metrics_and_ci.ipynb)).

| Modelo | Recall@10 | Recall@50 | Recall@100 | NDCG@100 | MRR |
|---|---|---|---|---|---|
| Popularidad global | 3.90 % | 13.20 % | 21.25 % | 0.0535 | 0.0219 |
| Co-ocurrencia ítem-ítem | 6.90 % | 16.85 % | 24.05 % | 0.0734 | **0.0389** |
| **Two-Towers (LogQ)** | **7.40 %** | **21.50 %** | **33.05 %** | **0.0858** | 0.0354 |

- **Two-Towers gana en Recall y NDCG** en todos los cortes; el salto más grande es en Recall@100 (33.05 % vs. 24.05 % del mejor baseline).
- En **MRR** la co-ocurrencia queda ligeramente arriba: el modelo denso prioriza cobertura amplia de candidatos por sobre ubicar un único acierto en la primerísima posición — comportamiento esperable en la fase de *retrieval*, donde importa que el ítem relevante entre en el Top-K, no que sea el #1.

### Cold-start (ver [experimentos/10_cold_start_evaluation.ipynb](experimentos/10_cold_start_evaluation.ipynb))

| Escenario | Recall@10 | Recall@100 | MRR |
|---|---|---|---|
| Usuarios nuevos | 7.70 % | 27.20 % | 0.0368 |
| Ítems nuevos | 0.00 % | 0.70 % | 0.0006 |

El modelo generaliza a **usuarios nuevos** vía sus atributos de contexto (ciudad, ruta, ubicación), pero el *cold-start de ítems* sigue siendo la limitación abierta: sin señal de co-compra histórica, la torre de candidatos no ubica SKUs recién incorporados.

---

## Estructura del repositorio

```
.
├── 01_unification_anonymization.ipynb   # Unifica CSV de ventas y anonimiza RUC
├── 02_eda.ipynb                         # Análisis exploratorio
├── 03_retrieval_dataset.ipynb           # Construye pares (cliente, producto) de entrenamiento
├── 04_query_tower.ipynb                 # Torre de consulta (cliente + contexto)
├── 05_candidate_tower.ipynb             # Torre de candidatos (producto + atributos)
├── 06_training_logq.ipynb               # Entrenamiento con corrección LogQ / in-batch negatives
├── 07_offline_evaluation.ipynb          # Evaluación offline (Recall, MRR)
├── 08_embeddings_visualization.ipynb    # Proyección t-SNE/UMAP de embeddings
├── run_embeddings_analysis.py           # Script auxiliar de análisis de embeddings
│
├── experimentos/                        # Experimentos y ablations de la tesis
│   ├── 09_ablation_and_diversity.ipynb  # Ablation de features + diversidad de recomendaciones
│   ├── 10_cold_start_evaluation.ipynb   # Evaluación de cold-start (usuarios / ítems)
│   ├── 11_ann_latency_vs_recall.ipynb   # Trade-off latencia ANN vs. recall (FAISS)
│   ├── 12_extended_metrics_and_ci.ipynb # NDCG + intervalos de confianza por bootstrap
│   ├── 13_topk_qualitative.ipynb        # Inspección cualitativa de recomendaciones Top-K
│   ├── 16_cooccurrence_baseline.ipynb   # Baseline de co-ocurrencia ítem-ítem
│   ├── 17_data_quality_audit.py         # Auditoría de calidad de datos
│   ├── 18_hyperparameter_selection.py   # Selección de hiperparámetros (2 fases)
│   ├── results_*.csv / *.json           # Métricas y resultados (versionados)
│   └── fig_*.png                        # Figuras de la tesis
│
├── models/                              # Resultados del modelo final (pesos NO incluidos)
│   ├── final_model_results.json
│   └── evaluation_results.txt
│
├── data_raw/                            # Datos crudos (privados — no incluidos)
└── data_processed/                      # Datos derivados (regenerables — no incluidos)
```

---

## Datos

Los datos son **transacciones reales y privadas** de una distribuidora ferretera; **no se incluyen en el repositorio**.

- `data_raw/` — CSV mensuales de ventas (2024–2026). Datos originales sin anonimizar. Solicitar a los autores.
- `data_processed/` — Parquets derivados (catálogo de productos, dataset de retrieval, mapeo de RUC anonimizados). Se **regeneran** ejecutando los notebooks `01`–`03`.

El notebook `01` anonimiza el `RUC` del cliente antes de cualquier análisis. El dataset de entrenamiento final contiene **~32.5 M de pares** (cliente, producto).

### Atributos usados

- **Torre de consulta (cliente):** `RUC` anonimizado, `CIUDAD`, `LATITUD`, `LONGITUD`, `RUTA` e historial reciente.
- **Torre de candidatos (producto):** `COD_PROD`, `MARCA`, `FAMILIA1`, `FAMILIA2`, precio y peso.

---

## Reproducción

Requiere **Python 3.12**. El proyecto usa [`uv`](https://github.com/astral-sh/uv) para gestión de dependencias.

Dependencias principales: `tensorflow==2.21.0`, `tensorflow-recommenders==0.7.2`, `faiss-cpu==1.14.2`, `scikit-surprise==1.1.5`, `polars`, `pandas`, `scikit-learn`, `umap-learn`, `matplotlib`, `seaborn`, `jupyter`.

```bash
# 1. Crear entorno e instalar dependencias
uv sync

# 2. Ejecutar el pipeline en orden (requiere datos en data_raw/)
uv run jupyter nbconvert --to notebook --execute --inplace 01_unification_anonymization.ipynb
# ... continuar con 02 a 08

# Ejecutar un experimento puntual, p. ej. métricas extendidas:
uv run jupyter nbconvert --to notebook --execute --inplace \
    experimentos/12_extended_metrics_and_ci.ipynb
```

> **Orden del pipeline:** `01 → 02 → 03` (datos) → `04 → 05 → 06` (torres + entrenamiento) → `07 → 08` (evaluación + visualización). Los notebooks de `experimentos/` dependen de los artefactos generados por el pipeline principal.

### Configuración del modelo final

`batch_size=1024`, `temperature=0.05`, `normalize=True`, `remove_accidental_hits=True`, `dropout=0.0`, `seed=42`. Selección de época por `Recall@100` (coseno) sobre el 25 % de validación. Ver [models/final_model_results.json](models/final_model_results.json).

---

## Arquitectura

| Componente | Rol |
|---|---|
| **Query Tower** | Codifica al cliente y su contexto (ubicación, ruta, historial) en un embedding denso de dimensión *D*. |
| **Candidate Tower** | Codifica cada SKU (marca, familias, precio, peso) en el mismo espacio de dimensión *D*. Se precomputa en batch. |
| **Motor ANN** | Indexa los embeddings del catálogo (FAISS) y recupera el Top-K por similitud coseno. |

Entrenamiento con **muestreo de negativos in-batch** y **corrección LogQ** para contrarrestar el sesgo de popularidad inducido por el muestreo.

---

## Limitaciones

- **Cold-start de ítems**: productos sin historial de co-compra no se recuperan bien (Recall@100 ≈ 0.7 %).
- Evaluación **offline**: no se midió impacto online (CTR, conversión) por falta de un entorno de A/B testing.
- Pesos del modelo y datos procesados **no se versionan** por tamaño y privacidad; se regeneran ejecutando el pipeline.

---

## Autor

Allan Orellana V., Paul Córtez B.  — Trabajo de Fin de Máster, Universidad Internacional del Ecuador (UIDE).
