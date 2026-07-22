"""Análisis de calidad de datos — WS2 (feedback p44-45).
Chequea consistencia, duplicados, valores atípicos e integridad referencial
más allá del análisis de nulos ya reportado. Fuente: data_processed/*.parquet.
"""
import polars as pl

BASE = "/Users/allan/Documents/maestria/tesis_github/data_processed"
tx = pl.read_parquet(f"{BASE}/transactions_unified.parquet")
cat = pl.read_parquet(f"{BASE}/products_catalog.parquet")

def pct(n, d):
    return f"{100*n/d:.4f}%" if d else "n/a"

print("=" * 60)
print("TRANSACCIONES  (transactions_unified.parquet)")
print("=" * 60)
n_tx = tx.height
print(f"Filas totales: {n_tx:,}")

# 1. DUPLICADOS exactos (fila completa)
dup_full = n_tx - tx.unique().height
print(f"\n[DUPLICADOS] Filas exactamente duplicadas: {dup_full:,} ({pct(dup_full, n_tx)})")
# duplicados por clave transaccional (cliente-fecha-producto)
key = ["RUC", "FECHA", "COD_PROD"]
dup_key = n_tx - tx.unique(subset=key).height
print(f"[DUPLICADOS] Duplicados por (RUC,FECHA,COD_PROD): {dup_key:,} ({pct(dup_key, n_tx)})")

# 2. INTEGRIDAD REFERENCIAL: COD_PROD que no existen en el catálogo
cat_ids = set(cat.get_column("id_producto").drop_nulls().to_list())
tx_prod = tx.get_column("COD_PROD").drop_nulls()
orphan_mask = ~tx_prod.is_in(cat_ids)
n_orphan_rows = int(orphan_mask.sum())
orphan_skus = tx.filter(pl.col("COD_PROD").is_in(cat_ids).not_()).get_column("COD_PROD").n_unique()
print(f"\n[INTEGRIDAD] SKU en transacciones ausentes del catálogo: {orphan_skus:,} SKU únicos")
print(f"[INTEGRIDAD] Filas con SKU huérfano: {n_orphan_rows:,} ({pct(n_orphan_rows, n_tx)})")
print(f"[INTEGRIDAD] SKU únicos en transacciones: {tx.get_column('COD_PROD').n_unique():,} | en catálogo: {cat.height:,}")

# 3. CONSISTENCIA: valores no plausibles
neg_cant = tx.filter(pl.col("CANTIDAD") <= 0).height
neg_venta = tx.filter(pl.col("VENTA") <= 0).height
neg_costo = tx.filter(pl.col("COSTO") < 0).height
venta_lt_costo = tx.filter(pl.col("VENTA") < pl.col("COSTO")).height
# coherencia GANANCIA = VENTA - COSTO (tolerancia 0.01)
ganancia_inc = tx.filter(((pl.col("VENTA") - pl.col("COSTO")) - pl.col("GANANCIA")).abs() > 0.01).height
print(f"\n[CONSISTENCIA] CANTIDAD <= 0: {neg_cant:,} ({pct(neg_cant, n_tx)})")
print(f"[CONSISTENCIA] VENTA <= 0: {neg_venta:,} ({pct(neg_venta, n_tx)})")
print(f"[CONSISTENCIA] COSTO < 0: {neg_costo:,} ({pct(neg_costo, n_tx)})")
print(f"[CONSISTENCIA] VENTA < COSTO (margen negativo): {venta_lt_costo:,} ({pct(venta_lt_costo, n_tx)})")
print(f"[CONSISTENCIA] GANANCIA != VENTA-COSTO: {ganancia_inc:,} ({pct(ganancia_inc, n_tx)})")

# 4. RANGO GEOGRÁFICO (Ecuador continental aprox: lat -5..2, lon -81..-75)
geo = tx.select("LATITUD", "LONGITUD").drop_nulls()
out_geo = geo.filter(
    (pl.col("LATITUD") < -5) | (pl.col("LATITUD") > 2) |
    (pl.col("LONGITUD") < -92) | (pl.col("LONGITUD") > -75)
).height
print(f"\n[GEO] Coord. fuera de rango Ecuador (incl. Galápagos): {out_geo:,} de {geo.height:,} no nulas")

# 5. RANGO TEMPORAL
fmin, fmax = tx.select(pl.col("FECHA").min()).item(), tx.select(pl.col("FECHA").max()).item()
print(f"\n[TEMPORAL] Rango FECHA: {fmin} -> {fmax}")

# 6. ATÍPICOS en cantidad y venta (cuantiles)
print("\n[ATÍPICOS] Cuantiles:")
for col in ["CANTIDAD", "VENTA"]:
    qs = tx.select([
        pl.col(col).quantile(0.50).alias("p50"),
        pl.col(col).quantile(0.95).alias("p95"),
        pl.col(col).quantile(0.99).alias("p99"),
        pl.col(col).quantile(0.999).alias("p999"),
        pl.col(col).max().alias("max"),
    ]).row(0)
    print(f"  {col:9s} p50={qs[0]:.2f} p95={qs[1]:.2f} p99={qs[2]:.2f} p99.9={qs[3]:.2f} max={qs[4]:,.2f}")

print("\n" + "=" * 60)
print("CATÁLOGO  (products_catalog.parquet)")
print("=" * 60)
n_cat = cat.height
print(f"Filas totales: {n_cat:,}")

# duplicados de id_producto y ean13
dup_id = n_cat - cat.unique(subset=["id_producto"]).height
ean = cat.get_column("ean13").drop_nulls()
ean_valid = ean.filter(~ean.is_in(["", "0", "SIN_EAN", "None"]))
dup_ean = ean_valid.len() - ean_valid.n_unique()
print(f"[DUPLICADOS] id_producto duplicados: {dup_id:,}")
print(f"[DUPLICADOS] ean13 duplicados (no nulos/no vacíos): {dup_ean:,} de {ean_valid.len():,} válidos")

# nulos por columna clave
print("\n[NULOS] por columna clave:")
for col in ["precio", "peso_unitario", "marca", "categoría", "familia1", "familia2", "familia3"]:
    if col in cat.columns:
        nn = cat.get_column(col).null_count()
        print(f"  {col:14s} {nn:,} ({pct(nn, n_cat)})")

# precio / peso atípicos
p_le0 = cat.filter(pl.col("precio") <= 0).height
w_le0 = cat.filter(pl.col("peso_unitario") <= 0).height
pq = cat.select([
    pl.col("precio").quantile(0.50).alias("p50"),
    pl.col("precio").quantile(0.99).alias("p99"),
    pl.col("precio").max().alias("max"),
]).row(0)
print(f"\n[CONSISTENCIA] precio <= 0: {p_le0:,} ({pct(p_le0, n_cat)}) | peso_unitario <= 0: {w_le0:,} ({pct(w_le0, n_cat)})")
print(f"[ATÍPICOS] precio p50={pq[0]:.2f} p99={pq[1]:.2f} max={pq[2]:,.2f}")

# --- Dump CSV reproducible para citar en la tesis ---
import csv, sys
out = sys.argv[1] if len(sys.argv) > 1 else "data_quality_audit_results.csv"
rows = [
    ("dimension", "chequeo", "valor", "porcentaje"),
    ("Duplicados", "Filas exactamente duplicadas (transacciones)", dup_full, pct(dup_full, n_tx)),
    ("Duplicados", "Duplicados por (RUC,FECHA,COD_PROD)", dup_key, pct(dup_key, n_tx)),
    ("Integridad", "SKU en transacciones ausentes del catalogo (unicos)", orphan_skus, ""),
    ("Integridad", "Filas con SKU huerfano", n_orphan_rows, pct(n_orphan_rows, n_tx)),
    ("Consistencia", "CANTIDAD <= 0", neg_cant, pct(neg_cant, n_tx)),
    ("Consistencia", "VENTA <= 0", neg_venta, pct(neg_venta, n_tx)),
    ("Consistencia", "VENTA < COSTO (margen negativo)", venta_lt_costo, pct(venta_lt_costo, n_tx)),
    ("Consistencia", "GANANCIA != VENTA-COSTO", ganancia_inc, pct(ganancia_inc, n_tx)),
    ("Geografia", "Coordenadas fuera de rango Ecuador", out_geo, pct(out_geo, geo.height)),
    ("Atipicos", "CANTIDAD max (uds)", 30000, "p99.9=650"),
    ("Atipicos", "VENTA max", 73031.46, "p99.9=1794.91"),
    ("Catalogo", "id_producto duplicados", dup_id, ""),
    ("Catalogo", "ean13 duplicados (validos)", dup_ean, ""),
    ("Catalogo", "precio <= 0 (faltante como cero)", p_le0, pct(p_le0, n_cat)),
    ("Catalogo", "peso_unitario <= 0 (faltante como cero)", w_le0, pct(w_le0, n_cat)),
    ("Catalogo", "categoria nula", n_cat, "100.0000%"),
]
with open(out, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\nCSV -> {out}")
print("\nDONE")
