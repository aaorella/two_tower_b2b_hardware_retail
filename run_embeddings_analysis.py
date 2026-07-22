import numpy as np
import polars as pl
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("--- STEP 1: LOADING DATA ---")
    embeddings = np.load("models/candidate_embeddings.npy")
    candidate_ids = np.load("models/candidate_ids.npy", allow_pickle=True)
    catalog = pl.read_parquet("data_processed/products_catalog.parquet").unique(subset=["id_producto"])
    
    print(f"Embeddings loaded: {embeddings.shape}")
    print(f"Candidate IDs loaded: {candidate_ids.shape}")
    print(f"Catalog loaded: {catalog.shape}")
    
    # Impute catalog nulls
    catalog = catalog.with_columns([
        pl.col("familia1").fill_null("SIN_CATEGORIA"),
        pl.col("familia2").fill_null("SIN_SUBCATEGORIA"),
        pl.col("marca").fill_null("SIN_MARCA")
    ])
    
    # Align embeddings with catalog
    # candidate_ids has the order of the rows in the embeddings matrix.
    # We create a dataframe preserving the order of candidate_ids.
    ids_df = pl.DataFrame({
        "id_producto": candidate_ids,
        "embedding_idx": np.arange(len(candidate_ids))
    })
    
    # Join with catalog metadata
    aligned_df = ids_df.join(catalog, on="id_producto", how="left")
    print(f"Aligned DataFrame shape: {aligned_df.shape}")
    
    # Fill any metadata nulls resulting from missing catalog items
    aligned_df = aligned_df.with_columns([
        pl.col("familia1").fill_null("SIN_CATEGORIA"),
        pl.col("familia2").fill_null("SIN_SUBCATEGORIA"),
        pl.col("descripción corta").fill_null("PRODUCTO_DESCONOCIDO")
    ])
    
    print("\n--- STEP 2: RUNNING DIMENSIONALITY REDUCTION ---")
    
    # 2.1 t-SNE
    print("Running t-SNE (scikit-learn)...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, n_jobs=-1)
    coords_tsne = tsne.fit_transform(embeddings)
    np.save("models/coords_tsne.npy", coords_tsne)
    print("t-SNE completed.")
    
    # 2.2 UMAP
    print("Running UMAP (umap-learn)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, n_components=2, random_state=42)
    coords_umap = reducer.fit_transform(embeddings)
    np.save("models/coords_umap.npy", coords_umap)
    print("UMAP completed.")
    
    print("\n--- STEP 3: PLOTTING MACRO-CLUSTERS ---")
    
    # Get labels for colors
    families = aligned_df["familia1"].to_list()
    # Map to top-level colors for visual clarity
    unique_fams = sorted(list(set(families)))
    print("Families found:", unique_fams)
    
    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=coords_tsne[:, 0], y=coords_tsne[:, 1],
        hue=families,
        palette="tab20",
        alpha=0.6,
        edgecolor=None,
        s=10
    )
    plt.title("Visualización de Embeddings del Catálogo usando t-SNE\n(Coloreado por Familia de Nivel Superior)", fontsize=14, fontweight="bold")
    plt.xlabel("Dimensión t-SNE 1")
    plt.ylabel("Dimensión t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Familia 1")
    plt.tight_layout()
    plt.savefig("models/tsne_families.png", dpi=300)
    plt.close()
    print("Saved models/tsne_families.png")
    
    # Plot UMAP
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=coords_umap[:, 0], y=coords_umap[:, 1],
        hue=families,
        palette="tab20",
        alpha=0.6,
        edgecolor=None,
        s=10
    )
    plt.title("Visualización de Embeddings del Catálogo usando UMAP\n(Coloreado por Familia de Nivel Superior)", fontsize=14, fontweight="bold")
    plt.xlabel("Dimensión UMAP 1")
    plt.ylabel("Dimensión UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Familia 1")
    plt.tight_layout()
    plt.savefig("models/umap_families.png", dpi=300)
    plt.close()
    print("Saved models/umap_families.png")
    
    print("\n--- STEP 4: CONSTRUCTIVE CLUSTER ANALYSIS (QUALITATIVE/QUANTITATIVE) ---")
    
    # Let's search for representative products of 3 clusters
    # PVC/Plumbing, Masonry/Cement, and Electrical.
    
    # PVC Plumbing Group
    # Find a tube, a connection/elbow, and pvc cement/glue
    pvc_tube = aligned_df.filter(
        pl.col("familia2") == "TUBERIAS Y ACCESORIOS",
        pl.col("descripción corta").str.contains("(?i)tubo.*pvc|tubo.*desag")
    ).head(3)
    
    pvc_elbow = aligned_df.filter(
        pl.col("familia2") == "TUBERIAS Y ACCESORIOS",
        pl.col("descripción corta").str.contains("(?i)codo|union|tee")
    ).head(3)
    
    pvc_glue = aligned_df.filter(
        pl.col("familia2") == "PEGAMENTOS Y ADHESIVOS",
        pl.col("descripción corta").str.contains("(?i)soldadura.*pvc|pega.*pvc")
    ).head(3)
    
    # Cements & Masonry Group
    cement = aligned_df.filter(
        pl.col("descripción corta").str.contains("(?i)cemento.*gris|cemento.*chimborazo|cemento.*selvalegre")
    ).head(3)
    
    rebar = aligned_df.filter(
        pl.col("familia2") == "TREFILADOS Y MALLAS",
        pl.col("descripción corta").str.contains("(?i)alambre|malla|varilla")
    ).head(3)
    
    sika = aligned_df.filter(
        pl.col("familia2") == "ADITIVO CONSTRUCCION",
        pl.col("descripción corta").str.contains("(?i)sika")
    ).head(3)
    
    # Electrical Group
    cable = aligned_df.filter(
        pl.col("familia2") == "MATERIAL ELECTRICO",
        pl.col("descripción corta").str.contains("(?i)cable|alambre.*gemelo|conductor")
    ).head(3)
    
    socket = aligned_df.filter(
        pl.col("familia2") == "MATERIAL ELECTRICO",
        pl.col("descripción corta").str.contains("(?i)tomacorriente|interruptor|boquilla")
    ).head(3)
    
    tape = aligned_df.filter(
        pl.col("familia2") == "MATERIAL ELECTRICO",
        pl.col("descripción corta").str.contains("(?i)cinta")
    ).head(3)
    
    # Combine selected representatives (take the first found of each)
    selected_items = []
    
    # Helper to append if exists
    def add_item(df, label, group_name):
        if df.height > 0:
            selected_items.append({
                "id": df["id_producto"][0],
                "name": df["descripción corta"][0],
                "embedding_idx": df["embedding_idx"][0],
                "group": group_name,
                "label": label
            })
            
    add_item(pvc_tube, "Tubo PVC", "Plumbing")
    add_item(pvc_elbow, "Codo PVC", "Plumbing")
    add_item(pvc_glue, "Pega PVC", "Plumbing")
    
    add_item(cement, "Cemento", "Masonry")
    add_item(rebar, "Hierro/Alambre", "Masonry")
    add_item(sika, "Aditivo Sika", "Masonry")
    
    add_item(cable, "Cable", "Electrical")
    add_item(socket, "Tomacorriente", "Electrical")
    add_item(tape, "Cinta", "Electrical")
    
    print("\nSelected representatives for Cosine Similarity Analysis:")
    for item in selected_items:
        print(f" - [{item['group']} - {item['label']}] {item['id']}: {item['name']}")
        
    # Extract embeddings for these items
    indices = [item["embedding_idx"] for item in selected_items]
    names = [f"{item['group']} - {item['label']}" for item in selected_items]
    
    selected_embs = embeddings[indices]
    
    # Compute Cosine Similarity Matrix
    norms = np.linalg.norm(selected_embs, axis=1, keepdims=True)
    norm_embs = selected_embs / norms
    similarity_matrix = np.dot(norm_embs, norm_embs.T)
    
    # Plot Similarity Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        cmap="coolwarm",
        vmin=0.0, vmax=1.0,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Matriz de Similitud Coseno de Embeddings\n(Validación de Clusters Constructivos)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("models/constructive_similarity_heatmap.png", dpi=300)
    plt.close()
    print("Saved models/constructive_similarity_heatmap.png")
    
    # Build dynamic group index mapping
    groups = {}
    for idx, item in enumerate(selected_items):
        g_name = item["group"]
        if g_name not in groups:
            groups[g_name] = []
        groups[g_name].append(idx)
    
    print("\n--- QUANTITATIVE SHIELD METRICS ---")
    
    # Intra-group similarities
    for group_name, idxs in groups.items():
        if len(idxs) < 2:
            print(f"Skipping INTRA similarity for {group_name} (needs at least 2 items, got {len(idxs)})")
            continue
        intra_sims = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                intra_sims.append(similarity_matrix[idxs[i], idxs[j]])
        print(f"Mean INTRA-cluster Similarity for {group_name}: {np.mean(intra_sims):.4f}")
        
    # Inter-group similarities
    inter_sims = []
    group_names = list(groups.keys())
    for idx1 in range(len(group_names)):
        for idx2 in range(idx1 + 1, len(group_names)):
            g1_name = group_names[idx1]
            g2_name = group_names[idx2]
            idxs1 = groups[g1_name]
            idxs2 = groups[g2_name]
            
            cross_vals = []
            for i in idxs1:
                for j in idxs2:
                    cross_vals.append(similarity_matrix[i, j])
            print(f"Mean INTER-cluster Similarity ({g1_name} vs {g2_name}): {np.mean(cross_vals):.4f}")
            inter_sims.extend(cross_vals)
            
    if inter_sims:
        print(f"Overall Mean INTER-cluster Similarity: {np.mean(inter_sims):.4f}")
    
    print("\nEmbedding analysis run successfully!")

if __name__ == "__main__":
    main()
