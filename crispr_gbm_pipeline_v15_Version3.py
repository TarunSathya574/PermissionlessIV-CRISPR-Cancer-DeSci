"""
CRISPRChain Pipeline v15: Core v14 pipeline PLUS Merkle Tree, Simulated ZKP, and AI Annotation.

- Performs CRISPR analysis: loading, filtering, scoring, manifold learning, clustering, and AI annotation.
- Annotates top essential genes using OpenAI GPT (if API key is set).
- Calculates Merkle root for all result files in 'results/'.
- Simulates a ZKP (proof-of-knowledge of the Merkle root hash).
- Anchors both Merkle root and ZKP on BNB Chain (if enabled).
"""

import os
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

# AI annotation (OpenAI)
try:
    import openai
    ANNOTATION_ENABLED = True
except ImportError:
    ANNOTATION_ENABLED = False

# Blockchain imports
from web3 import Web3

# ---- Parameters ----
RESULTS_DIR = "results"
DATA_DIR = "data"
ANCHOR_ON_CHAIN = False  # Set to True to anchor on BNB Chain

# Blockchain parameters (fill with your own for mainnet/testnet as appropriate)
BNB_RPC_URL = os.getenv("BNB_RPC_URL", "")
BNB_PRIVATE_KEY = os.getenv("BNB_PRIVATE_KEY", "")
BNB_ACCOUNT_ADDRESS = os.getenv("BNB_ACCOUNT_ADDRESS", "")

# ---- Utility Functions ----

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def merkle_parent(a, b):
    return hashlib.sha256((a + b).encode()).hexdigest()

def merkle_tree(leaves):
    """Builds the Merkle tree; returns root hash and full tree as list of levels."""
    current_level = leaves[:]
    tree = [current_level]
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i+1] if i + 1 < len(current_level) else current_level[i]
            next_level.append(merkle_parent(left, right))
        current_level = next_level
        tree.insert(0, current_level)
    return current_level[0], tree

def simulate_zkp(merkle_root):
    """
    Simulates a zero-knowledge proof for knowledge of the Merkle root.
    In real-world use, integrate with a ZKP library (e.g., zk-SNARKs, zk-STARKs).
    Here, we just simulate a "proof" as a hash of the root + a nonce.
    """
    nonce = os.urandom(16).hex()
    fake_proof = hashlib.sha256((merkle_root + nonce).encode()).hexdigest()
    return {"nonce": nonce, "proof": fake_proof}

def anchor_on_bnb_chain(data_hash, zkp_proof=None):
    """Send transaction with proof to BNB Chain."""
    if not BNB_RPC_URL or not BNB_PRIVATE_KEY or not BNB_ACCOUNT_ADDRESS:
        print("BNB anchoring not configured properly.")
        return None

    w3 = Web3(Web3.HTTPProvider(BNB_RPC_URL))
    nonce = w3.eth.get_transaction_count(BNB_ACCOUNT_ADDRESS)

    tx_data = {
        "merkle_root": data_hash,
        "zkp": zkp_proof
    }
    tx = {
        "nonce": nonce,
        "to": BNB_ACCOUNT_ADDRESS,  # self-send
        "value": 0,
        "gas": 21000 + 1000 * len(json.dumps(tx_data)),  # rough estimate
        "gasPrice": w3.to_wei("5", "gwei"),
        "data": w3.to_hex(text=json.dumps(tx_data))[:1024],  # max 1024 chars
        "chainId": w3.eth.chain_id
    }
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=BNB_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return w3.to_hex(tx_hash)

def annotate_genes_openai(genes, openai_api_key=None):
    """
    Annotate a list of genes using GPT (if OpenAI API key provided).
    Returns a list of dictionaries with 'gene' and 'annotation' keys.
    """
    if not ANNOTATION_ENABLED or openai_api_key is None:
        return [{"gene": g, "annotation": "Annotation not performed."} for g in genes]
    openai.api_key = openai_api_key
    annotations = []
    for gene in genes:
        prompt = f"Provide a one-sentence summary of the gene {gene} in the context of glioblastoma."
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            summary = resp["choices"][0]["message"]["content"]
        except Exception as e:
            summary = f"Annotation failed: {e}"
        annotations.append({"gene": gene, "annotation": summary})
    return annotations

# ---- Main Analysis Pipeline ----

if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    # 1. Data loading & filtering
    crispr = pd.read_csv(f"{DATA_DIR}/CRISPRGeneEffect.csv", index_col=0)
    sample_info = pd.read_csv(f"{DATA_DIR}/sample_info.csv", index_col=0)
    gbm_samples = sample_info[sample_info["disease"].str.contains("glioblastoma", case=False)]["cell_line"].tolist()
    gbm_crispr = crispr[gbm_samples]

    # 2. Score genes (mean essentiality across GBM samples)
    gene_scores = gbm_crispr.mean(axis=1).sort_values()
    top_10 = gene_scores.head(10)
    top_10.to_csv(f"{RESULTS_DIR}/top10_essential_genes.csv")

    # 3. Annotate top genes (optional, OpenAI)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    annotations = annotate_genes_openai(top_10.index.tolist(), openai_api_key=OPENAI_API_KEY)
    annotated_df = pd.DataFrame(annotations)
    annotated_df["score"] = top_10.values
    annotated_df.to_csv(f"{RESULTS_DIR}/top10_essential_genes_annotated.csv", index=False)

    # 4. Manifold learning & clustering
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(gbm_crispr.fillna(0).T)
    umap_result = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(pca_result)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(umap_result)
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=umap_result[:,0], y=umap_result[:,1], hue=kmeans.labels_, palette="tab10")
    plt.title("UMAP Clusters of GBM Cell Lines")
    plt.savefig(f"{RESULTS_DIR}/umap_gbm_clusters.png", bbox_inches="tight")
    plt.close()

    # 5. Collect result files and compute their hashes
    result_files = sorted(glob.glob(f"{RESULTS_DIR}/*"))
    file_hashes = [sha256_file(f) for f in result_files]
    file_hashes_map = dict(zip([os.path.basename(f) for f in result_files], file_hashes))
    with open(f"{RESULTS_DIR}/results_hashes.json", "w") as f:
        json.dump(file_hashes_map, f, indent=2)

    # 6. Build Merkle tree of result file hashes
    merkle_root, merkle_tree_levels = merkle_tree(file_hashes)
    with open(f"{RESULTS_DIR}/merkle_tree.json", "w") as f:
        json.dump({"merkle_root": merkle_root, "tree_levels": merkle_tree_levels}, f, indent=2)

    # 7. Simulate ZKP for Merkle root knowledge
    zkp = simulate_zkp(merkle_root)
    with open(f"{RESULTS_DIR}/zkp_proof.json", "w") as f:
        json.dump(zkp, f, indent=2)

    # 8. Anchor on blockchain (optional)
    if ANCHOR_ON_CHAIN:
        tx_hash = anchor_on_bnb_chain(merkle_root, zkp)
        if tx_hash:
            with open(f"{RESULTS_DIR}/bnb_anchor_tx_hash.txt", "w") as f:
                f.write(tx_hash)
            print(f"Anchored proof to BNB Chain! TX hash: {tx_hash}")
        else:
            print("Blockchain anchoring failed.")
    else:
        print("Blockchain anchoring skipped (set ANCHOR_ON_CHAIN=True to enable).")

    # 9. Print summary
    print(f"Merkle Root: {merkle_root}")
    print(f"Simulated ZKP: {zkp}")
    print("AI-annotated gene list:")
    print(annotated_df)
    print("Pipeline complete. All results + proofs are in the 'results/' directory.")