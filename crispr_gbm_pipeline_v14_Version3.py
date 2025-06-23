import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import openai
import os
import time
from web3 import Web3

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENABLE_LLM_ANNOTATION = True  # Set to False to skip LLM annotation
TOP_N_GENES = 10

# BNB Chain integration parameters
BNB_RPC = os.getenv("BNB_RPC_URL")  # e.g., "https://bsc-dataseed.binance.org/"
BNB_PRIVATE_KEY = os.getenv("BNB_PRIVATE_KEY")
BNB_ACCOUNT_ADDRESS = os.getenv("BNB_ACCOUNT_ADDRESS")
ENABLE_BNB_ANCHORING = True  # Set to False to skip BNB anchoring

if ENABLE_LLM_ANNOTATION and not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY as an environment variable.")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# --- DATA LOADING ---
print("Loading data...")
gene_effect = pd.read_csv("data/CRISPRGeneEffect.csv", index_col=0)
sample_info = pd.read_csv("data/sample_info.csv", index_col=0)

# --- FILTER FOR GBM CELL LINES ---
print("Filtering for GBM cell lines...")
gbm_samples = sample_info[sample_info['disease'].str.lower().str.contains("glioblastoma", na=False)].index
gene_effect_gbm = gene_effect[gbm_samples]

# --- ESSENTIALITY SCORING ---
print("Calculating gene essentiality...")
essentiality_scores = gene_effect_gbm.mean(axis=1)
top_genes = essentiality_scores.nsmallest(TOP_N_GENES)
top_genes_df = top_genes.reset_index().rename(columns={'index': 'Gene', 0: 'Essentiality_Score'})
top_genes_df.to_csv("results/top10_essential_genes.csv", index=False)

# --- CLUSTERING & UMAP ---
print("Performing UMAP and clustering...")
umap_model = umap.UMAP(random_state=0)
umap_embedding = umap_model.fit_transform(gene_effect_gbm.T)
kmeans = KMeans(n_clusters=3, random_state=0).fit(umap_embedding)
clusters = kmeans.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(x=umap_embedding[:,0], y=umap_embedding[:,1], hue=clusters, palette="Set1")
plt.title("GBM Cell Line Clusters (UMAP)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(title="Cluster")
plt.savefig("results/umap_gbm_clusters.png")
plt.close()

# --- CRYPTOGRAPHIC PROOF ---
print("Generating cryptographic proof of results...")
def sha256_of_file(filename):
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

proofs = {}
for fname in ["results/top10_essential_genes.csv", "results/umap_gbm_clusters.png"]:
    proofs[fname] = sha256_of_file(fname)
proof_text = ""
for k, v in proofs.items():
    proof_text += f"{k}: {v}\n"
with open("results/results_proof.txt", "w") as f:
    f.write(proof_text)

# --- BNB CHAIN ANCHORING ---
def anchor_proof_on_bnb(proof_text):
    if not (BNB_RPC and BNB_PRIVATE_KEY and BNB_ACCOUNT_ADDRESS):
        print("BNB Chain credentials not set. Skipping anchoring.")
        return None
    try:
        w3 = Web3(Web3.HTTPProvider(BNB_RPC))
        if not w3.is_connected():
            print("Failed to connect to BNB RPC endpoint.")
            return None
        nonce = w3.eth.get_transaction_count(BNB_ACCOUNT_ADDRESS)
        # Only use first 128 bytes for data, due to gas limits
        data_bytes = proof_text.encode('utf-8')[:128]
        txn = {
            'nonce': nonce,
            'to': BNB_ACCOUNT_ADDRESS,
            'value': 0,
            'gas': 21000 + 68 * len(data_bytes),
            'gasPrice': w3.to_wei('5', 'gwei'),
            'data': data_bytes
        }
        signed_txn = w3.eth.account.sign_transaction(txn, BNB_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        tx_hash_hex = w3.to_hex(tx_hash)
        print("Proof anchored on BNB Chain! TX hash:", tx_hash_hex)
        # Save the tx hash
        with open("results/bnb_anchor_tx_hash.txt", "w") as f:
            f.write(tx_hash_hex + "\n")
        return tx_hash_hex
    except Exception as e:
        print("Error anchoring proof on BNB:", e)
        return None

if ENABLE_BNB_ANCHORING:
    print("Anchoring proof to BNB Chain...")
    anchor_proof_on_bnb(proof_text)

# --- LLM ANNOTATION (INTEGRATED) ---
def annotate_gene_with_llm(gene_symbol, cancer_type="glioblastoma"):
    prompt = (
        f"Summarize the known or suspected role of the gene {gene_symbol} in {cancer_type}. "
        "Mention evidence from PubMed or GeneCards if possible. "
        "If little is known, say so. List PubMed IDs (PMIDs) or GeneCards links if available."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

if ENABLE_LLM_ANNOTATION:
    print("Annotating top genes with LLM...")
    annotations = []
    for gene in top_genes.index:
        print(f"Annotating {gene}...")
        annotation = annotate_gene_with_llm(gene)
        annotations.append(annotation)
        time.sleep(1.5)  # To avoid API rate limits
    top_genes_df["LLM_Annotation"] = annotations
    top_genes_df.to_csv("results/top10_essential_genes_annotated.csv", index=False)
    print("LLM annotation complete: see results/top10_essential_genes_annotated.csv")

print("Pipeline complete.")