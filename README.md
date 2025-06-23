# PermissionlessIV-CRISPR-Cancer-DeSci
Decentralized, Permissionless, and auditable CRISPR target discovery, annotation, and verification powered by Agentic AI/ML and BNB Blockchain. Open, reproducible pipeline for exploring glioblastoma (GBM) tumor heterogeneity using CRISPR gene essentiality data, advanced manifold learning (UMAP, PHATE, etc.) for powerful data analysis, AI agent annotation, and blockchain-based proof anchoring.**

---

# Permissionless IV Hackathon: BNB DeSci Track

**Open, reproducible pipeline for CRISPR-based GBM heterogeneity research, built for the Permissionless IV Hackathon (BNB DeSci Track).**

---

## Overview

This repository contains a transparent and auditable pipeline for exploring glioblastoma (GBM) tumor heterogeneity using CRISPR gene essentiality data, UMAP/PHATE manifold learning, AI annotation, and blockchain-based proof anchoring (with Merkle tree and simulated ZKP support). Built for the BNB DeSci track at Permissionless IV.

---

## Structure

```
.
├── README.md
├── crispr_gbm_pipeline_v15.py
├── requirements.txt
├── LICENSE
├── data/
│   ├── CRISPRGeneEffect.csv
│   └── sample_info.csv
├── results/
│   ├── top10_essential_genes.csv
│   ├── top10_essential_genes_annotated.csv
│   ├── umap_gbm_clusters.png
│   ├── results_hashes.json
│   ├── merkle_tree.json
│   ├── zkp_proof.json
│   ├── bnb_anchor_tx_hash.txt
│   └── .gitkeep
```

---

## Quickstart

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Set environment variables for LLM and BNB Chain anchoring as needed:**
    - `OPENAI_API_KEY` (optional, for AI annotation)
    - `BNB_RPC_URL`, `BNB_PRIVATE_KEY`, `BNB_ACCOUNT_ADDRESS` (for BNB Chain proof anchoring)

3. **Place data files in the `data/` directory (see sample CSVs).**

4. **Run the pipeline:**
    ```bash
    python crispr_gbm_pipeline_v15.py
    ```

---

## Features

- **Safe, chunked data loading** and GBM filtering
- **Gene scoring:** Calculates gene essentiality across GBM cell lines
- **Manifold Learning:** PCA, UMAP for visualization; KMeans clustering for heterogeneity
- **AI Annotation:** Annotates top essential genes with GPT (if OpenAI API key provided)
- **Merkle Tree Proof:** Computes file hashes for all results and creates a Merkle root for tamper-evident proofs
- **Simulated Zero-Knowledge Proof (ZKP):** Simulates a ZKP of Merkle root knowledge (for demonstration)
- **Blockchain Anchoring:** Anchors Merkle root and ZKP on BNB Smart Chain (if enabled)
- **Reproducible and auditable:** All proof files are written to the `results/` directory

---

## Example Outputs

- `results/top10_essential_genes.csv`: Table of top essential genes
- `results/top10_essential_genes_annotated.csv`: Same table, with AI-generated annotations
- `results/umap_gbm_clusters.png`: UMAP projection of GBM cell lines
- `results/results_hashes.json`: SHA-256 hashes of all result files
- `results/merkle_tree.json`: Merkle root and tree structure for result hashes
- `results/zkp_proof.json`: Simulated ZKP for Merkle root knowledge
- `results/bnb_anchor_tx_hash.txt`: BNB Chain transaction hash (if enabled)

---

## License

MIT License

---

## Notes

- **AI annotation is optional:** If you do not provide an OpenAI API key, annotation will be skipped.
- **Blockchain anchoring is optional:** Set `ANCHOR_ON_CHAIN=True` in the script to enable. Requires BNB account and sufficient gas.
- **Merkle tree/ZKP:** This implementation uses a basic Merkle tree and simulates the ZKP (for demonstration). For production, integrate with a real ZKP system.

---


