# PermissionlessIV-CRISPR-Cancer-DeSci
Decentralized, Permissionless, and auditable CRISPR target discovery, annotation, and verification powered by Agentic AI/ML and BNB Blockchain. Open, reproducible pipeline for exploring glioblastoma (GBM) tumor heterogeneity using CRISPR gene essentiality data, advanced manifold learning (UMAP, PHATE, etc.) for powerful data analysis, AI agent annotation, and blockchain-based proof anchoring.**

---

## 🚀 Overview

This project enables permissionless, transparent research in cancer genomics by leveraging CRISPR functional genomics, state-of-the-art manifold learning, AI-driven literature annotation, and Web3 cryptographic proofs. The pipeline is designed for reproducibility, extensibility, and open science, embodying DeSci principles.

---

## 💡 Key Features

- **Functional Heterogeneity Analysis:**  
  Analyze GBM cell lines using CRISPR gene essentiality, not just expression data.
- **Manifold Learning:**  
  Visualize and cluster data with UMAP, PHATE, and more for deep biological insight.
- **Clustering:**  
  Identify novel GBM subtypes or dependencies via clustering in low-dimensional space.
- **Top Essential Genes:**  
  Discover and annotate high-priority genes with AI (LLM/Anthropic MCP).
- **Proof Anchoring:**  
  Generate SHA-256 cryptographic hashes and anchor key results on the BNB blockchain.
- **Extensible & Open:**  
  Framework for integrating multi-omics, VTE phenotypes, or federated privacy-preserving learning.

---

## 🗂️ Project Structure

```
permissionless-IV-crispr-gbm-deSci/
├── README.md
├── crispr_gbm_pipeline_v14.py
├── requirements.txt
├── LICENSE
├── data/
│   ├── CRISPRGeneEffect.csv
│   └── sample_info.csv
├── results/
│   ├── top10_essential_genes.csv
│   ├── top10_essential_genes_annotated.csv
│   ├── umap_gbm_clusters.png
│   ├── results_proof.txt
│   └── bnb_anchor_tx_hash.txt
```

---

## ⚙️ Usage

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Set Environment Variables**
- For LLM annotation:  
  `OPENAI_API_KEY`
- For BNB Chain anchoring:  
  `BNB_RPC_URL`, `BNB_PRIVATE_KEY`, `BNB_ACCOUNT_ADDRESS`

### 3. **Prepare Data**
- Place `CRISPRGeneEffect.csv` and `sample_info.csv` in the `data/` directory.

### 4. **Run Pipeline**
```bash
python crispr_gbm_pipeline_v14.py
```

---

## 🔍 Pipeline Steps

1. **Load Data:** Import CRISPR gene effect scores and sample metadata.
2. **GBM Filtering:** Select glioblastoma cell lines.
3. **Essentiality Analysis:** Compute and rank mean gene essentiality.
4. **Manifold Learning & Clustering:**  
   - UMAP, PHATE, or other methods for 2D projection.  
   - KMeans or other clustering for subtype discovery.
5. **Proof Generation:**  
   - SHA-256 hash output files.  
   - (Optional) Anchor hash to BNB Chain for tamper-evident record.
6. **LLM Annotation:**  
   - Use OpenAI or Anthropic MCP to annotate key genes.

---

## 🌟 Example Output

- **UMAP Plot:**  
  ![umap_gbm_clusters.png](results/umap_gbm_clusters.png)
- **Top Genes Table (Annotated):**
  | Gene | Essentiality_Score | LLM_Annotation |
  |------|--------------------|----------------|
  | ...  | ...                | ...            |

---

## 🛡️ DeSci & Reproducibility

- All results are cryptographically hashed and may be blockchain-anchored.
- The pipeline is modular and transparent; all code and annotations are open for verification and extension.

---

## 🧩 Extending the Pipeline

- Swap UMAP for PHATE, Multi-PHATE, MAGIC, or diffusion maps.
- Integrate clinical/VTE phenotypes or multi-modal omics.
- Use LLM/MCP for more advanced annotation, hypothesis generation, or sgRNA prioritization.
- Adapt for other cancer types or functional screens.

---

## 🤖 Credits

Created by [TarunSathya574](https://github.com/TarunSathya574) and contributors.  
Inspired by DeSci and open science.

---

## 📄 License

MIT License

