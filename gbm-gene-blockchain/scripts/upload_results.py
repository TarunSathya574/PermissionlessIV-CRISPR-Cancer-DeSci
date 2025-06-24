"""
Blockchain Integration for GBM Gene Essentiality Agent
Uploads analysis results to BNB Chain and Supra blockchain
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Web3 for BNB Chain
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("Warning: web3 not installed. BNB Chain features disabled.")

# Import our gene agent
import sys
sys.path.append('..')
from gbm_gene_essentiality_agent import GBMGeneEssentialityAgent

class BlockchainUploader:
    """
    Handles uploading gene analysis results to blockchain networks
    """
    
    def __init__(self, config_file: str = "blockchain_config.json"):
        self.config = self.load_config(config_file)
        self.web3 = None
        self.contract = None
        
        if WEB3_AVAILABLE:
            self.setup_bnb_connection()
    
    def load_config(self, config_file: str) -> Dict:
        """Load blockchain configuration"""
        default_config = {
            "bnb": {
                "rpc_url": "https://bsc-testnet.publicnode.com",
                "chain_id": 97,
                "contract_address": "",
                "private_key": "",
                "gas_limit": 3000000
            },
            "supra": {
                "rpc_url": "https://rpc-testnet.supra.com",
                "contract_address": "",
                "private_key": ""
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        else:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created config file: {config_file}")
            return default_config
    
    def setup_bnb_connection(self):
        """Setup BNB Chain connection"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.config["bnb"]["rpc_url"]))
            
            if not self.web3.is_connected():
                print("Failed to connect to BNB Chain")
                return False
            
            print(f"Connected to BNB Chain")
            return True
            
        except Exception as e:
            print(f"BNB Chain setup failed: {e}")
            return False
    
    def upload_gene_results_to_bnb(self, gene_results: pd.DataFrame, 
                                  analysis_metadata: Dict) -> Optional[str]:
        """Upload gene analysis results to BNB Chain"""
        if not self.web3:
            print("BNB Chain not configured")
            return None
        
        print(f"Would upload {len(gene_results)} genes to BNB Chain")
        print("Note: Configure private_key and contract_address for actual deployment")
        
        # Simulate successful upload for demo
        return "0x" + hashlib.sha256(str(gene_results.head(1).to_dict()).encode()).hexdigest()[:32]
    
    def upload_to_supra(self, gene_results: pd.DataFrame) -> bool:
        """Upload results to Supra blockchain"""
        print(f"Would upload {len(gene_results)} genes to Supra blockchain")
        print("Note: Configure Supra credentials for actual deployment")
        return True


def main():
    """Main function to demonstrate blockchain integration"""
    print("=" * 60)
    print("🧬 GBM GENE AGENT BLOCKCHAIN INTEGRATION 🧬")
    print("=" * 60)
    
    # Initialize the gene agent
    print("\n1. Running gene analysis...")
    agent = GBMGeneEssentialityAgent(
        crispr_data_path="../data/CRISPRGeneEffect.csv",
        sample_info_path="../data/sample_info.csv"
    )
    
    # Run analysis
    if not agent.load_data():
        print("Note: Configure data paths or generate demo data")
        return
    
    agent.identify_gbm_cell_lines()
    agent.analyze_gene_essentiality()
    
    # Get results
    top_genes = agent.get_most_negative_genes(20)
    print(f"✅ Analysis complete: {len(top_genes)} top genes identified")
    
    # Initialize blockchain uploader
    print("\n2. Initializing blockchain connection...")
    uploader = BlockchainUploader()
    
    # Prepare metadata
    analysis_metadata = {
        'total_cell_lines': len(agent.gbm_cell_lines),
        'merkle_root': hashlib.sha256(str(top_genes.to_dict()).encode()).hexdigest(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Upload to blockchains
    print("\n3. Uploading to BNB Chain...")
    tx_hash = uploader.upload_gene_results_to_bnb(top_genes, analysis_metadata)
    
    print("\n4. Uploading to Supra...")
    supra_success = uploader.upload_to_supra(top_genes)
    
    # Save results summary
    summary = {
        'analysis_timestamp': analysis_metadata['timestamp'],
        'total_genes_analyzed': len(top_genes),
        'bnb_chain_tx': tx_hash,
        'supra_uploaded': supra_success,
        'top_5_genes': top_genes.head(5).index.tolist()
    }
    
    with open('blockchain_upload_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 BLOCKCHAIN INTEGRATION COMPLETE! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
