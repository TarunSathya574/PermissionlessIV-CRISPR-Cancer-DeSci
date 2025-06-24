"""
Query Blockchain for Gene Data
Simple script to demonstrate querying the deployed smart contracts
"""

import json
import sys
sys.path.append('.')
from upload_results import BlockchainUploader

def query_gene_interactive():
    """Interactive gene query interface"""
    print("🧬 GBM Gene Blockchain Query Tool")
    print("=" * 40)
    
    uploader = BlockchainUploader()
    
    while True:
        print("\nOptions:")
        print("1. Query specific gene")
        print("2. Check contract status")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            gene_name = input("Enter gene name (e.g., EGFR): ").strip().upper()
            if gene_name:
                print(f"\n🔍 Querying {gene_name} from blockchain...")
                
                # Try BNB Chain query
                gene_data = uploader.query_gene_from_bnb(gene_name)
                
                if gene_data:
                    print(f"✅ Found {gene_name} on BNB Chain:")
                    print(f"   Essentiality Score: {gene_data['essentiality_score']}")
                    print(f"   Consistency: {gene_data['consistency_pct']}%")
                    print(f"   Cell Lines: {gene_data['cell_line_count']}")
                    print(f"   Known GBM Gene: {gene_data['is_known_gbm_gene']}")
                    print(f"   Data Hash: {gene_data['data_hash']}")
                else:
                    print(f"❌ Gene {gene_name} not found in blockchain registry")
            else:
                print("Please enter a valid gene name")
                
        elif choice == "2":
            print("\n📊 Contract Status:")
            if uploader.web3 and uploader.web3.is_connected():
                print("✅ BNB Chain: Connected")
                print(f"   Network: {uploader.web3.eth.chain_id}")
                print(f"   Contract: {uploader.config['bnb']['contract_address'] or 'Not configured'}")
            else:
                print("❌ BNB Chain: Not connected")
            
            print("ℹ️  Supra: Integration ready")
            
        elif choice == "3":
            print("👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def demo_queries():
    """Demo queries for common genes"""
    print("🧬 Demo Gene Queries")
    print("=" * 30)
    
    uploader = BlockchainUploader()
    demo_genes = ["EGFR", "TP53", "PTEN", "IDH1", "ATRX"]
    
    for gene in demo_genes:
        print(f"\n🔍 Querying {gene}...")
        
        # Simulate query result (since we might not have deployed contracts)
        demo_result = {
            "gene_name": gene,
            "essentiality_score": -1.5 - (len(gene) * 0.1),  # Demo calculation
            "consistency_pct": 85 + (len(gene) % 15),
            "cell_line_count": 40 + (len(gene) % 10),
            "is_known_gbm_gene": gene in ["EGFR", "TP53", "PTEN"],
            "data_hash": f"0x{hash(gene) % 10000:04x}..."
        }
        
        print(f"   Score: {demo_result['essentiality_score']:.2f}")
        print(f"   Consistency: {demo_result['consistency_pct']}%")
        print(f"   Known GBM: {demo_result['is_known_gbm_gene']}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_queries()
    else:
        query_gene_interactive()
