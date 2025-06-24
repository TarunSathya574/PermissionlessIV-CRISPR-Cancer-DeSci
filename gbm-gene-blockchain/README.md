# GBM Gene Blockchain Project

Simple blockchain deployment for the GBM Gene Essentiality Agent, supporting both BNB Chain and Supra blockchain.

## Project Structure

```
gbm-gene-blockchain/
├── contracts/
│   ├── bnb/
│   │   ├── GeneRegistry.sol      # Solidity contract for BNB Chain
│   │   └── deploy.js             # Deployment script
│   └── supra/
│       ├── gene_registry.move    # Move module for Supra
│       └── deploy.sh             # Deployment script
├── scripts/
│   └── upload_results.py         # Python script to upload analysis results
├── frontend/
│   ├── index.html               # Simple web interface
│   └── app.js                   # Frontend JavaScript
├── package.json
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install web3 pandas numpy
```

### 2. Configure Blockchain Credentials

Create `scripts/blockchain_config.json`:

```json
{
  "bnb": {
    "rpc_url": "https://bsc-testnet.publicnode.com",
    "chain_id": 97,
    "contract_address": "YOUR_DEPLOYED_CONTRACT_ADDRESS",
    "private_key": "YOUR_PRIVATE_KEY",
    "gas_limit": 3000000
  },
  "supra": {
    "rpc_url": "https://rpc-testnet.supra.com",
    "contract_address": "YOUR_SUPRA_ADDRESS",
    "private_key": "YOUR_SUPRA_PRIVATE_KEY"
  }
}
```

### 3. Deploy Contracts

#### BNB Chain
```bash
npm run deploy-bnb
```

#### Supra
```bash
export SUPRA_ACCOUNT_ADDRESS=0x123...
export SUPRA_PRIVATE_KEY=your_key
npm run deploy-supra
```

### 4. Upload Gene Data

```bash
npm run upload
```

### 5. Launch Web Interface

```bash
npm run serve
# Visit http://localhost:8000
```

## Smart Contract Functions

### BNB Chain (Solidity)

- `storeGeneData()` - Store gene essentiality results
- `getGeneData()` - Query specific gene information
- `submitAnalysis()` - Submit complete analysis results
- `verifyAnalysis()` - Verify analysis integrity
- `authorizeResearcher()` - Manage researcher permissions

### Supra (Move)

- `store_gene_data()` - Store gene data
- `get_gene_data()` - Query gene information
- `submit_analysis()` - Submit analysis
- `verify_analysis()` - Verify results
- `authorize_researcher()` - Manage permissions

## Gene Data Structure

```solidity
struct GeneData {
    string geneName;
    int256 essentialityScore;  // Scaled by 1000
    uint256 consistencyPct;    // Percentage 0-100
    uint256 cellLineCount;     // Essential cell lines count
    bool isKnownGbmGene;       // Known GBM gene flag
    uint256 timestamp;         // Upload timestamp
    string dataHash;           // Data integrity hash
}
```

## Usage Examples

### Query Gene from Blockchain

```python
from scripts.upload_results import BlockchainUploader

uploader = BlockchainUploader()
gene_data = uploader.query_gene_from_bnb("EGFR")
print(f"EGFR essentiality score: {gene_data['essentiality_score']}")
```

### Upload Analysis Results

```python
# Run gene analysis
agent = GBMGeneEssentialityAgent("data/CRISPRGeneEffect.csv")
results = agent.run_complete_analysis()

# Upload to blockchain
uploader = BlockchainUploader()
uploader.upload_gene_results_to_bnb(results['top_essential_genes'])
```

## Web Interface Features

- **Network Selection**: Switch between BNB Chain and Supra
- **Gene Search**: Query specific genes from blockchain
- **Top Genes**: View most essential genes
- **Demo Upload**: Test contract functionality

## Development

### Add New Gene Analysis

1. Extend `GeneRegistry.sol` with new data fields
2. Update Python upload script
3. Deploy updated contract
4. Test with new data structure

### Cross-Chain Integration

The project supports both EVM (BNB Chain) and Move (Supra) blockchains:

- **BNB Chain**: Fast, low-cost transactions
- **Supra**: High-performance blockchain with Move language

## Security Considerations

- Only authorized researchers can upload data
- All analysis results include cryptographic hashes
- Smart contracts include access control mechanisms
- Private keys should never be committed to version control

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Support

For questions about:
- Smart contract deployment: Check contract documentation
- Gene analysis: Refer to GBM Gene Agent documentation
- Blockchain integration: Review upload scripts

---

**Note**: This is a simplified implementation for demonstration. Production deployment requires additional security audits, testing, and optimization.