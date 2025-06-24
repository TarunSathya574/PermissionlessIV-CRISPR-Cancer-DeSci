// Simple frontend app for GBM Gene Registry
let currentNetwork = 'bnb';
let web3 = null;
let contract = null;

// Demo data for testing
const demoGenes = [
    { name: 'EGFR', score: -1.85, consistency: 95, cellLines: 48, isKnown: true },
    { name: 'TP53', score: -1.72, consistency: 88, cellLines: 44, isKnown: true },
    { name: 'PTEN', score: -1.68, consistency: 92, cellLines: 46, isKnown: true },
    { name: 'IDH1', score: -1.45, consistency: 78, cellLines: 39, isKnown: true },
    { name: 'ATRX', score: -1.38, consistency: 82, cellLines: 41, isKnown: true }
];

// Network selection
function selectNetwork(network) {
    currentNetwork = network;
    
    // Update UI
    document.querySelectorAll('.network-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update connection status
    const statusDiv = document.getElementById('connectionStatus');
    if (network === 'bnb') {
        statusDiv.innerHTML = '<span class="status success">Connected to BNB Chain Testnet</span>';
        initializeBNBConnection();
    } else if (network === 'supra') {
        statusDiv.innerHTML = '<span class="status success">Connected to Supra Testnet</span>';
        initializeSupraConnection();
    }
}

// Initialize BNB Chain connection
async function initializeBNBConnection() {
    try {
        if (typeof window.ethereum !== 'undefined') {
            web3 = new Web3(window.ethereum);
            
            // Request account access
            await window.ethereum.request({ method: 'eth_requestAccounts' });
            
            // Switch to BNB testnet if needed
            try {
                await window.ethereum.request({
                    method: 'wallet_switchEthereumChain',
                    params: [{ chainId: '0x61' }], // BNB testnet
                });
            } catch (switchError) {
                console.log('Please switch to BNB testnet manually');
            }
            
            showStatus('Connected to BNB Chain via MetaMask', 'success');
        } else {
            showStatus('Please install MetaMask to connect to BNB Chain', 'error');
        }
    } catch (error) {
        showStatus('Failed to connect to BNB Chain: ' + error.message, 'error');
    }
}

// Initialize Supra connection
function initializeSupraConnection() {
    showStatus('Supra connection simulated (SDK integration needed)', 'success');
}

// Search for specific gene
async function searchGene() {
    const geneName = document.getElementById('geneInput').value.trim().toUpperCase();
    
    if (!geneName) {
        showStatus('Please enter a gene name', 'error');
        return;
    }
    
    showStatus('Searching for gene: ' + geneName, 'loading');
    
    // Simulate blockchain query with demo data
    setTimeout(() => {
        const gene = demoGenes.find(g => g.name === geneName);
        
        if (gene) {
            displayGeneResult(gene);
            showStatus('Gene found successfully', 'success');
        } else {
            displayNoResult(geneName);
            showStatus('Gene not found in registry', 'error');
        }
    }, 1500);
}

// Load top essential genes
function loadTopGenes() {
    showStatus('Loading top essential genes...', 'loading');
    
    setTimeout(() => {
        displayTopGenes(demoGenes);
        showStatus('Top essential genes loaded', 'success');
    }, 1000);
}

// Upload demo data
function uploadDemo() {
    showStatus('Uploading demo data to blockchain...', 'loading');
    
    setTimeout(() => {
        showStatus('Demo data uploaded successfully! Transaction: 0x123...abc', 'success');
    }, 2000);
}

// Display gene search result
function displayGeneResult(gene) {
    const resultsDiv = document.getElementById('searchResults');
    resultsDiv.innerHTML = `
        <div class="gene-card">
            <div class="gene-name">${gene.name} ${gene.isKnown ? '⭐' : ''}</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 0.9em; color: #666;">Essentiality Score</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: ${gene.score < -1.5 ? '#dc3545' : '#fd7e14'};">${gene.score}</div>
                </div>
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 0.9em; color: #666;">Consistency</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #28a745;">${gene.consistency}%</div>
                </div>
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 0.9em; color: #666;">Essential Cell Lines</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #17a2b8;">${gene.cellLines}/50</div>
                </div>
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 0.9em; color: #666;">Known GBM Gene</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: ${gene.isKnown ? '#28a745' : '#6c757d'};">${gene.isKnown ? 'Yes' : 'No'}</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #e9ecef; border-radius: 5px; font-size: 0.9em;">
                <strong>Network:</strong> ${currentNetwork.toUpperCase()} Chain | 
                <strong>Last Updated:</strong> ${new Date().toLocaleDateString()} |
                <strong>Data Hash:</strong> 0x${Math.random().toString(16).substr(2, 16)}...
            </div>
        </div>
    `;
}

// Display no result found
function displayNoResult(geneName) {
    const resultsDiv = document.getElementById('searchResults');
    resultsDiv.innerHTML = `
        <div class="gene-card" style="border-left-color: #dc3545;">
            <div class="gene-name" style="color: #dc3545;">Gene "${geneName}" not found</div>
            <p>This gene is not currently in the blockchain registry. You can:</p>
            <ul style="margin: 10px 0 0 20px;">
                <li>Check the spelling and try again</li>
                <li>Upload new analysis data containing this gene</li>
                <li>Browse the available top essential genes</li>
            </ul>
        </div>
    `;
}

// Display top essential genes
function displayTopGenes(genes) {
    const resultsDiv = document.getElementById('searchResults');
    
    const genesHtml = genes.map((gene, index) => `
        <div class="gene-card">
            <div class="gene-name">#${index + 1} ${gene.name} ${gene.isKnown ? '⭐' : ''}</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>Score: <strong style="color: #dc3545;">${gene.score}</strong></span>
                <span>Consistency: <strong style="color: #28a745;">${gene.consistency}%</strong></span>
                <span>Cell Lines: <strong style="color: #17a2b8;">${gene.cellLines}/50</strong></span>
            </div>
        </div>
    `).join('');
    
    resultsDiv.innerHTML = `
        <h3>Top Essential Genes in Glioblastoma</h3>
        ${genesHtml}
        <div style="text-align: center; margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 10px;">
            <p><strong>Total Genes in Registry:</strong> 1,000+ | <strong>Last Analysis:</strong> ${new Date().toLocaleDateString()}</p>
            <p><strong>Network:</strong> ${currentNetwork.toUpperCase()} Chain | <strong>Registry Contract:</strong> 0x123...abc</p>
        </div>
    `;
}

// Show status message
function showStatus(message, type) {
    const statusDiv = document.getElementById('connectionStatus');
    statusDiv.className = `status ${type}`;
    
    if (type === 'loading') {
        statusDiv.innerHTML = `<div style="display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 10px;"></div>${message}`;
    } else {
        statusDiv.innerHTML = message;
    }
}

// Add CSS animation for loading spinner
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

// Initialize with BNB Chain
document.addEventListener('DOMContentLoaded', function() {
    selectNetwork('bnb');
    
    // Add enter key support for search
    document.getElementById('geneInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchGene();
        }
    });
});
