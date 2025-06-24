// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title GeneRegistry
 * @dev Simple contract to store and retrieve GBM gene essentiality data
 * @notice This contract stores gene analysis results from the GBM Gene Essentiality Agent
 */
contract GeneRegistry {
    
    struct GeneData {
        string geneName;
        int256 essentialityScore;  // Scaled by 1000 (e.g., -1500 = -1.5)
        uint256 consistencyPct;   // Percentage (0-100)
        uint256 cellLineCount;    // Number of cell lines where gene is essential
        bool isKnownGbmGene;
        uint256 timestamp;
        string dataHash;          // Hash of full analysis data
    }
    
    struct AnalysisResults {
        string analysisId;
        uint256 totalGenes;
        uint256 totalCellLines;
        string merkleRoot;
        address researcher;
        uint256 timestamp;
        bool verified;
    }
    
    // State variables
    address public owner;
    uint256 public analysisCount;
    
    // Mappings
    mapping(string => GeneData) public genes;
    mapping(uint256 => AnalysisResults) public analyses;
    mapping(address => bool) public authorizedResearchers;
    
    // Arrays for enumeration
    string[] public geneNames;
    
    // Events
    event GeneDataStored(string geneName, int256 score, address researcher);
    event AnalysisSubmitted(uint256 analysisId, string merkleRoot, address researcher);
    event ResearcherAuthorized(address researcher);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorizedResearchers[msg.sender] || msg.sender == owner, "Not authorized researcher");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        authorizedResearchers[msg.sender] = true;
    }
    
    /**
     * @dev Store gene essentiality results
     */
    function storeGeneData(
        string memory _geneName,
        int256 _essentialityScore,
        uint256 _consistencyPct,
        uint256 _cellLineCount,
        bool _isKnownGbmGene,
        string memory _dataHash
    ) external onlyAuthorized {
        require(bytes(_geneName).length > 0, "Gene name cannot be empty");
        require(_consistencyPct <= 100, "Consistency percentage cannot exceed 100");
        
        // If this is a new gene, add to array
        if (bytes(genes[_geneName].geneName).length == 0) {
            geneNames.push(_geneName);
        }
        
        genes[_geneName] = GeneData({
            geneName: _geneName,
            essentialityScore: _essentialityScore,
            consistencyPct: _consistencyPct,
            cellLineCount: _cellLineCount,
            isKnownGbmGene: _isKnownGbmGene,
            timestamp: block.timestamp,
            dataHash: _dataHash
        });
        
        emit GeneDataStored(_geneName, _essentialityScore, msg.sender);
    }
    
    /**
     * @dev Submit complete analysis results
     */
    function submitAnalysis(
        string memory _analysisId,
        uint256 _totalGenes,
        uint256 _totalCellLines,
        string memory _merkleRoot
    ) external onlyAuthorized {
        require(bytes(_analysisId).length > 0, "Analysis ID cannot be empty");
        require(_totalGenes > 0, "Must analyze at least one gene");
        
        analysisCount++;
        
        analyses[analysisCount] = AnalysisResults({
            analysisId: _analysisId,
            totalGenes: _totalGenes,
            totalCellLines: _totalCellLines,
            merkleRoot: _merkleRoot,
            researcher: msg.sender,
            timestamp: block.timestamp,
            verified: false
        });
        
        emit AnalysisSubmitted(analysisCount, _merkleRoot, msg.sender);
    }
    
    /**
     * @dev Get gene data by name
     */
    function getGeneData(string memory _geneName) external view returns (GeneData memory) {
        require(bytes(genes[_geneName].geneName).length > 0, "Gene not found");
        return genes[_geneName];
    }
    
    /**
     * @dev Get top essential genes
     */
    function getTopGenes(uint256 _count) external view returns (string[] memory) {
        require(_count <= geneNames.length, "Count exceeds available genes");
        
        string[] memory topGenes = new string[](_count);
        for (uint256 i = 0; i < _count; i++) {
            topGenes[i] = geneNames[i];
        }
        return topGenes;
    }
    
    /**
     * @dev Verify analysis integrity
     */
    function verifyAnalysis(uint256 _analysisId, string memory _merkleRoot) external view returns (bool) {
        require(_analysisId <= analysisCount && _analysisId > 0, "Analysis not found");
        return keccak256(abi.encodePacked(analyses[_analysisId].merkleRoot)) == 
               keccak256(abi.encodePacked(_merkleRoot));
    }
    
    /**
     * @dev Authorize researcher
     */
    function authorizeResearcher(address _researcher) external onlyOwner {
        authorizedResearchers[_researcher] = true;
        emit ResearcherAuthorized(_researcher);
    }
    
    /**
     * @dev Get total genes count
     */
    function getTotalGenes() external view returns (uint256) {
        return geneNames.length;
    }
}