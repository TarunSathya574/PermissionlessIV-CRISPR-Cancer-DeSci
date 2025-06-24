const { ethers } = require("hardhat");

async function main() {
    console.log("🚀 Deploying GeneRegistry to BNB Chain...");
    
    // Get the contract factory
    const GeneRegistry = await ethers.getContractFactory("GeneRegistry");
    
    // Deploy the contract
    console.log("📝 Deploying contract...");
    const geneRegistry = await GeneRegistry.deploy();
    
    // Wait for deployment
    await geneRegistry.waitForDeployment();
    
    const contractAddress = await geneRegistry.getAddress();
    console.log("✅ GeneRegistry deployed to:", contractAddress);
    
    // Save deployment info
    const deploymentInfo = {
        contractAddress: contractAddress,
        network: "BNB Chain Testnet",
        deploymentTime: new Date().toISOString(),
        deployer: (await ethers.getSigners())[0].address
    };
    
    const fs = require('fs');
    fs.writeFileSync(
        'deployment.json', 
        JSON.stringify(deploymentInfo, null, 2)
    );
    
    console.log("📁 Deployment info saved to deployment.json");
    
    // Verify deployment by calling a view function
    try {
        const totalGenes = await geneRegistry.getTotalGenes();
        console.log("🔍 Contract verification - Total genes:", totalGenes.toString());
        console.log("✅ Contract deployment successful!");
    } catch (error) {
        console.log("⚠️  Contract verification failed:", error.message);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("❌ Deployment failed:", error);
        process.exit(1);
    });
