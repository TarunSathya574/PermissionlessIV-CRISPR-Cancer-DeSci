require("@nomiclabs/hardhat-ethers");

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    bnb_testnet: {
      url: "https://bsc-testnet.publicnode.com",
      chainId: 97,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      gas: 3000000,
      gasPrice: 5000000000 // 5 gwei
    },
    bnb_mainnet: {
      url: "https://bsc-dataseed.binance.org",
      chainId: 56,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      gas: 3000000,
      gasPrice: 5000000000
    }
  },
  paths: {
    sources: "./contracts/bnb",
    artifacts: "./artifacts",
    cache: "./cache"
  }
};
