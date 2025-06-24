#!/bin/bash

echo "🚀 Deploying Gene Registry to Supra Blockchain..."

# Check if supra CLI is installed
if ! command -v supra &> /dev/null; then
    echo "❌ Supra CLI not found. Please install Supra CLI first."
    echo "Visit: https://docs.supra.com for installation instructions"
    exit 1
fi

# Set network (testnet/devnet)
NETWORK="testnet"
ACCOUNT_ADDRESS=${SUPRA_ACCOUNT_ADDRESS:-""}
PRIVATE_KEY=${SUPRA_PRIVATE_KEY:-""}

if [ -z "$ACCOUNT_ADDRESS" ] || [ -z "$PRIVATE_KEY" ]; then
    echo "⚠️  Please set SUPRA_ACCOUNT_ADDRESS and SUPRA_PRIVATE_KEY environment variables"
    echo "Example:"
    echo "export SUPRA_ACCOUNT_ADDRESS=0x123..."
    echo "export SUPRA_PRIVATE_KEY=your_private_key"
    exit 1
fi

echo "📝 Compiling Move module..."

# Compile the Move module
supra move compile --package-dir . --named-addresses gene_registry=$ACCOUNT_ADDRESS

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed"
    exit 1
fi

echo "✅ Compilation successful"

echo "🚀 Publishing module to $NETWORK..."

# Publish the module
supra move publish \
    --package-dir . \
    --named-addresses gene_registry=$ACCOUNT_ADDRESS \
    --private-key $PRIVATE_KEY \
    --network $NETWORK \
    --gas-budget 50000

if [ $? -eq 0 ]; then
    echo "✅ Module published successfully!"
    echo "📝 Contract Address: $ACCOUNT_ADDRESS"
    
    # Save deployment info
    cat > deployment_supra.json << EOF
{
    "contractAddress": "$ACCOUNT_ADDRESS",
    "network": "Supra $NETWORK",
    "deploymentTime": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployer": "$ACCOUNT_ADDRESS"
}
EOF
    
    echo "📁 Deployment info saved to deployment_supra.json"
    
    # Initialize the registry
    echo "🔧 Initializing registry..."
    supra move run \
        --function-id ${ACCOUNT_ADDRESS}::registry::initialize \
        --private-key $PRIVATE_KEY \
        --network $NETWORK \
        --gas-budget 10000
    
    if [ $? -eq 0 ]; then
        echo "✅ Registry initialized successfully!"
    else
        echo "⚠️  Registry initialization failed, but deployment was successful"
    fi
    
else
    echo "❌ Deployment failed"
    exit 1
fi

echo "🎉 Supra deployment complete!"
