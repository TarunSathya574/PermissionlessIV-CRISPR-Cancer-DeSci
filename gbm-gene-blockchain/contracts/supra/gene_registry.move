// gene_registry.move
// Simple Move module for storing gene essentiality data on Supra

module gene_registry::registry {
    use std::string::{Self, String};
    use std::vector;
    use aptos_framework::timestamp;
    use aptos_framework::account;
    use aptos_framework::event;

    /// Gene data structure
    struct GeneData has key, store, copy, drop {
        gene_name: String,
        essentiality_score: i64,  // Scaled by 1000
        consistency_pct: u64,     // Percentage 0-100
        cell_line_count: u64,
        is_known_gbm_gene: bool,
        timestamp: u64,
        data_hash: String,
    }

    /// Analysis results structure
    struct AnalysisResults has key, store, copy, drop {
        analysis_id: String,
        total_genes: u64,
        total_cell_lines: u64,
        merkle_root: String,
        researcher: address,
        timestamp: u64,
        verified: bool,
    }

    /// Registry resource
    struct Registry has key {
        genes: vector<GeneData>,
        analyses: vector<AnalysisResults>,
        authorized_researchers: vector<address>,
        owner: address,
        analysis_count: u64,
    }

    /// Events
    struct GeneDataStored has drop, store {
        gene_name: String,
        score: i64,
        researcher: address,
    }

    /// Error codes
    const E_NOT_AUTHORIZED: u64 = 1;
    const E_GENE_NOT_FOUND: u64 = 2;
    const E_INVALID_DATA: u64 = 3;

    /// Initialize the registry
    public entry fun initialize(account: &signer) {
        let addr = account::address_of(account);
        let registry = Registry {
            genes: vector::empty<GeneData>(),
            analyses: vector::empty<AnalysisResults>(),
            authorized_researchers: vector::singleton(addr),
            owner: addr,
            analysis_count: 0,
        };
        move_to(account, registry);
    }

    /// Store gene essentiality data
    public entry fun store_gene_data(
        account: &signer,
        gene_name: String,
        essentiality_score: i64,
        consistency_pct: u64,
        cell_line_count: u64,
        is_known_gbm_gene: bool,
        data_hash: String,
    ) acquires Registry {
        let researcher_addr = account::address_of(account);
        let registry = borrow_global_mut<Registry>(@gene_registry);
        
        // Check authorization
        assert!(
            vector::contains(&registry.authorized_researchers, &researcher_addr) || 
            researcher_addr == registry.owner,
            E_NOT_AUTHORIZED
        );

        let gene_data = GeneData {
            gene_name: gene_name,
            essentiality_score,
            consistency_pct,
            cell_line_count,
            is_known_gbm_gene,
            timestamp: timestamp::now_microseconds(),
            data_hash,
        };

        vector::push_back(&mut registry.genes, gene_data);

        // Emit event
        event::emit(GeneDataStored {
            gene_name: gene_data.gene_name,
            score: essentiality_score,
            researcher: researcher_addr,
        });
    }

    /// Get gene data by name
    public fun get_gene_data(gene_name: String): GeneData acquires Registry {
        let registry = borrow_global<Registry>(@gene_registry);
        let (found, index) = find_gene_index(&registry.genes, &gene_name);
        assert!(found, E_GENE_NOT_FOUND);
        *vector::borrow(&registry.genes, index)
    }

    /// Helper function to find gene index
    fun find_gene_index(genes: &vector<GeneData>, gene_name: &String): (bool, u64) {
        let len = vector::length(genes);
        let i = 0;
        while (i < len) {
            let gene = vector::borrow(genes, i);
            if (gene.gene_name == *gene_name) {
                return (true, i)
            };
            i = i + 1;
        };
        (false, 0)
    }

    /// Get total genes count
    public fun get_total_genes(): u64 acquires Registry {
        let registry = borrow_global<Registry>(@gene_registry);
        vector::length(&registry.genes)
    }
}