"""
GBM Gene Essentiality Agent
Specialized agent for identifying genes with the most negative scores (highest essentiality) 
across glioblastoma cell lines from CRISPR screening data.

Built for integration with VerifiCRISPR Pipeline
Author: CRISPR-Cancer-DeSci Team
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GBMGeneEssentialityAgent:
    """
    Specialized agent for analyzing gene essentiality in glioblastoma (GBM) cell lines.
    
    This agent identifies genes with the most negative CRISPR scores, indicating genes
    that are essential for cell survival when knocked out.
    
    Key Features:
    - Automatic GBM cell line identification
    - Multiple essentiality analysis methods
    - Cross-cell-line pattern analysis
    - Statistical analysis and visualization
    - Export capabilities
    - Integration with VerifiCRISPR pipeline
    """
    
    def __init__(self, crispr_data_path: str, sample_info_path: Optional[str] = None):
        """
        Initialize the GBM Gene Essentiality Agent.
        
        Args:
            crispr_data_path: Path to CRISPR gene effect data (CSV format)
            sample_info_path: Optional path to sample information file
        """
        self.crispr_data_path = crispr_data_path
        self.sample_info_path = sample_info_path
        
        # Data containers
        self.crispr_data = None
        self.sample_info = None
        self.gbm_data = None
        self.gbm_cell_lines = []
        
        # Analysis results
        self.essentiality_scores = None
        self.analysis_results = {}
        
        # Configuration
        self.config = {
            'negative_threshold': -0.5,  # Threshold for considering a gene essential
            'consistency_threshold': 0.8,  # Fraction of cell lines for consistency analysis
            'min_cell_lines': 3,  # Minimum cell lines required for analysis
            'known_gbm_genes': [
                'EGFR', 'TP53', 'PTEN', 'IDH1', 'ATRX', 'CIC', 'FUBP1', 
                'PIK3CA', 'PIK3R1', 'NF1', 'RB1', 'CDKN2A', 'MDM2', 'MDM4'
            ]
        }
        
        logger.info(f"Initialized GBM Gene Essentiality Agent")
        logger.info(f"CRISPR data path: {crispr_data_path}")
        
    def load_data(self) -> bool:
        """
        Load CRISPR data and sample information.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            logger.info("Loading CRISPR gene effect data...")
            self.crispr_data = pd.read_csv(self.crispr_data_path, index_col=0)
            logger.info(f"Loaded CRISPR data: {self.crispr_data.shape[0]} genes × {self.crispr_data.shape[1]} cell lines")
            
            # Load sample info if provided
            if self.sample_info_path and os.path.exists(self.sample_info_path):
                logger.info("Loading sample information...")
                self.sample_info = pd.read_csv(self.sample_info_path)
                logger.info(f"Loaded sample info for {len(self.sample_info)} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def identify_gbm_cell_lines(self) -> List[str]:
        """
        Identify GBM cell lines from the dataset.
        
        Returns:
            List[str]: List of GBM cell line names
        """
        gbm_lines = []
        
        # Method 1: Use sample info if available
        if self.sample_info is not None:
            gbm_samples = self.sample_info[
                self.sample_info["disease"].str.contains("glioblastoma", case=False, na=False) |
                self.sample_info["cancer_type"].str.contains("brain", case=False, na=False)
            ]
            if not gbm_samples.empty:
                gbm_lines = gbm_samples["cell_line"].tolist()
                logger.info(f"Identified {len(gbm_lines)} GBM cell lines from sample info")
        
        # Method 2: Pattern matching in column names
        if not gbm_lines:
            gbm_patterns = ['GBM', 'glioblastoma', 'brain', 'CNS']
            for pattern in gbm_patterns:
                pattern_matches = [col for col in self.crispr_data.columns 
                                 if pattern.lower() in col.lower()]
                gbm_lines.extend(pattern_matches)
            
            gbm_lines = list(set(gbm_lines))  # Remove duplicates
            logger.info(f"Identified {len(gbm_lines)} GBM cell lines from pattern matching")
        
        # Method 3: Fallback - assume all are GBM if no specific identification
        if not gbm_lines:
            gbm_lines = self.crispr_data.columns.tolist()
            logger.warning(f"No GBM-specific cell lines identified. Using all {len(gbm_lines)} cell lines")
        
        self.gbm_cell_lines = gbm_lines
        self.gbm_data = self.crispr_data[gbm_lines]
        
        return gbm_lines
    
    def analyze_gene_essentiality(self) -> pd.Series:
        """
        Analyze gene essentiality across GBM cell lines.
        
        Returns:
            pd.Series: Gene essentiality scores (mean across cell lines)
        """
        if self.gbm_data is None:
            logger.error("GBM data not available. Run identify_gbm_cell_lines() first.")
            return None
        
        logger.info("Analyzing gene essentiality...")
        
        # Calculate essentiality scores
        self.essentiality_scores = self.gbm_data.mean(axis=1).sort_values()
        
        # Calculate additional statistics
        essentiality_stats = {
            'mean_scores': self.gbm_data.mean(axis=1),
            'median_scores': self.gbm_data.median(axis=1),
            'std_scores': self.gbm_data.std(axis=1),
            'min_scores': self.gbm_data.min(axis=1),
            'max_scores': self.gbm_data.max(axis=1),
            'essential_cell_count': (self.gbm_data < self.config['negative_threshold']).sum(axis=1)
        }
        
        self.analysis_results['essentiality_stats'] = pd.DataFrame(essentiality_stats)
        
        logger.info(f"Calculated essentiality scores for {len(self.essentiality_scores)} genes")
        
        return self.essentiality_scores
    
    def get_most_negative_genes(self, n_genes: int = 20, method: str = 'mean') -> pd.DataFrame:
        """
        Get genes with the most negative scores (highest essentiality).
        
        Args:
            n_genes: Number of top genes to return
            method: Method for ranking ('mean', 'median', 'min', 'consistency')
            
        Returns:
            pd.DataFrame: Top essential genes with detailed statistics
        """
        if self.essentiality_scores is None:
            self.analyze_gene_essentiality()
        
        stats_df = self.analysis_results['essentiality_stats']
        
        # Select ranking method
        if method == 'mean':
            ranking_scores = stats_df['mean_scores']
        elif method == 'median':
            ranking_scores = stats_df['median_scores']
        elif method == 'min':
            ranking_scores = stats_df['min_scores']
        elif method == 'consistency':
            ranking_scores = stats_df['essential_cell_count']
            ranking_scores = ranking_scores.sort_values(ascending=False)
        else:
            logger.warning(f"Unknown method {method}, using 'mean'")
            ranking_scores = stats_df['mean_scores']
        
        # Get top genes
        if method == 'consistency':
            top_genes = ranking_scores.head(n_genes).index
        else:
            top_genes = ranking_scores.sort_values().head(n_genes).index
        
        # Create detailed results
        top_genes_df = stats_df.loc[top_genes].copy()
        top_genes_df['rank'] = range(1, len(top_genes) + 1)
        top_genes_df['is_known_gbm_gene'] = top_genes_df.index.isin(self.config['known_gbm_genes'])
        
        # Add consistency metrics
        top_genes_df['consistency_pct'] = (top_genes_df['essential_cell_count'] / 
                                          len(self.gbm_cell_lines)) * 100
        
        # Reorder columns
        columns_order = ['rank', 'mean_scores', 'median_scores', 'std_scores', 
                        'min_scores', 'max_scores', 'essential_cell_count', 
                        'consistency_pct', 'is_known_gbm_gene']
        top_genes_df = top_genes_df[columns_order]
        
        logger.info(f"Identified top {n_genes} essential genes using method: {method}")
        
        return top_genes_df
    
    def get_consistent_essential_genes(self, threshold: float = None, 
                                     consistency_pct: float = None) -> pd.DataFrame:
        """
        Get genes that are consistently essential across multiple cell lines.
        
        Args:
            threshold: Essentiality threshold (default from config)
            consistency_pct: Minimum percentage of cell lines (default from config)
            
        Returns:
            pd.DataFrame: Consistently essential genes
        """
        if threshold is None:
            threshold = self.config['negative_threshold']
        if consistency_pct is None:
            consistency_pct = self.config['consistency_threshold']
        
        if self.gbm_data is None:
            logger.error("GBM data not available.")
            return pd.DataFrame()
        
        # Count cell lines where each gene is essential
        essential_mask = self.gbm_data < threshold
        essential_counts = essential_mask.sum(axis=1)
        consistency_ratios = essential_counts / len(self.gbm_cell_lines)
        
        # Filter consistent genes
        consistent_genes = consistency_ratios[consistency_ratios >= consistency_pct]
        
        if len(consistent_genes) == 0:
            logger.warning(f"No genes found with {consistency_pct*100}% consistency at threshold {threshold}")
            return pd.DataFrame()
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'gene': consistent_genes.index,
            'consistency_ratio': consistent_genes.values,
            'essential_cell_count': essential_counts[consistent_genes.index],
            'mean_score': self.gbm_data.loc[consistent_genes.index].mean(axis=1),
            'median_score': self.gbm_data.loc[consistent_genes.index].median(axis=1),
            'std_score': self.gbm_data.loc[consistent_genes.index].std(axis=1)
        }).sort_values('consistency_ratio', ascending=False)
        
        results_df['is_known_gbm_gene'] = results_df['gene'].isin(self.config['known_gbm_genes'])
        
        logger.info(f"Found {len(results_df)} consistently essential genes")
        
        return results_df
    
    def query_gene(self, gene_name: str) -> Dict:
        """
        Query detailed information about a specific gene.
        
        Args:
            gene_name: Name of the gene to query
            
        Returns:
            Dict: Detailed gene information
        """
        if self.gbm_data is None:
            logger.error("GBM data not available.")
            return {}
        
        if gene_name not in self.gbm_data.index:
            logger.error(f"Gene {gene_name} not found in data")
            return {}
        
        gene_scores = self.gbm_data.loc[gene_name]
        
        result = {
            'gene_name': gene_name,
            'mean_score': float(gene_scores.mean()),
            'median_score': float(gene_scores.median()),
            'std_score': float(gene_scores.std()),
            'min_score': float(gene_scores.min()),
            'max_score': float(gene_scores.max()),
            'essential_cell_lines': gene_scores[gene_scores < self.config['negative_threshold']].index.tolist(),
            'non_essential_cell_lines': gene_scores[gene_scores >= self.config['negative_threshold']].index.tolist(),
            'consistency_ratio': float((gene_scores < self.config['negative_threshold']).sum() / len(gene_scores)),
            'is_known_gbm_gene': gene_name in self.config['known_gbm_genes'],
            'percentile_rank': float(stats.percentileofscore(self.gbm_data.mean(axis=1), gene_scores.mean())),
            'cell_line_scores': gene_scores.to_dict()
        }
        
        return result
    
    def visualize_essentiality(self, genes: Optional[List[str]] = None, 
                             plot_type: str = 'heatmap', save_path: Optional[str] = None) -> None:
        """
        Create visualizations of gene essentiality patterns.
        
        Args:
            genes: Specific genes to visualize (default: top 20 essential)
            plot_type: Type of plot ('heatmap', 'distribution', 'all')
            save_path: Optional path to save the plot
        """
        if self.gbm_data is None:
            logger.error("GBM data not available.")
            return
        
        if genes is None:
            top_genes_df = self.get_most_negative_genes(20)
            genes = top_genes_df.index.tolist()
        
        available_genes = [g for g in genes if g in self.gbm_data.index]
        
        plt.style.use('seaborn-v0_8')
        
        if plot_type == 'all':
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('GBM Gene Essentiality Analysis', fontsize=16, fontweight='bold')
            
            # Heatmap
            ax1 = axes[0, 0]
            gene_data = self.gbm_data.loc[available_genes[:15]]  # Top 15 for visibility
            sns.heatmap(gene_data, cmap='RdBu_r', center=0, ax=ax1, 
                       xticklabels=False, cbar_kws={'label': 'Essentiality Score'})
            ax1.set_title('Essentiality Heatmap (Top 15 Genes)')
            
            # Distribution
            ax2 = axes[0, 1]
            all_scores = self.gbm_data.values.flatten()
            all_scores = all_scores[~np.isnan(all_scores)]
            ax2.hist(all_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.axvline(np.percentile(all_scores, 5), color='red', linestyle='--', 
                       label=f'5th percentile: {np.percentile(all_scores, 5):.2f}')
            ax2.set_xlabel('Essentiality Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Score Distribution')
            ax2.legend()
            
            # Top genes bar plot
            ax3 = axes[1, 0]
            top_10_scores = self.gbm_data.loc[available_genes[:10]].mean(axis=1)
            bars = ax3.barh(range(len(top_10_scores)), top_10_scores.values, 
                           color='crimson', alpha=0.7)
            ax3.set_yticks(range(len(top_10_scores)))
            ax3.set_yticklabels(top_10_scores.index)
            ax3.set_xlabel('Mean Essentiality Score')
            ax3.set_title('Top 10 Essential Genes')
            ax3.invert_yaxis()
            
            # Consistency analysis
            ax4 = axes[1, 1]
            essential_counts = (self.gbm_data.loc[available_genes] < 
                              self.config['negative_threshold']).sum(axis=1)
            consistency_pct = (essential_counts / len(self.gbm_cell_lines)) * 100
            ax4.scatter(self.gbm_data.loc[available_genes].mean(axis=1), 
                       consistency_pct, alpha=0.7, s=60)
            ax4.set_xlabel('Mean Essentiality Score')
            ax4.set_ylabel('Consistency (% Cell Lines)')
            ax4.set_title('Essentiality vs Consistency')
            ax4.grid(True, alpha=0.3)
            
        elif plot_type == 'heatmap':
            fig, ax = plt.subplots(figsize=(12, 8))
            gene_data = self.gbm_data.loc[available_genes]
            sns.heatmap(gene_data, cmap='RdBu_r', center=0, ax=ax,
                       cbar_kws={'label': 'Essentiality Score'})
            ax.set_title(f'Gene Essentiality Heatmap ({len(available_genes)} genes)')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, output_dir: str, format: str = 'csv') -> None:
        """
        Export analysis results to files.
        
        Args:
            output_dir: Output directory path
            format: Export format ('csv', 'json', 'excel')
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Exporting results to {output_dir} in {format} format")
        
        # Get analysis results
        top_genes = self.get_most_negative_genes(50)
        consistent_genes = self.get_consistent_essential_genes()
        
        if format == 'csv':
            top_genes.to_csv(output_path / 'top_essential_genes.csv')
            consistent_genes.to_csv(output_path / 'consistent_essential_genes.csv', index=False)
                
        elif format == 'json':
            top_genes.to_json(output_path / 'top_essential_genes.json', indent=2)
            consistent_genes.to_json(output_path / 'consistent_essential_genes.json', indent=2)
                
        logger.info(f"Results exported successfully to {output_dir}")
    
    def run_complete_analysis(self, output_dir: Optional[str] = None) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            output_dir: Optional output directory for results
            
        Returns:
            Dict: Complete analysis results
        """
        logger.info("Starting complete GBM gene essentiality analysis...")
        
        # Load data
        if not self.load_data():
            return {"error": "Failed to load data"}
        
        # Identify GBM cell lines
        self.identify_gbm_cell_lines()
        
        if len(self.gbm_cell_lines) < self.config['min_cell_lines']:
            error_msg = f"Insufficient GBM cell lines ({len(self.gbm_cell_lines)}) for analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Run analyses
        self.analyze_gene_essentiality()
        
        results = {
            'top_essential_genes': self.get_most_negative_genes(50),
            'consistent_essential_genes': self.get_consistent_essential_genes(),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Export results if requested
        if output_dir:
            self.export_results(output_dir)
            self.visualize_essentiality(plot_type='all', 
                                      save_path=f"{output_dir}/essentiality_analysis.png")
        
        logger.info("Complete analysis finished successfully!")
        
        return results


# Example usage
def main_example():
    """Example usage of the GBM Gene Essentiality Agent"""
    
    # Initialize agent
    agent = GBMGeneEssentialityAgent(
        crispr_data_path="data/CRISPRGeneEffect.csv",
        sample_info_path="data/sample_info.csv"
    )
    
    # Run complete analysis
    results = agent.run_complete_analysis(output_dir="gbm_analysis_results")
    
    # Display key findings
    if "error" not in results:
        print("\n" + "="*60)
        print("🧬 GBM GENE ESSENTIALITY ANALYSIS COMPLETE 🧬")
        print("="*60)
        
        top_genes = results['top_essential_genes'].head(10)
        print(f"\n🏆 TOP 10 MOST ESSENTIAL GENES:")
        print("-" * 50)
        for i, (gene, row) in enumerate(top_genes.iterrows(), 1):
            known_marker = "⭐" if row['is_known_gbm_gene'] else "  "
            print(f"{i:2d}. {known_marker} {gene:8s} | Score: {row['mean_scores']:6.3f} | Consistency: {row['consistency_pct']:5.1f}%")
        
        print("\n" + "="*60)
    else:
        print(f"Analysis failed: {results['error']}")


if __name__ == "__main__":
    main_example()
