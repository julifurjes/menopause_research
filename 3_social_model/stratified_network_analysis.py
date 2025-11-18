import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir

class StratifiedNetworkAnalysis:
    """
    Network analysis stratified by menopausal stage to account for repeated measures.
    """

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.output_dir = get_output_dir('3_social_model', 'stratified_networks')

        # Define variable groups
        self.social_support_vars = ['LISTEN', 'TAKETOM', 'HELPSIC', 'CONFIDE']
        self.emotional_struggle_vars = ['EMOCTDW', 'EMOACCO', 'EMOCARE']
        self.social_struggle_vars = ['INTERFR', 'SOCIAL']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']

        # Menopausal stages in order
        self.stage_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']

    def preprocess_data(self):
        """Prepare data for stratified network analysis"""
        # Convert variables to numeric
        relevant_vars = (self.social_support_vars + self.emotional_struggle_vars +
                        self.social_struggle_vars + self.cognitive_vars +
                        ['STATUS', 'SWANID', 'VISIT'])

        for col in relevant_vars:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Filter to relevant menopausal statuses
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]

        # Map status to labels
        status_map = {
            1: 'Surgical',
            2: 'Post-menopause',
            3: 'Late Peri',
            4: 'Early Peri',
            5: 'Pre-menopause',
            8: 'Surgical'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)

        # Create composite scores
        self.data['social_support'] = self.data[self.social_support_vars].mean(axis=1)
        self.data['emotional_struggle'] = self.data[self.emotional_struggle_vars].mean(axis=1)
        self.data['social_struggle'] = self.data[self.social_struggle_vars].mean(axis=1)
        self.data['cognitive_function'] = self.data[self.cognitive_vars].mean(axis=1)

        # Drop rows with missing composite values
        composite_vars = ['social_support', 'emotional_struggle', 'social_struggle', 'cognitive_function']
        self.data = self.data.dropna(subset=composite_vars)

        print(f"\nData preprocessed: {len(self.data)} observations from {self.data['SWANID'].nunique()} subjects")
        print(f"Observations by stage:")
        print(self.data['STATUS_Label'].value_counts())

    def create_network_by_stage(self, stage):
        """Create correlation network for a specific menopausal stage"""
        # Filter data for this stage
        stage_data = self.data[self.data['STATUS_Label'] == stage]

        if len(stage_data) < 30:  # Minimum sample size check
            print(f"Warning: Only {len(stage_data)} observations for {stage}. Skipping network.")
            return None

        # Define network variables
        network_vars = ['social_support', 'emotional_struggle', 'social_struggle', 'cognitive_function']

        # Calculate correlation matrix
        corr_matrix = stage_data[network_vars].corr()

        # Create network graph
        G = nx.Graph()

        # Node labels
        node_labels = {
            'social_support': 'Social\nSupport',
            'emotional_struggle': 'Emotional\nStruggle',
            'social_struggle': 'Social\nStruggle',
            'cognitive_function': 'Cognitive\nFunction'
        }

        # Add nodes
        for var in network_vars:
            G.add_node(var, label=node_labels[var])

        # Add edges with correlation as weight
        for i, var1 in enumerate(network_vars):
            for j, var2 in enumerate(network_vars):
                if i < j:  # Only add each edge once
                    corr_val = corr_matrix.loc[var1, var2]
                    if abs(corr_val) > 0.1:  # Only show correlations > 0.1
                        G.add_edge(var1, var2, weight=abs(corr_val), correlation=corr_val)

        return G, corr_matrix, len(stage_data)

    def plot_stratified_networks(self):
        """Create network plots for each menopausal stage"""
        # Create figure with subplots for each stage
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()

        # Color palette
        green_palette = sns.color_palette("YlGn", n_colors=8)
        node_color = green_palette[2]
        text_color = green_palette[7]

        # Warm colors for negative correlations
        warm_colors = {
            'strong': '#B8860B',
            'moderate': '#DAA520',
            'weak': '#F4E4BC'
        }

        network_stats = []

        for idx, stage in enumerate(self.stage_order):
            ax = axes[idx]

            # Create network for this stage
            result = self.create_network_by_stage(stage)

            if result is None:
                ax.text(0.5, 0.5, f'{stage}\nInsufficient data',
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                continue

            G, corr_matrix, n_obs = result

            # Position nodes in a circle
            pos = nx.circular_layout(G)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                  node_color=node_color,
                                  node_size=2000,
                                  alpha=0.8,
                                  edgecolors=green_palette[5],
                                  linewidths=2,
                                  ax=ax)

            # Prepare edges
            edges = list(G.edges())
            edge_colors = []
            edge_widths = []

            for edge in edges:
                correlation = G[edge[0]][edge[1]]['correlation']
                weight = G[edge[0]][edge[1]]['weight']

                # Calculate edge width
                width = max(weight * 10, 2)
                edge_widths.append(width)

                # Calculate edge color
                if correlation > 0:
                    if weight > 0.6:
                        edge_colors.append(green_palette[7])
                    elif weight > 0.4:
                        edge_colors.append(green_palette[6])
                    elif weight > 0.2:
                        edge_colors.append(green_palette[4])
                    else:
                        edge_colors.append(green_palette[3])
                else:
                    if weight > 0.6:
                        edge_colors.append(warm_colors['strong'])
                    elif weight > 0.4:
                        edge_colors.append(warm_colors['moderate'])
                    else:
                        edge_colors.append(warm_colors['weak'])

            # Draw edges
            if edges:
                nx.draw_networkx_edges(G, pos,
                                      edgelist=edges,
                                      width=edge_widths,
                                      edge_color=edge_colors,
                                      alpha=0.9,
                                      ax=ax)

            # Draw labels
            labels = {node: data['label'] for node, data in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels,
                                   font_size=12,
                                   font_weight='bold',
                                   font_color=text_color,
                                   ax=ax)

            # Add correlation values as edge labels
            edge_labels = {}
            for u, v in G.edges():
                correlation = G[u][v]['correlation']
                edge_labels[(u, v)] = f'{correlation:.2f}'

            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                        font_size=10,
                                        font_color=text_color,
                                        bbox=dict(boxstyle='round,pad=0.2',
                                                facecolor='white',
                                                edgecolor=green_palette[4],
                                                alpha=0.8),
                                        ax=ax)

            # Add title with sample size
            ax.set_title(f'{stage}\n(n={n_obs})', fontsize=14, fontweight='bold')
            ax.axis('off')

            # Store network statistics
            network_stats.append({
                'Stage': stage,
                'N': n_obs,
                'N_Edges': len(edges),
                'Avg_Correlation': np.mean([abs(G[u][v]['correlation']) for u, v in edges]) if edges else 0
            })

        # Hide the 6th subplot
        axes[5].axis('off')

        # Add overall title
        fig.suptitle('Social Support-Cognition Networks by Menopausal Stage\n(Stratified Analysis to Account for Repeated Measures)',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(self.output_dir, 'stratified_networks.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"\nStratified networks saved to: {output_path}")

        # Print network statistics
        stats_df = pd.DataFrame(network_stats)
        print("\nNetwork Statistics by Stage:")
        print(stats_df.to_string(index=False))

        # Save statistics
        stats_path = os.path.join(self.output_dir, 'network_statistics.csv')
        stats_df.to_csv(stats_path, index=False)

        return stats_df

    def compare_networks_across_stages(self):
        """Compare network characteristics across menopausal stages"""
        comparison_data = []

        for stage in self.stage_order:
            result = self.create_network_by_stage(stage)

            if result is None:
                continue

            G, corr_matrix, n_obs = result

            # Calculate network metrics
            density = nx.density(G)
            avg_clustering = nx.average_clustering(G) if len(G.nodes()) > 0 else 0

            # Get correlation strengths
            correlations = [G[u][v]['correlation'] for u, v in G.edges()]

            comparison_data.append({
                'Stage': stage,
                'N_Observations': n_obs,
                'Network_Density': density,
                'Avg_Clustering': avg_clustering,
                'N_Edges': len(G.edges()),
                'Avg_Correlation': np.mean(correlations) if correlations else 0,
                'Max_Correlation': np.max(np.abs(correlations)) if correlations else 0
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Print comparison
        print("\nNetwork Comparison Across Stages:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        # Save comparison
        comparison_path = os.path.join(self.output_dir, 'network_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)

        return comparison_df

    def run_analysis(self):
        """Run complete stratified network analysis"""
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            self.preprocess_data()

            print("\nCreating stratified networks...")
            stats = self.plot_stratified_networks()

            print("\nComparing networks across stages...")
            comparison = self.compare_networks_across_stages()

            print("STRATIFIED ANALYSIS COMPLETE")
            print(f"\nResults saved to: {self.output_dir}")

            return stats, comparison

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    analysis = StratifiedNetworkAnalysis("processed_combined_data.csv")
    analysis.run_analysis()
