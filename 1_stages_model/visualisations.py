import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import get_output_dir
from utils.plot_config import STAGE_COLORS, set_apa_style, CONSTRUCT_COLORS

class MenopauseVisualisations:
    """Generate visualizations for cognitive and emotional measures across menopausal stages."""

    def __init__(self, data):
        self.data = data
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.transformed_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES_log', 'SAD_sqrt', 'FEARFULA_sqrt']
        self.var_labels = {
            'TOTIDE1': 'Cognitive Performance (Immediate Recall)',
            'TOTIDE2': 'Cognitive Performance (Delayed Recall)',
            'NERVES_log': 'Nervousness (Log)',
            'SAD_sqrt': 'Sadness (Sqrt)',
            'FEARFULA_sqrt': 'Fearfulness (Sqrt)'
        }
        self.output_dir = get_output_dir('1_stages_model')
        os.makedirs(self.output_dir, exist_ok=True)
        set_apa_style()

    def plot_violin_plots(self):
        """Create violin plots showing distributions by menopausal stage."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        axes = axes.flatten()

        stage_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
        stage_palette = [STAGE_COLORS.get(stage, '#888888') for stage in stage_order]

        for i, measure in enumerate(self.transformed_vars):
            ax = axes[i]

            sns.violinplot(
                data=self.data, x='STATUS_Label', y=measure,
                hue='STATUS_Label', legend=False, ax=ax,
                inner='box', palette=stage_palette
            )

            sns.stripplot(
                data=self.data.sample(min(500, len(self.data))),
                x='STATUS_Label', y=measure, ax=ax,
                color='black', alpha=0.05, size=1,
                jitter=True, dodge=False
            )

            measure_name = self.var_labels.get(measure, measure)
            ax.set_title(f'Distribution of {measure_name}', fontsize=14)
            sns.despine(ax=ax)
            ax.set_xlabel('Menopausal Stage', fontsize=10)
            ax.set_ylabel(measure_name, fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            means_by_stage = self.data.groupby('STATUS_Label', observed=True)[measure].mean().round(3)
            medians_by_stage = self.data.groupby('STATUS_Label', observed=True)[measure].median().round(3)
            counts_by_stage = self.data.groupby('STATUS_Label', observed=True)[measure].count()

            stats_text = "Group Statistics:\n"
            for stage in self.data['STATUS_Label'].cat.categories:
                if stage in means_by_stage:
                    stats_text += f"{stage} (n={counts_by_stage[stage]}):\n"
                    stats_text += f"  μ={means_by_stage[stage]}, Md={medians_by_stage[stage]}\n"

            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   va='top', ha='right', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        axes[5].set_visible(False)

        fig.suptitle(
            'Distribution of Cognitive and Emotional Measures Across Menopausal Stages',
            fontsize=16, y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'outcome_violin_plots.png'), dpi=300)
        plt.close()

    def plot_stages_vs_scores(self):
        """Create line plots showing mean scores across menopausal stages with error bars."""
        plot_data = self.data.copy()

        stage_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
        plot_data['STATUS_Label'] = pd.Categorical(
            plot_data['STATUS_Label'],
            categories=stage_order,
            ordered=True
        )

        melted_data = pd.melt(
            plot_data,
            id_vars=['SWANID', 'STATUS_Label'],
            value_vars=self.transformed_vars,
            var_name='Measure',
            value_name='Score'
        )

        melted_data['Measure_Label'] = melted_data['Measure'].map(self.var_labels)

        summary_data = melted_data.groupby(['Measure', 'Measure_Label', 'STATUS_Label'], observed=True).agg(
            Mean=('Score', 'mean'),
            SE=('Score', lambda x: x.std() / np.sqrt(len(x))),
            Count=('Score', 'count')
        ).reset_index()

        fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)
        axes = axes.flatten()

        line_color = CONSTRUCT_COLORS['cognitive_function']
        errorbar_color = '#004488'

        for i, measure in enumerate(self.transformed_vars):
            ax = axes[i]
            measure_data = summary_data[summary_data['Measure'] == measure]

            ax.set_title(self.var_labels.get(measure, measure), fontsize=14)
            sns.despine(ax=ax)

            if len(measure_data) == 0:
                ax.text(0.5, 0.5, f"No data available for {self.var_labels.get(measure, measure)}",
                       ha='center', va='center', fontsize=12)
                continue

            ax.errorbar(
                x=measure_data['STATUS_Label'].cat.codes,
                y=measure_data['Mean'],
                yerr=measure_data['SE'] * 1.96,
                fmt='o-', linewidth=2, markersize=8, capsize=5,
                color=line_color, ecolor=errorbar_color,
                label=self.var_labels.get(measure, measure)
            )

            measure_counts = melted_data[melted_data['Measure'] == measure].groupby('STATUS_Label', observed=True).size().reset_index(name='Count')
            measure_counts = measure_counts.set_index('STATUS_Label')

            for j, status in enumerate(stage_order):
                status_data = measure_data[measure_data['STATUS_Label'] == status]

                if not status_data.empty:
                    y_position = status_data['Mean'].values[0]
                    count = status_data['Count'].values[0]
                else:
                    y_position = 0
                    count = measure_counts.loc[status, 'Count'] if status in measure_counts.index else 0

                ax.annotate(
                    f'n={count}', xy=(j, y_position),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=18,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                )

            ax.set_title(self.var_labels.get(measure, measure), fontsize=18)
            ax.set_xticks(range(len(stage_order)))
            ax.set_xticklabels(stage_order, rotation=45, ha='right', fontsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.grid(True, linestyle=':', alpha=0.6)

            min_val = measure_data['Mean'].min() if not measure_data.empty else 0
            max_val = measure_data['Mean'].max() if not measure_data.empty else 0
            if min_val < 0 < max_val:
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        file_name = os.path.join(self.output_dir, 'stages_vs_scores_faceted.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_trajectory_classes(self):
        """Plot longitudinal trajectories showing how outcomes change over time by stage."""
        if 'VISIT' not in self.data.columns:
            return

        trajectory_data = self.data.copy()

        measures_to_plot = ['TOTIDE1', 'TOTIDE2', 'NERVES_log']
        measure_labels_short = {
            'TOTIDE1': 'Immediate Recall Score',
            'TOTIDE2': 'Delayed Recall Score',
            'NERVES_log': 'Nervousness Score (log)'
        }

        fig, axes = plt.subplots(3, 1, figsize=(12, 14))

        stage_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']

        stage_colors_map = {
            'Pre-menopause': STAGE_COLORS['Pre-menopause'],
            'Early Peri': STAGE_COLORS['Early Peri'],
            'Late Peri': STAGE_COLORS['Late Peri'],
            'Post-menopause': STAGE_COLORS['Post-menopause']
        }

        for idx, measure in enumerate(measures_to_plot):
            ax = axes[idx]

            if measure not in trajectory_data.columns:
                ax.text(0.5, 0.5, f'{measure} not available',
                       ha='center', va='center', fontsize=12)
                continue

            for stage in stage_order:
                if stage not in trajectory_data['STATUS_Label'].values:
                    continue

                stage_data = trajectory_data[trajectory_data['STATUS_Label'] == stage].copy()

                visit_stats = stage_data.groupby('VISIT').agg({
                    measure: ['mean', 'sem', 'count']
                }).reset_index()

                visit_stats.columns = ['VISIT', 'mean', 'sem', 'count']
                visit_stats = visit_stats[visit_stats['count'] >= 10]

                if len(visit_stats) < 2:
                    continue

                visit_stats['ci_lower'] = visit_stats['mean'] - 1.96 * visit_stats['sem']
                visit_stats['ci_upper'] = visit_stats['mean'] + 1.96 * visit_stats['sem']

                color = stage_colors_map.get(stage, '#888888')

                ax.plot(visit_stats['VISIT'], visit_stats['mean'],
                       marker='o', linewidth=2, markersize=6,
                       color=color, label=stage, alpha=0.9)

                ax.fill_between(visit_stats['VISIT'],
                               visit_stats['ci_lower'],
                               visit_stats['ci_upper'],
                               color=color, alpha=0.15)

            ax.set_xlabel('Follow-up Time Point (Visit)', fontsize=12)
            ax.set_ylabel(measure_labels_short.get(measure, measure), fontsize=12)
            ax.set_title(f'({chr(97+idx)}) {measure_labels_short.get(measure, measure)}',
                        fontsize=14, loc='left', fontweight='bold')

            if idx == 0:
                ax.legend(title='Menopausal Stage', fontsize=10,
                         title_fontsize=11, loc='upper right',
                         frameon=True, fancybox=True, shadow=True)

            ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=11)
            sns.despine(ax=ax)

        fig.suptitle('Longitudinal Trajectories of Cognitive and Emotional Outcomes\nAcross Menopausal Transition',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        output_file = os.path.join(self.output_dir, 'trajectory_classes.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def create_all_visualizations(self):
        """Generate all visualization outputs."""
        self.plot_violin_plots()
        self.plot_stages_vs_scores()
        self.plot_trajectory_classes()