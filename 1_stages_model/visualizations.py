import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.plot_config import STAGE_COLORS, set_apa_style, CONSTRUCT_COLORS, SIGNIFICANCE_COLORS, get_significance_color, get_categorical_palette


def plot_violin_plots(analysis_obj):
    """Create violin plots showing distributions by menopausal stage."""
    set_apa_style()

    data = analysis_obj.data
    output_dir = analysis_obj.output_dir

    outcome_vars = ['TOTIDE1', 'TOTIDE2']  # Only cognitive outcomes
    transformed_vars = ['TOTIDE1', 'TOTIDE2']
    var_labels = {
        'TOTIDE1': 'Cognitive Performance (Immediate Recall)',
        'TOTIDE2': 'Cognitive Performance (Delayed Recall)'
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes = axes.flatten()

    stage_order = ['Premenopause', 'Early Perimenopause', 'Late Perimenopause', 'Postmenopause', 'Surgical']
    stage_palette = [STAGE_COLORS.get(stage, '#888888') for stage in stage_order]

    for i, measure in enumerate(transformed_vars):
        ax = axes[i]

        sns.violinplot(
            data=data, x='STATUS_Label', y=measure,
            hue='STATUS_Label', legend=False, ax=ax,
            inner='box', palette=stage_palette
        )

        sns.stripplot(
            data=data.sample(min(500, len(data))),
            x='STATUS_Label', y=measure, ax=ax,
            color='black', alpha=0.05, size=1,
            jitter=True, dodge=False
        )

        measure_name = var_labels.get(measure, measure)
        ax.set_title(f'Distribution of {measure_name}', fontsize=14)
        sns.despine(ax=ax)
        ax.set_xlabel('Menopausal Stage', fontsize=10)
        ax.set_ylabel(measure_name, fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        means_by_stage = data.groupby('STATUS_Label', observed=True)[measure].mean().round(3)
        medians_by_stage = data.groupby('STATUS_Label', observed=True)[measure].median().round(3)
        counts_by_stage = data.groupby('STATUS_Label', observed=True)[measure].count()

        stats_text = "Group Statistics:\n"
        for stage in data['STATUS_Label'].cat.categories:
            if stage in means_by_stage:
                stats_text += f"{stage} (n={counts_by_stage[stage]}):\n"
                stats_text += f"  μ={means_by_stage[stage]}, Md={medians_by_stage[stage]}\n"

        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               va='top', ha='right', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    fig.suptitle(
        'Distribution of Cognitive Measures Across Menopausal Stages',
        fontsize=16, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'outcome_violin_plots.png'), dpi=300)
    plt.close()


def plot_stages_vs_scores(analysis_obj):
    """Create line plots showing mean scores across menopausal stages with error bars."""
    set_apa_style()

    data = analysis_obj.data
    output_dir = analysis_obj.output_dir

    transformed_vars = ['TOTIDE1', 'TOTIDE2']  # Only cognitive outcomes
    var_labels = {
        'TOTIDE1': 'Cognitive Performance (Immediate Recall)',
        'TOTIDE2': 'Cognitive Performance (Delayed Recall)'
    }

    plot_data = data.copy()

    stage_order = ['Premenopause', 'Early Perimenopause', 'Late Perimenopause', 'Postmenopause', 'Surgical']
    plot_data['STATUS_Label'] = pd.Categorical(
        plot_data['STATUS_Label'],
        categories=stage_order,
        ordered=True
    )

    melted_data = pd.melt(
        plot_data,
        id_vars=['SWANID', 'STATUS_Label'],
        value_vars=transformed_vars,
        var_name='Measure',
        value_name='Score'
    )

    melted_data['Measure_Label'] = melted_data['Measure'].map(var_labels)

    summary_data = melted_data.groupby(['Measure', 'Measure_Label', 'STATUS_Label'], observed=True).agg(
        Mean=('Score', 'mean'),
        SE=('Score', lambda x: x.std() / np.sqrt(len(x))),
        Count=('Score', 'count')
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    line_color = CONSTRUCT_COLORS['cognitive_function']
    errorbar_color = '#004488'

    for i, measure in enumerate(transformed_vars):
        ax = axes[i]
        measure_data = summary_data[summary_data['Measure'] == measure]

        ax.set_title(var_labels.get(measure, measure), fontsize=14)
        sns.despine(ax=ax)

        if len(measure_data) == 0:
            ax.text(0.5, 0.5, f"No data available for {var_labels.get(measure, measure)}",
                   ha='center', va='center', fontsize=12)
            continue

        ax.errorbar(
            x=measure_data['STATUS_Label'].cat.codes,
            y=measure_data['Mean'],
            yerr=measure_data['SE'] * 1.96,
            fmt='o-', linewidth=2, markersize=8, capsize=5,
            color=line_color, ecolor=errorbar_color,
            label=var_labels.get(measure, measure)
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

        ax.set_title(var_labels.get(measure, measure), fontsize=18)
        ax.set_xticks(range(len(stage_order)))
        ax.set_xticklabels(stage_order, rotation=45, ha='right', fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, linestyle=':', alpha=0.6)

        min_val = measure_data['Mean'].min() if not measure_data.empty else 0
        max_val = measure_data['Mean'].max() if not measure_data.empty else 0
        if min_val < 0 < max_val:
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    file_name = os.path.join(output_dir, 'stages_vs_scores_faceted.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trajectory_classes(analysis_obj):
    """Plot longitudinal trajectories showing how outcomes change over time by stage."""
    set_apa_style()

    data = analysis_obj.data
    output_dir = analysis_obj.output_dir

    if 'VISIT' not in data.columns:
        return

    trajectory_data = data.copy()

    measures_to_plot = ['TOTIDE1', 'TOTIDE2']  # Only cognitive outcomes
    measure_labels_short = {
        'TOTIDE1': 'Immediate Recall Score',
        'TOTIDE2': 'Delayed Recall Score'
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    stage_order = ['Premenopause', 'Early Perimenopause', 'Late Perimenopause', 'Postmenopause']

    stage_colors_map = {
        'Premenopause': STAGE_COLORS['Premenopause'],
        'Early Perimenopause': STAGE_COLORS['Early Perimenopause'],
        'Late Perimenopause': STAGE_COLORS['Late Perimenopause'],
        'Postmenopause': STAGE_COLORS['Postmenopause']
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
                     title_fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1),
                     frameon=True, fancybox=True, shadow=True)

        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=11)
        sns.despine(ax=ax)

    fig.suptitle('Longitudinal Trajectories of Cognitive Outcomes\nAcross Menopausal Transition',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = os.path.join(output_dir, 'trajectory_classes.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decline_proportions_mcid(analysis_obj):
    """Plot proportions of participants showing clinically meaningful decline (> MCID) by stage."""
    set_apa_style()

    data = analysis_obj.data
    output_dir = analysis_obj.output_dir

    # Calculate MCID thresholds
    mcid_thresholds = _calculate_mcid_thresholds(data)

    # Only analyze cognitive measures
    measures = ['TOTIDE1', 'TOTIDE2']
    measure_labels = {
        'TOTIDE1': 'Cognitive Performance\n(Immediate Recall)',
        'TOTIDE2': 'Cognitive Performance\n(Delayed Recall)'
    }

    stage_order = ['Premenopause', 'Early Perimenopause', 'Late Perimenopause', 'Postmenopause', 'Surgical']

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, measure in enumerate(measures):
        ax = axes[idx]

        if measure not in mcid_thresholds:
            continue

        mcid = mcid_thresholds[measure]

        # Calculate baseline (premenopause) mean
        baseline_data = data[data['STATUS_Label'] == 'Premenopause']
        baseline_mean = pd.to_numeric(baseline_data[measure], errors='coerce').mean()

        # For each stage, calculate proportion with clinically meaningful decline
        proportions = []
        counts = []
        total_ns = []

        for stage in stage_order:
            stage_data = data[data['STATUS_Label'] == stage]
            stage_values = pd.to_numeric(stage_data[measure], errors='coerce').dropna()

            if len(stage_values) == 0:
                proportions.append(0)
                counts.append(0)
                total_ns.append(0)
                continue

            # Calculate decline from baseline
            declines = baseline_mean - stage_values

            # Count those with decline > MCID
            meaningful_decline = (declines > mcid).sum()
            total = len(stage_values)

            proportions.append((meaningful_decline / total) * 100)
            counts.append(meaningful_decline)
            total_ns.append(total)

        # Create bar plot
        x_pos = np.arange(len(stage_order))
        bars = ax.bar(x_pos, proportions, color=get_categorical_palette(len(stage_order)),
                     edgecolor='black', linewidth=1.5)

        # Add percentage labels on top of bars
        for i, (bar, proportion, total_n) in enumerate(zip(bars, proportions, total_ns)):
            height = bar.get_height()
            if total_n > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{proportion:.1f}%',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Formatting
        ax.set_xlabel('Menopausal Stage', fontsize=12)
        ax.set_ylabel('Proportion with Clinically\nMeaningful Decline (%)', fontsize=12)
        ax.set_title(measure_labels.get(measure, measure), fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_order, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, max(proportions) * 1.25 if max(proportions) > 0 else 10)
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)

        # Add MCID threshold note with improved styling
        ax.text(0.02, 0.98, f'MCID Threshold: {mcid:.2f} points\n(0.5 × baseline SD)',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                        edgecolor='navy', alpha=0.8, linewidth=1.5))

        sns.despine(ax=ax)

    fig.suptitle('Proportion of Participants with Clinically Meaningful Cognitive Decline\n(Decline > MCID from Premenopause Baseline)',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'menopausal_decline_proportions_mcid.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Decline proportions plot saved to: 1_stages_model/output/menopausal_decline_proportions_mcid.png")


def create_all_visualizations(analysis_obj):
    """Generate all visualization outputs."""
    plot_violin_plots(analysis_obj)
    plot_stages_vs_scores(analysis_obj)
    plot_trajectory_classes(analysis_obj)
    plot_decline_proportions_mcid(analysis_obj)


def _calculate_reasonable_limits(coefs, errors, percentile=95):
    """Calculate axis limits using percentiles to avoid extreme outliers."""
    all_ends = []
    for coef, error in zip(coefs, errors):
        all_ends.append(coef + (error * 1.96))
        all_ends.append(coef - (error * 1.96))

    if all_ends:
        min_val = np.percentile(all_ends, 100 - percentile)
        max_val = np.percentile(all_ends, percentile)

        min_val = min(min_val, 0)
        max_val = max(max_val, 0)

        range_val = max_val - min_val
        min_val -= range_val * 0.1
        max_val += range_val * 0.1

        return min_val, max_val

    return -1, 1


def _calculate_mcid_thresholds(data):
    """Calculate MCID thresholds using 0.5 * SD from pre-menopause baseline."""
    mcid_thresholds = {}

    if 'STATUS_Label' in data.columns:
        baseline_data = data[data['STATUS_Label'] == 'Premenopause']
    else:
        baseline_data = data

    for measure in ['TOTIDE1', 'TOTIDE2']:
        if measure in data.columns:
            measure_data = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
            if len(measure_data) > 0:
                mcid_thresholds[measure] = 0.5 * measure_data.std()

    return mcid_thresholds


def plot_forest_plot_from_models(analysis_obj):
    """Create forest plot showing stage effects with confidence intervals."""
    set_apa_style()

    if not analysis_obj.mixed_model_results:
        return

    measure_labels = {
        'TOTIDE1': 'Cognitive Performance (Immediate Recall)',
        'TOTIDE2': 'Cognitive Performance (Delayed Recall)'
    }

    status_effects = ['Early Perimenopause', 'Late Perimenopause', 'Postmenopause', 'Surgical']

    mcid_thresholds = _calculate_mcid_thresholds(analysis_obj.data)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
    axes = axes.flatten()

    for i, (outcome, results) in enumerate(analysis_obj.mixed_model_results.items()):
        ax = axes[i]
        coefs, errors, pvalues, names = [], [], [], []

        for status in status_effects:
            param_name = f"C(STATUS_Label, Treatment('Premenopause'))[T.{status}]"
            if param_name in results.params.index:
                coefs.append(results.params[param_name])
                errors.append(results.bse[param_name])
                pvalues.append(results.pvalues[param_name])
                names.append(status)

        if not coefs:
            ax.set_visible(False)
            continue

        y_positions = np.arange(len(names))
        min_bound, max_bound = _calculate_reasonable_limits(coefs, errors, percentile=95)
        ax.set_xlim(min_bound, max_bound)

        for y, coef, error, p in zip(y_positions, coefs, errors, pvalues):
            color, marker = get_significance_color(p)

            lower_error = max(min_bound * 0.95, coef - error * 1.96)
            upper_error = min(max_bound * 0.95, coef + error * 1.96)

            left_err = coef - lower_error
            right_err = upper_error - coef

            ax.errorbar(
                x=coef, y=y,
                xerr=[[left_err], [right_err]],
                fmt='o', color=color,
                capsize=5, markersize=8,
                elinewidth=2, capthick=2
            )

            label_x = upper_error + (max_bound - min_bound) * 0.02 + 0.01
            ax.text(
                label_x, y, f'{coef:.3f} {marker}',
                va='center', ha='right', color=color,
                fontweight='bold' if p < 0.05 else 'normal',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')
            )

            if coef - error * 1.96 < min_bound * 0.95:
                ax.annotate('', xy=(min_bound * 0.98, y), xytext=(min_bound * 0.9, y),
                          arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            if coef + error * 1.96 > max_bound * 0.95:
                ax.annotate('', xy=(max_bound * 0.98, y), xytext=(max_bound * 0.9, y),
                          arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Add MCID reference lines for cognitive measures
        if outcome in mcid_thresholds:
            mcid = mcid_thresholds[outcome]
            ax.axvline(x=mcid, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'MCID threshold')
            ax.axvline(x=-mcid, color='orange', linestyle=':', linewidth=2, alpha=0.7)

            # Add text annotation for MCID - positioned above the plot to avoid overlap
            ax.text(mcid, len(names), f'  MCID: ±{mcid:.2f}',
                   fontsize=9, color='orange', va='bottom', ha='left',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='orange'))

        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_title(measure_labels.get(outcome, outcome), fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.4)
        sns.despine(ax=ax)

    fig.text(0.5, 0.01, '* p<0.05   ** p<0.01   *** p<0.001', ha='center', fontsize=12)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(wspace=0.3)
    except:
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05, wspace=0.3)

    file_name = os.path.join(analysis_obj.output_dir, 'model_forest_plot.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
