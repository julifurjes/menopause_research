import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.plot_config import STAGE_COLORS, set_apa_style


def plot_moderation_effects(analysis_obj):
    """Create visualizations showing moderation effects."""
    if 'moderation' not in analysis_obj.results:
        print("Run fit_moderation_models() first")
        return

    result = analysis_obj.results['moderation']

    # Extract coefficients
    params = result.params

    # Create visualization of simple slopes at different stages
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Apply APA style
    set_apa_style()

    # Use colorblind-friendly colors
    colors = STAGE_COLORS

    # Plot 1: Simple slopes
    ax = axes[0]

    # Range of social support values (centered)
    support_range = np.linspace(-2, 2, 100)

    # Calculate predicted cognitive function for each stage
    for stage, color in colors.items():
        # Base prediction
        pred = params['Intercept'] + params['social_support_centered'] * support_range

        # Add stage-specific effects
        if stage == 'Early Peri':
            pred += params['Early_Peri']
            pred += params['Early_Peri_x_Support'] * support_range
        elif stage == 'Late Peri':
            pred += params['Late_Peri']
            pred += params['Late_Peri_x_Support'] * support_range
        elif stage == 'Post-menopause':
            pred += params['Post_Menopause']
            pred += params['Post_Menopause_x_Support'] * support_range
        elif stage == 'Surgical':
            pred += params['Surgical']
            pred += params['Surgical_x_Support'] * support_range

        ax.plot(support_range, pred, label=stage, color=color, linewidth=3)

    ax.set_xlabel('Social Support (Centered)', fontsize=12)
    ax.set_ylabel('Predicted Cognitive Function', fontsize=12)
    ax.set_title('Moderation Effect: Social Support × Menopausal Stage', fontsize=14)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    sns.despine(ax=ax)

    # Plot 2: Interaction coefficients
    ax = axes[1]

    stages = ['Pre-menopause\n(reference)', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']
    interaction_coefs = [
        0,  # Reference
        params.get('Early_Peri_x_Support', 0),
        params.get('Late_Peri_x_Support', 0),
        params.get('Post_Menopause_x_Support', 0),
        params.get('Surgical_x_Support', 0)
    ]

    # Get standard errors for confidence intervals
    bse = result.bse
    interaction_ses = [
        0,
        bse.get('Early_Peri_x_Support', 0),
        bse.get('Late_Peri_x_Support', 0),
        bse.get('Post_Menopause_x_Support', 0),
        bse.get('Surgical_x_Support', 0)
    ]

    y_pos = np.arange(len(stages))

    # Plot bars
    bars = ax.barh(y_pos, interaction_coefs, color=[colors[s.split('\n')[0]] for s in stages])

    # Add error bars
    ax.errorbar(interaction_coefs, y_pos, xerr=[1.96 * se for se in interaction_ses],
               fmt='none', color='black', capsize=5)

    # Add value labels
    for i, (coef, se) in enumerate(zip(interaction_coefs, interaction_ses)):
        if i > 0:  # Skip reference
            p_val = 2 * (1 - stats.norm.cdf(abs(coef / se))) if se > 0 else 1
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            ax.text(coef, i, f'  {coef:.3f}{sig}', va='center', fontsize=12, fontweight='bold')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=11)
    ax.set_xlabel('Interaction Coefficient\n(Social Support × Stage Effect)', fontsize=12)
    ax.set_title('Strength of Moderation by Stage', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    sns.despine(ax=ax)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(analysis_obj.output_dir, 'moderation_effects.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nModeration effects plot saved to: 3_social_model/output/moderation_effects.png")
