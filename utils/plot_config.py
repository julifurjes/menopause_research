"""
Color and style configuration for all visualizations.
Uses colorblind-friendly palettes and APA formatting standards.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-friendly palettes based on Paul Tol's schemes and ColorBrewer
# These are tested for various types of colorblindness

# Categorical colors for menopausal stages (consistent across all models)
STAGE_COLORS = {
    'Pre-menopause': '#4477AA',     # Blue
    'Early Peri': '#66CCEE',        # Cyan
    'Late Peri': '#228833',         # Green
    'Post-menopause': '#CCBB44',    # Yellow
    'Surgical': '#EE6677'           # Red/Pink
}

# Categorical colors for constructs/variables (consistent across models)
CONSTRUCT_COLORS = {
    'cognitive_function': '#4477AA',      # Blue
    'Cognitive Function': '#4477AA',
    'social_support': '#228833',          # Green
    'Social Support': '#228833',
    'emotional_struggle': '#EE6677',      # Red/Pink
    'Emotional Struggle': '#EE6677',
    'social_struggle': '#CCBB44',         # Yellow
    'Social Struggle': '#CCBB44',
    'symptom_severity': '#AA3377',        # Purple
    'Symptom Severity': '#AA3377',
    'socioeconomic_status': '#66CCEE',    # Cyan
    'Socioeconomic Status': '#66CCEE'
}

# Symptom-specific colors
SYMPTOM_COLORS = {
    'Hot Flashes': '#EE6677',      # Red/Pink
    'Night Sweats': '#AA3377',     # Purple
    'Cold Sweats': '#66CCEE',      # Cyan
    'Stiffness': '#228833',        # Green
    'Irritability': '#CCBB44',     # Yellow
    'Mood Changes': '#4477AA'      # Blue
}

# Diverging colormap for correlations (colorblind-friendly)
# Blue-White-Red scheme
CORRELATION_CMAP = 'RdBu_r'

# Sequential colorblind-friendly palette for significance levels
SIGNIFICANCE_COLORS = {
    'p<0.001': '#004488',    # Dark blue (most significant)
    'p<0.01': '#117733',     # Dark green (very significant)
    'p<0.05': '#DDAA33',     # Orange (significant)
    'ns': '#888888'          # Gray (not significant)
}

def get_significance_color(p_value):
    """Return color and label based on p-value."""
    if p_value < 0.001:
        return SIGNIFICANCE_COLORS['p<0.001'], '***'
    elif p_value < 0.01:
        return SIGNIFICANCE_COLORS['p<0.01'], '**'
    elif p_value < 0.05:
        return SIGNIFICANCE_COLORS['p<0.05'], '*'
    else:
        return SIGNIFICANCE_COLORS['ns'], ''

# APA Style Settings
APA_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'axes.linewidth': 1,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}

def set_apa_style():
    """Apply APA style settings to all plots."""
    plt.rcParams.update(APA_STYLE)
    sns.set_style("ticks", {
        'axes.spines.top': False,
        'axes.spines.right': False
    })

def get_categorical_palette(n_colors):
    """
    Get a colorblind-friendly categorical palette.
    Based on Paul Tol's bright scheme.
    """
    bright_palette = [
        '#4477AA',  # Blue
        '#EE6677',  # Red
        '#228833',  # Green
        '#CCBB44',  # Yellow
        '#66CCEE',  # Cyan
        '#AA3377',  # Purple
        '#BBBBBB'   # Gray
    ]
    return bright_palette[:n_colors]

def get_sequential_palette(n_colors, color='blue'):
    """
    Get a colorblind-friendly sequential palette.
    """
    if color == 'blue':
        return sns.light_palette('#4477AA', n_colors=n_colors)
    elif color == 'green':
        return sns.light_palette('#228833', n_colors=n_colors)
    elif color == 'red':
        return sns.light_palette('#EE6677', n_colors=n_colors)
    else:
        return sns.light_palette('#4477AA', n_colors=n_colors)

# Initialize APA style as default when module is imported
set_apa_style()
