"""
Script to update all visualization files to use colorblind-friendly palettes and APA formatting.
Run this from the project root directory.
"""
import re
import sys

# Files to update
files_to_update = [
    '1_stages_model/visualisations.py',
    '1_stages_model/proportion_analysis.py',
    '2_symptoms_model/analysis.py',
    'preparations/impute_data.py',
    'utils/data_validation.py'
]

# Regex patterns for finding and replacing color definitions
patterns = [
    # Replace green_palette definitions
    (r'green_palette\s*=\s*sns\.color_palette\(["\']YlGn["\'],\s*n_colors=\d+\)',
     'from utils.plot_config import get_categorical_palette, STAGE_COLORS, CONSTRUCT_COLORS, set_apa_style'),

    # Replace cmap='YlGn' or similar
    (r"cmap\s*=\s*['\"]YlGn['\"]",
     "cmap=get_categorical_palette(10)"),

    # Replace cmap='coolwarm'
    (r"cmap\s*=\s*['\"]coolwarm['\"]",
     "cmap='RdBu_r'"),
]

def update_file(filepath):
    """Update a single file with new color scheme."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Check if plot_config is already imported
        if 'from utils.plot_config import' not in content:
            # Add import after other utils imports
            if 'from utils.save_output import' in content:
                content = content.replace(
                    'from utils.save_output import',
                    'from utils.plot_config import (STAGE_COLORS, CONSTRUCT_COLORS, SIGNIFICANCE_COLORS,\n                               get_significance_color, set_apa_style, get_categorical_palette)\nfrom utils.save_output import'
                )
            elif 'import matplotlib.pyplot as plt' in content:
                # Insert after matplotlib import
                content = content.replace(
                    'import matplotlib.pyplot as plt',
                    'import matplotlib.pyplot as plt\nimport sys\nimport os\nproject_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nif project_root not in sys.path:\n    sys.path.append(project_root)\nfrom utils.plot_config import (STAGE_COLORS, CONSTRUCT_COLORS, SIGNIFICANCE_COLORS,\n                               get_significance_color, set_apa_style, get_categorical_palette)'
                )

        # Replace green_palette references with colorblind-friendly colors
        content = re.sub(r'green_palette\s*=\s*sns\.color_palette\(["\']YlGn["\'],\s*n_colors=\d+\)',
                        'stage_colors = list(STAGE_COLORS.values())', content)

        # Replace cmap references
        content = re.sub(r"cmap\s*=\s*['\"]YlGn['\"]",
                        "cmap=sns.light_palette('#228833', n_colors=10, as_cmap=True)", content)
        content = re.sub(r"cmap\s*=\s*['\"]coolwarm['\"]",
                        "cmap='RdBu_r'", content)

        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return False
    except Exception as e:
        print(f"Error updating {filepath}: {str(e)}")
        return False

def main():
    """Update all visualization files."""
    print("Updating visualization files to use colorblind-friendly colors and APA formatting...")
    print("-" * 80)

    updated_count = 0
    for filepath in files_to_update:
        if update_file(filepath):
            updated_count += 1

    print("-" * 80)
    print(f"Complete! Updated {updated_count} out of {len(files_to_update)} files.")
    print("\nNote: Some files may require manual adjustment for specific color references.")
    print("Run the analysis scripts to verify the changes work correctly.")

if __name__ == '__main__':
    main()
