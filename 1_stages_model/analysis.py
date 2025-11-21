import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import os
import sys

from visualisations import MenopauseVisualisations
from proportion_analysis import MenopauseDeclineAnalysis

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir
from utils.plot_config import get_significance_color, set_apa_style

class MenopauseCognitionAnalysis:
    """Analysis of cognitive and emotional outcomes across menopausal stages using mixed-effects models."""

    def __init__(self, file_path, use_langcog=True):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.mixed_model_results = {}
        self.output_dir = get_output_dir('1_stages_model')
        self.use_langcog = use_langcog

    def transform_variables(self):
        """Apply log and sqrt transformations to address skewness."""
        self.data['NERVES_log'] = np.log1p(self.data['NERVES'])
        self.data['SAD_sqrt'] = np.sqrt(self.data['SAD'])
        self.data['FEARFULA_sqrt'] = np.sqrt(self.data['FEARFULA'])
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES_log', 'SAD_sqrt', 'FEARFULA_sqrt']

        if self.use_langcog and 'LANGCOG' in self.data.columns:
            self.data['LANGCOG'] = self.data['LANGCOG'].astype('category')

    def filter_status(self):
        """Filter to relevant menopausal stages and create categorical labels."""
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]

        status_map = {
            1: 'Surgical', 2: 'Post-menopause', 3: 'Late Peri',
            4: 'Early Peri', 5: 'Pre-menopause', 8: 'Surgical'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        self.data['Menopause_Type'] = np.where(self.data['STATUS'].isin([1, 8]), 'Surgical', 'Natural')

        natural_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=['Surgical'] + natural_order,
            ordered=True
        )

        self.data['AGE'] = pd.to_numeric(self.data['AGE'], errors='coerce')
        self.data['VISIT'] = pd.to_numeric(self.data['VISIT'], errors='coerce')
        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        self.data['AGE_BASELINE'] = self.data.groupby('SWANID')['AGE'].transform('first')

    def run_mixed_models(self):
        """Run mixed-effects models with random intercepts for each subject."""
        if self.use_langcog and 'LANGCOG' in self.data.columns:
            reference_language = self.data['LANGCOG'].mode()[0]

        for var in self.outcome_vars:
            self.data[var] = pd.to_numeric(self.data[var], errors='coerce')

        covariates = ['AGE_BASELINE']
        if self.use_langcog and 'LANGCOG' in self.data.columns:
            covariates.append('LANGCOG')

        self.mixed_model_results = {}

        for outcome in self.outcome_vars:
            if self.use_langcog and 'LANGCOG' in self.data.columns:
                covariate_formula = f"AGE_BASELINE + C(LANGCOG, Treatment({reference_language}))"
            else:
                covariate_formula = "AGE_BASELINE"
            formula = f"{outcome} ~ C(STATUS_Label, Treatment('Pre-menopause')) + VISIT + {covariate_formula}"

            try:
                analysis_data = self.data.dropna(subset=[outcome] + covariates if covariates else [outcome])

                status_counts = analysis_data['STATUS_Label'].value_counts()
                analysis_data['weights'] = analysis_data['STATUS_Label'].map(
                    lambda x: 1 / (status_counts[x] / sum(status_counts))
                )

                model = mixedlm(
                    formula=formula,
                    groups=analysis_data["SWANID"],
                    data=analysis_data,
                    re_formula="~VISIT"
                )

                results = model.fit(reml=True, weights=analysis_data['weights'])
                self.mixed_model_results[outcome] = results

                print(f"\nMixed Model Results for {outcome}")
                print("=" * 50)
                print(results.summary())

                try:
                    resid_var = results.scale
                    re_var = results.cov_re.iloc[0, 0] if hasattr(results.cov_re, 'iloc') else results.cov_re[0][0]
                    var_fixed = np.var(results.predict(analysis_data))

                    marginal_r2 = var_fixed / (var_fixed + re_var + resid_var)
                    conditional_r2 = (var_fixed + re_var) / (var_fixed + re_var + resid_var)

                    print(f"\nMarginal R² (fixed effects): {marginal_r2:.4f}")
                    print(f"Conditional R² (fixed + random): {conditional_r2:.4f}")

                except Exception as e:
                    print(f"Error calculating R-squared: {str(e)}")

                self.check_model_diagnostics(results, outcome, analysis_data)

            except Exception as e:
                print(f"Error in mixed model for {outcome}: {str(e)}")
    
    def check_model_diagnostics(self, model_results, outcome, data):
        """Generate diagnostic plots for mixed model residuals."""
        try:
            predicted = model_results.predict(data)
            residuals = data[outcome] - predicted

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            axes[0, 0].scatter(predicted, residuals, alpha=0.5)
            axes[0, 0].axhline(y=0, color='r', linestyle='-')
            axes[0, 0].set_title(f'Residuals vs Fitted for {outcome}')
            axes[0, 0].set_xlabel('Fitted values')
            axes[0, 0].set_ylabel('Residuals')

            sns.histplot(residuals, kde=True, ax=axes[0, 1])
            axes[0, 1].set_title('Histogram of Residuals')

            sm.qqplot(residuals.dropna(), line='s', ax=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot of Residuals')

            axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5)
            axes[1, 1].axhline(y=0, color='r', linestyle='-')
            axes[1, 1].set_title('Residuals vs Order')
            axes[1, 1].set_xlabel('Observation Order')
            axes[1, 1].set_ylabel('Residuals')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{outcome}_mixed_diagnostics.png'))
            plt.close()

        except Exception as e:
            print(f"Error in model diagnostics: {str(e)}")

    def _calculate_reasonable_limits(self, coefs, errors, percentile=95):
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

    def plot_forest_plot_from_models(self):
        """Create forest plot showing stage effects with confidence intervals."""
        if not self.mixed_model_results:
            return

        measure_labels = {
            'TOTIDE1': 'Cognitive Performance (Immediate Recall)',
            'TOTIDE2': 'Cognitive Performance (Delayed Recall)',
            'NERVES_log': 'Nervousness',
            'SAD_sqrt': 'Sadness',
            'FEARFULA_sqrt': 'Fearfulness'
        }

        status_effects = ['Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']

        fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=False)
        axes = axes.flatten()
        set_apa_style()

        for i, (outcome, results) in enumerate(self.mixed_model_results.items()):
            ax = axes[i]
            coefs, errors, pvalues, names = [], [], [], []

            for status in status_effects:
                param_name = f"C(STATUS_Label, Treatment('Pre-menopause'))[T.{status}]"
                if param_name in results.params.index:
                    coefs.append(results.params[param_name])
                    errors.append(results.bse[param_name])
                    pvalues.append(results.pvalues[param_name])
                    names.append(status)

            if not coefs:
                ax.set_visible(False)
                continue

            y_positions = np.arange(len(names))
            min_bound, max_bound = self._calculate_reasonable_limits(coefs, errors, percentile=95)
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
            ax.set_yticks(y_positions)
            ax.set_yticklabels(names, fontsize=11)
            ax.set_title(measure_labels.get(outcome, outcome), fontsize=14)
            ax.grid(True, linestyle=':', alpha=0.4)
            sns.despine(ax=ax)

        fig.text(0.5, 0.01, '* p<0.05   ** p<0.01   *** p<0.001', ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(wspace=0.3)

        file_name = os.path.join(self.output_dir, 'model_forest_plot.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.transform_variables()
        self.filter_status()

        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            print("\nRunning mixed-effects models...")
            self.run_mixed_models()
            self.plot_forest_plot_from_models()
            print("\nAnalysis complete.")
        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    # Main analysis: cognitive and emotional outcomes by menopausal stage
    analysis = MenopauseCognitionAnalysis("processed_combined_data.csv", use_langcog=False)
    analysis.run_complete_analysis()

    # Proportion analysis: women experiencing cognitive/emotional decline
    proportion_analysis = MenopauseDeclineAnalysis(analysis.data)
    proportion_analysis.run_analysis()

    # Visualizations: violin plots, trajectories, and stage comparisons
    viz = MenopauseVisualisations(analysis.data)
    viz.create_all_visualizations()