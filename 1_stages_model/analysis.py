import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from visualizations import create_all_visualizations, plot_forest_plot_from_models

class MenopauseCognitionAnalysis:
    """Analysis of cognitive and emotional outcomes across menopausal stages using mixed-effects models."""

    def __init__(self, file_path, use_langcog=True):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.mixed_model_results = {}
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
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
                analysis_data = self.data.dropna(subset=[outcome] + covariates if covariates else [outcome]).copy()

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


    def interpret_clinical_significance(self):
        """Interpret model results using MCID thresholds for clinical significance."""
        if not self.mixed_model_results:
            return

        from visualizations import _calculate_mcid_thresholds
        mcid_thresholds = _calculate_mcid_thresholds(self.data)

        print("\n" + "="*80)
        print("CLINICAL SIGNIFICANCE INTERPRETATION")
        print("="*80)
        print("\nMethodology:")
        print("  - MCID (Minimally Clinically Important Difference) = 0.5 x baseline SD")
        print("  - Baseline SD calculated from pre-menopause group")
        print("  - Clinical significance: |effect| >= MCID threshold")
        print("="*80)

        for outcome, results in self.mixed_model_results.items():
            if outcome in mcid_thresholds:
                mcid = mcid_thresholds[outcome]
                baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause']
                baseline_sd = pd.to_numeric(baseline_data[outcome], errors='coerce').dropna().std()

                print(f"\n{outcome}:")
                print(f"  Baseline SD: {baseline_sd:.3f}")
                print(f"  MCID threshold: {mcid:.3f} points (0.5 x {baseline_sd:.3f})")
                print("-" * 80)

                for stage in ['Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']:
                    param_name = f"C(STATUS_Label, Treatment('Pre-menopause'))[T.{stage}]"

                    if param_name in results.params.index:
                        coef = results.params[param_name]
                        p_val = results.pvalues[param_name]

                        stat_sig = p_val < 0.05
                        clin_sig = abs(coef) >= mcid

                        sig_stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                        print(f"\n  {stage}:")
                        print(f"    Effect size: {coef:+.3f} points {sig_stars}")
                        print(f"    p-value: {p_val:.4f}")
                        print(f"    Statistically significant: {'Yes' if stat_sig else 'No'} (p {'<' if stat_sig else '>='} 0.05)")
                        print(f"    Clinically meaningful: {'Yes' if clin_sig else 'No'} (|{coef:.3f}| {'>=' if clin_sig else '<'} {mcid:.3f})")

                        if stat_sig and clin_sig:
                            print(f"    -> BOTH statistically significant AND clinically meaningful")
                        elif stat_sig and not clin_sig:
                            print(f"    -> Statistically significant but NOT clinically meaningful")
                        elif not stat_sig and clin_sig:
                            print(f"    -> Clinically meaningful but NOT statistically significant")
                        else:
                            print(f"    -> Neither statistically nor clinically significant")

        print("\n" + "="*80)

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        self.transform_variables()
        self.filter_status()

        print("\nRunning mixed-effects models...")
        self.run_mixed_models()
        self.interpret_clinical_significance()
        plot_forest_plot_from_models(self)
        print("\nAnalysis complete.")

if __name__ == "__main__":
    # Main analysis: cognitive and emotional outcomes by menopausal stage
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_combined_data.csv")
    analysis = MenopauseCognitionAnalysis(data_path, use_langcog=False)
    analysis.run_complete_analysis()

    # Visualizations: violin plots, trajectories, and stage comparisons
    create_all_visualizations(analysis)