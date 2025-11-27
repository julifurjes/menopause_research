import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
from scipy import stats
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from visualizations import plot_moderation_effects

class ModerationAnalysis:
    """Analyze moderation effects of social support on cognitive outcomes across menopausal stages."""

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.social_support_vars = ['LISTEN', 'TAKETOM', 'HELPSIC', 'CONFIDE']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.control_vars = ['AGE_BASELINE', 'VISIT']
        self.results = {}

    def preprocess_data(self):
        """Prepare data by creating composite scores and interaction terms."""
        relevant_vars = (self.social_support_vars + self.cognitive_vars +
                        ['STATUS', 'SWANID', 'VISIT', 'AGE'])

        for col in relevant_vars:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]

        status_map = {
            1: 'Surgical', 2: 'Post-menopause', 3: 'Late Peri',
            4: 'Early Peri', 5: 'Pre-menopause', 8: 'Surgical'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        self.data = self.data.sort_values(['SWANID', 'VISIT'])

        if 'AGE' in self.data.columns:
            self.data['AGE_BASELINE'] = self.data.groupby('SWANID')['AGE'].transform('first')

        self.data['social_support'] = self.data[self.social_support_vars].mean(axis=1)
        self.data['cognitive_function'] = self.data[self.cognitive_vars].mean(axis=1)

        self.data['social_support_centered'] = (self.data['social_support'] -
                                                 self.data['social_support'].mean())
        # Create dummy variables for all menopausal stages (Pre-menopause will be reference)
        self.data['Early_Peri'] = (self.data['STATUS_Label'] == 'Early Peri').astype(int)
        self.data['Late_Peri'] = (self.data['STATUS_Label'] == 'Late Peri').astype(int)
        self.data['Post_Menopause'] = (self.data['STATUS_Label'] == 'Post-menopause').astype(int)
        self.data['Surgical'] = (self.data['STATUS_Label'] == 'Surgical').astype(int)

        # Create interaction terms for all stages
        self.data['Early_Peri_x_Support'] = self.data['Early_Peri'] * self.data['social_support_centered']
        self.data['Late_Peri_x_Support'] = self.data['Late_Peri'] * self.data['social_support_centered']
        self.data['Post_Menopause_x_Support'] = self.data['Post_Menopause'] * self.data['social_support_centered']
        self.data['Surgical_x_Support'] = self.data['Surgical'] * self.data['social_support_centered']

        # Drop missing values
        required_vars = (['social_support', 'cognitive_function', 'AGE_BASELINE', 'VISIT'] +
                        ['Early_Peri', 'Late_Peri', 'Post_Menopause', 'Surgical'])
        self.data = self.data.dropna(subset=required_vars).copy()

        # Ensure SWANID is string for grouping
        self.data['SWANID'] = self.data['SWANID'].astype(str)

        print(f"\nData preprocessed: {len(self.data)} observations from {self.data['SWANID'].nunique()} subjects")
        print(f"\nObservations by stage:")
        print(self.data['STATUS_Label'].value_counts())

    def fit_moderation_models(self):
        """Fit mixed-effects models testing moderation of social support effects by menopausal stage."""
        print("MODERATION ANALYSIS: Social Support × Menopausal Stage")
        print("=" * 80)

        # Model 1: Main effects only (no interaction)
        print("\nModel 1: Main Effects Only")
        print("-" * 80)

        formula_main = ("cognitive_function ~ social_support_centered + "
                       "Early_Peri + Late_Peri + Post_Menopause + Surgical + "
                       "AGE_BASELINE + VISIT")

        model_main = mixedlm(formula=formula_main,
                            data=self.data,
                            groups=self.data["SWANID"],
                            re_formula="~1")  # Random intercept only

        # Fit with ML for model comparison (not REML)
        result_main_ml = model_main.fit(reml=False, method='lbfgs')

        # Also fit with REML for final parameter estimates
        result_main_reml = model_main.fit(reml=True, method='lbfgs')
        self.results['main_effects'] = result_main_reml

        print(result_main_reml.summary())
        print(f"\nAIC (ML): {result_main_ml.aic:.2f}")
        print(f"BIC (ML): {result_main_ml.bic:.2f}")
        print(f"Log-Likelihood (ML): {result_main_ml.llf:.2f}")

        # Model 2: With interaction terms (moderation model)
        print("\n" + "=" * 80)
        print("Model 2: Moderation Model (with interactions)")
        print("-" * 80)

        formula_interaction = ("cognitive_function ~ social_support_centered + "
                              "Early_Peri + Late_Peri + Post_Menopause + Surgical + "
                              "Early_Peri_x_Support + Late_Peri_x_Support + Post_Menopause_x_Support + Surgical_x_Support + "
                              "AGE_BASELINE + VISIT")

        model_interaction = mixedlm(formula=formula_interaction,
                                    data=self.data,
                                    groups=self.data["SWANID"],
                                    re_formula="~1")

        # Fit with ML for model comparison (not REML)
        result_interaction_ml = model_interaction.fit(reml=False, method='lbfgs')

        # Also fit with REML for final parameter estimates
        result_interaction_reml = model_interaction.fit(reml=True, method='lbfgs')
        self.results['moderation'] = result_interaction_reml

        print(result_interaction_reml.summary())
        print(f"\nAIC (ML): {result_interaction_ml.aic:.2f}")
        print(f"BIC (ML): {result_interaction_ml.bic:.2f}")
        print(f"Log-Likelihood (ML): {result_interaction_ml.llf:.2f}")

        # Compare models using ML estimates
        print("\n" + "=" * 80)
        print("MODEL COMPARISON (using ML estimates)")
        print("=" * 80)

        aic_diff = result_main_ml.aic - result_interaction_ml.aic
        bic_diff = result_main_ml.bic - result_interaction_ml.bic

        print(f"AIC difference (Main - Moderation): {aic_diff:.2f}")
        print(f"BIC difference (Main - Moderation): {bic_diff:.2f}")

        if aic_diff > 2:
            print("  -> Moderation model shows better fit (AIC diff > 2)")
        elif aic_diff < -2:
            print("  -> Main effects model shows better fit (AIC diff < -2)")
        else:
            print("  -> Models show similar fit (|AIC diff| < 2)")

        # Likelihood ratio test using ML estimates
        lr_stat = -2 * (result_main_ml.llf - result_interaction_ml.llf)
        df_diff = len(result_interaction_ml.params) - len(result_main_ml.params)

        from scipy import stats

        # LR test is only valid if lr_stat is positive
        if lr_stat >= 0:
            p_value = stats.chi2.sf(lr_stat, df_diff)
        else:
            # If negative, the more complex model actually fits worse
            print(f"\nLR statistic is negative ({lr_stat:.2f})")
            print("This suggests the moderation model fits worse than main effects.")
            print("This can happen when interactions add noise rather than signal.")
            p_value = 1.0  # Not significant

        print(f"\nLikelihood Ratio Test:")
        print(f"  LR statistic: {lr_stat:.2f}")
        print(f"  df: {df_diff}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  *** Moderation model significantly better (p < 0.05)")
        else:
            print("  -> No significant improvement from moderation model")

        # Store ML results for proper model comparison
        self.results['main_effects_ml'] = result_main_ml
        self.results['moderation_ml'] = result_interaction_ml

        return result_main_reml, result_interaction_reml

    def interpret_results(self):
        """Print interpretation of moderation results."""
        if 'moderation' not in self.results:
            print("Run fit_moderation_models() first")
            return

        result = self.results['moderation']
        params = result.params
        pvalues = result.pvalues

        print("INTERPRETATION OF MODERATION RESULTS")
        print("=" * 80)

        print("\nMain Effect of Social Support:")
        coef_support = params['social_support_centered']
        p_support = pvalues['social_support_centered']
        print(f"  Coefficient: {coef_support:.4f}")
        print(f"  p-value: {p_support:.4f}")

        if p_support < 0.05:
            direction = "higher" if coef_support > 0 else "lower"
            print(f"  -> Social support is associated with {direction} cognitive function (p < 0.05)")
        else:
            print(f"  -> No significant main effect of social support")

        print("\nInteraction Effects (Moderation):")

        interactions = {
            'Early Peri': 'Early_Peri_x_Support',
            'Late Peri': 'Late_Peri_x_Support',
            'Post-menopause': 'Post_Menopause_x_Support',
            'Surgical': 'Surgical_x_Support'
        }

        for stage, param_name in interactions.items():
            coef = params.get(param_name, 0)
            p_val = pvalues.get(param_name, 1)

            print(f"\n  {stage}:")
            print(f"    Coefficient: {coef:.4f}")
            print(f"    p-value: {p_val:.4f}")

            if p_val < 0.05:
                if coef > 0:
                    print(f"    -> Social support has a STRONGER protective effect in {stage}")
                    print(f"       compared to pre-menopause (p < 0.05)")
                else:
                    print(f"    -> Social support has a WEAKER protective effect in {stage}")
                    print(f"       compared to pre-menopause (p < 0.05)")
            else:
                print(f"    -> No significant difference in social support effect")
                print(f"       between {stage} and pre-menopause")

    def run_analysis(self):
        """Run the complete moderation analysis pipeline."""
        self.preprocess_data()

        print("\nFitting moderation models...")
        main_result, mod_result = self.fit_moderation_models()

        print("\nCreating visualizations...")
        plot_moderation_effects(self)

        print("\nInterpreting results...")
        self.interpret_results()

        print("MODERATION ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: 3_social_model/output/")

        return self.results

if __name__ == "__main__":
    # Social support moderation analysis: testing whether social support moderates stage effects
    from scipy import stats
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_combined_data.csv")
    analysis = ModerationAnalysis(data_path)
    analysis.run_analysis()
