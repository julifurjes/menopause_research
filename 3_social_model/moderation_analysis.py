import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.save_output import OutputCapture, get_output_dir

class ModerationAnalysis:
    """
    Mixed-effects moderation analysis testing whether social support buffers
    the relationship between menopausal stage and cognitive function.

    This properly accounts for repeated measures using random intercepts for subjects
    while testing the interaction: Stage × Social Support → Cognition
    """

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.output_dir = get_output_dir('3_social_model', 'moderation')

        # Define variable groups
        self.social_support_vars = ['LISTEN', 'TAKETOM', 'HELPSIC', 'CONFIDE']
        self.cognitive_vars = ['TOTIDE1', 'TOTIDE2']
        self.control_vars = ['AGE_BASELINE', 'VISIT']

        self.results = {}

    def preprocess_data(self):
        """Prepare data for moderation analysis"""
        # Convert variables to numeric
        relevant_vars = (self.social_support_vars + self.cognitive_vars +
                        ['STATUS', 'SWANID', 'VISIT', 'AGE'])

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

        # Sort by subject and visit
        self.data = self.data.sort_values(['SWANID', 'VISIT'])

        # Create baseline age
        if 'AGE' in self.data.columns:
            self.data['AGE_BASELINE'] = self.data.groupby('SWANID')['AGE'].transform('first')

        # Create composite scores
        self.data['social_support'] = self.data[self.social_support_vars].mean(axis=1)
        self.data['cognitive_function'] = self.data[self.cognitive_vars].mean(axis=1)

        # Center social support for interaction interpretation
        self.data['social_support_centered'] = (self.data['social_support'] -
                                                 self.data['social_support'].mean())

        # Create dummy variables for STATUS with Pre-menopause as reference
        # Exclude Surgical for now to focus on natural transition
        self.data['Early_Peri'] = (self.data['STATUS_Label'] == 'Early Peri').astype(int)
        self.data['Late_Peri'] = (self.data['STATUS_Label'] == 'Late Peri').astype(int)
        self.data['Post_Menopause'] = (self.data['STATUS_Label'] == 'Post-menopause').astype(int)

        # Create interaction terms
        self.data['Early_Peri_x_Support'] = self.data['Early_Peri'] * self.data['social_support_centered']
        self.data['Late_Peri_x_Support'] = self.data['Late_Peri'] * self.data['social_support_centered']
        self.data['Post_Menopause_x_Support'] = self.data['Post_Menopause'] * self.data['social_support_centered']

        # Drop missing values
        required_vars = (['social_support', 'cognitive_function', 'AGE_BASELINE', 'VISIT'] +
                        ['Early_Peri', 'Late_Peri', 'Post_Menopause'])
        self.data = self.data.dropna(subset=required_vars)

        # Ensure SWANID is string for grouping
        self.data['SWANID'] = self.data['SWANID'].astype(str)

        print(f"\nData preprocessed: {len(self.data)} observations from {self.data['SWANID'].nunique()} subjects")
        print(f"\nObservations by stage:")
        print(self.data['STATUS_Label'].value_counts())

    def fit_moderation_models(self):
        """Fit mixed-effects models testing moderation"""

        print("\n" + "=" * 80)
        print("MODERATION ANALYSIS: Social Support × Menopausal Stage")
        print("=" * 80)

        # Model 1: Main effects only (no interaction)
        print("\nModel 1: Main Effects Only")
        print("-" * 80)

        formula_main = ("cognitive_function ~ social_support_centered + "
                       "Early_Peri + Late_Peri + Post_Menopause + "
                       "AGE_BASELINE + VISIT")

        model_main = mixedlm(formula=formula_main,
                            data=self.data,
                            groups=self.data["SWANID"],
                            re_formula="~1")  # Random intercept only

        result_main = model_main.fit(reml=True)
        self.results['main_effects'] = result_main

        print(result_main.summary())
        print(f"\nAIC: {result_main.aic:.2f}")
        print(f"BIC: {result_main.bic:.2f}")

        # Model 2: With interaction terms (moderation model)
        print("\n" + "=" * 80)
        print("Model 2: Moderation Model (with interactions)")
        print("-" * 80)

        formula_interaction = ("cognitive_function ~ social_support_centered + "
                              "Early_Peri + Late_Peri + Post_Menopause + "
                              "Early_Peri_x_Support + Late_Peri_x_Support + Post_Menopause_x_Support + "
                              "AGE_BASELINE + VISIT")

        model_interaction = mixedlm(formula=formula_interaction,
                                    data=self.data,
                                    groups=self.data["SWANID"],
                                    re_formula="~1")

        result_interaction = model_interaction.fit(reml=True)
        self.results['moderation'] = result_interaction

        print(result_interaction.summary())
        print(f"\nAIC: {result_interaction.aic:.2f}")
        print(f"BIC: {result_interaction.bic:.2f}")

        # Compare models
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        aic_diff = result_main.aic - result_interaction.aic
        bic_diff = result_main.bic - result_interaction.bic

        print(f"AIC difference (Main - Moderation): {aic_diff:.2f}")
        print(f"BIC difference (Main - Moderation): {bic_diff:.2f}")

        if aic_diff > 2:
            print("\nModeration model shows better fit (ΔAIC > 2)")
        else:
            print("\nNo substantial improvement from adding interactions")

        # Likelihood ratio test
        lr_stat = -2 * (result_main.llf - result_interaction.llf)
        df_diff = len(result_interaction.params) - len(result_main.params)

        from scipy import stats
        p_value = stats.chi2.sf(lr_stat, df_diff)

        print(f"\nLikelihood Ratio Test:")
        print(f"  LR statistic: {lr_stat:.2f}")
        print(f"  df: {df_diff}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  *** Moderation model significantly better (p < 0.05)")
        else:
            print("  No significant improvement from moderation model")

        return result_main, result_interaction

    def plot_moderation_effects(self):
        """Visualize the moderation effects"""

        if 'moderation' not in self.results:
            print("Run fit_moderation_models() first")
            return

        result = self.results['moderation']

        # Extract coefficients
        params = result.params

        # Create visualization of simple slopes at different stages
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Color palette
        green_palette = sns.color_palette("YlGn", n_colors=8)
        colors = {
            'Pre-menopause': green_palette[2],
            'Early Peri': green_palette[4],
            'Late Peri': green_palette[6],
            'Post-menopause': green_palette[7]
        }

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

            ax.plot(support_range, pred, label=stage, color=color, linewidth=3)

        ax.set_xlabel('Social Support (Centered)', fontsize=14)
        ax.set_ylabel('Predicted Cognitive Function', fontsize=14)
        ax.set_title('Moderation Effect: Social Support × Menopausal Stage', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 2: Interaction coefficients
        ax = axes[1]

        stages = ['Pre-menopause\n(reference)', 'Early Peri', 'Late Peri', 'Post-menopause']
        interaction_coefs = [
            0,  # Reference
            params.get('Early_Peri_x_Support', 0),
            params.get('Late_Peri_x_Support', 0),
            params.get('Post_Menopause_x_Support', 0)
        ]

        # Get standard errors for confidence intervals
        bse = result.bse
        interaction_ses = [
            0,
            bse.get('Early_Peri_x_Support', 0),
            bse.get('Late_Peri_x_Support', 0),
            bse.get('Post_Menopause_x_Support', 0)
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
        ax.set_yticklabels(stages, fontsize=12)
        ax.set_xlabel('Interaction Coefficient\n(Social Support × Stage Effect)', fontsize=14)
        ax.set_title('Strength of Moderation by Stage', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(self.output_dir, 'moderation_effects.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nModeration effects plot saved to: {output_path}")

    def interpret_results(self):
        """Print interpretation of moderation results"""

        if 'moderation' not in self.results:
            print("Run fit_moderation_models() first")
            return

        result = self.results['moderation']
        params = result.params
        pvalues = result.pvalues

        print("\n" + "=" * 80)
        print("INTERPRETATION OF MODERATION RESULTS")
        print("=" * 80)

        print("\nMain Effect of Social Support:")
        coef_support = params['social_support_centered']
        p_support = pvalues['social_support_centered']
        print(f"  Coefficient: {coef_support:.4f}")
        print(f"  p-value: {p_support:.4f}")

        if p_support < 0.05:
            direction = "higher" if coef_support > 0 else "lower"
            print(f"  → Social support is associated with {direction} cognitive function (p < 0.05)")
        else:
            print(f"  → No significant main effect of social support")

        print("\nInteraction Effects (Moderation):")

        interactions = {
            'Early Peri': 'Early_Peri_x_Support',
            'Late Peri': 'Late_Peri_x_Support',
            'Post-menopause': 'Post_Menopause_x_Support'
        }

        for stage, param_name in interactions.items():
            coef = params.get(param_name, 0)
            p_val = pvalues.get(param_name, 1)

            print(f"\n  {stage}:")
            print(f"    Coefficient: {coef:.4f}")
            print(f"    p-value: {p_val:.4f}")

            if p_val < 0.05:
                if coef > 0:
                    print(f"    → Social support has a STRONGER protective effect in {stage}")
                    print(f"       compared to pre-menopause (p < 0.05)")
                else:
                    print(f"    → Social support has a WEAKER protective effect in {stage}")
                    print(f"       compared to pre-menopause (p < 0.05)")
            else:
                print(f"    → No significant difference in social support effect")
                print(f"       between {stage} and pre-menopause")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\nThis moderation analysis properly accounts for repeated measures by using")
        print("mixed-effects models with random intercepts for each subject.")
        print("\nIt tests whether social support differentially predicts cognitive function")
        print("across menopausal stages, addressing the question: Does social support")
        print("buffer cognitive changes during the menopausal transition?")
        print("=" * 80)

    def run_analysis(self):
        """Run complete moderation analysis"""
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up output capture
        output_capture = OutputCapture(self.output_dir)
        sys.stdout = output_capture

        try:
            print("=" * 80)
            print("MODERATION ANALYSIS: Social Support × Menopausal Stage")
            print("=" * 80)
            print("\nThis analysis uses mixed-effects models to test whether social support")
            print("moderates the relationship between menopausal stage and cognitive function,")
            print("while properly accounting for repeated measures from the same individuals.")
            print("=" * 80)

            self.preprocess_data()

            print("\nFitting moderation models...")
            main_result, mod_result = self.fit_moderation_models()

            print("\nCreating visualizations...")
            self.plot_moderation_effects()

            print("\nInterpreting results...")
            self.interpret_results()

            print("\n" + "=" * 80)
            print("MODERATION ANALYSIS COMPLETE")
            print("=" * 80)
            print(f"\nResults saved to: {self.output_dir}")

            return self.results

        finally:
            sys.stdout = output_capture.terminal
            output_capture.close()

if __name__ == "__main__":
    from scipy import stats  # Import here for simple slopes calculation
    analysis = ModerationAnalysis("processed_combined_data.csv")
    analysis.run_analysis()
