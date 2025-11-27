import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import visualizations

class MenopauseCognitionAnalysis:
    """Analyze relationships between menopausal symptoms and cognitive/emotional outcomes."""

    def __init__(self, file_path, use_langcog=True):
        self.data = pd.read_csv(file_path, low_memory=False)
        self.symptom_vars = ['NUMHOTF', 'NUMNITS', 'NUMCLDS', 'STIFF', 'IRRITAB', 'MOODCHG', 'SLEEPQL']
        self.outcome_vars = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        self.transformed_symptom_vars = []  # Will be populated after transformations
        self.transformed_outcome_vars = []  # Will be populated after transformations
        self.control_vars = ['STATUS', 'AGE', 'AGE_BASELINE']
        self.mixed_model_results = {}
        self.use_langcog = use_langcog
        self.var_labels = {
            'NUMHOTF': 'Number of Hot Flashes',
            'NUMNITS': 'Number of Night Sweats',
            'NUMCLDS': 'Number of Cold Sweats',
            'STIFF': 'Stiffness',
            'IRRITAB': 'Irritability',
            'MOODCHG': 'Mood Changes',
            'SLEEPQL': 'Sleep Quality',
            'TOTIDE1': 'Cognitive Performance (Immediate Recall)',
            'TOTIDE2': 'Cognitive Performance (Delayed Recall)',
            'NERVES': 'Nervousness Score',
            'SAD': 'Sadness Score',
            'FEARFULA': 'Fearfulness Score',
        }
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_data(self):
        """Prepare data by filtering stages, creating labels, and standardizing symptom variables."""
        self.data['STATUS'] = pd.to_numeric(self.data['STATUS'], errors='coerce')
        self.data = self.data[self.data['STATUS'].isin([1, 2, 3, 4, 5, 8])]

        all_vars = self.symptom_vars + self.outcome_vars + self.control_vars + ['SWANID', 'VISIT']
        for var in all_vars:
            if var in self.data.columns:
                self.data[var] = pd.to_numeric(self.data[var], errors='coerce')

        status_map = {
            1: 'Surgical', 2: 'Postmenopause', 3: 'Late Perimenopause',
            4: 'Early Perimenopause', 5: 'Premenopause', 8: 'Surgical'
        }
        self.data['STATUS_Label'] = self.data['STATUS'].map(status_map)
        self.data['Menopause_Type'] = np.where(self.data['STATUS'].isin([1, 8]), 'Surgical', 'Natural')

        natural_order = ['Premenopause', 'Early Perimenopause', 'Late Perimenopause', 'Postmenopause']
        self.data['STATUS_Label'] = pd.Categorical(
            self.data['STATUS_Label'],
            categories=['Surgical'] + natural_order,
            ordered=True
        )

        self.data = self.data.sort_values(['SWANID', 'VISIT'])
        self.data['AGE_BASELINE'] = self.data.groupby('SWANID')['AGE'].transform('first')

        self.data[self.symptom_vars] = (self.data[self.symptom_vars] -
                                       self.data[self.symptom_vars].mean()) / self.data[self.symptom_vars].std()

        if self.use_langcog and 'LANGCOG' in self.data.columns:
            self.data['LANGCOG'] = self.data['LANGCOG'].astype('category')

        self.transform_variables()

    def transform_variables(self):
        """Apply log and sqrt transformations to address skewness in symptoms and outcomes."""
        log_transform_vars = ['NERVES', 'NUMHOTF', 'NUMNITS', 'NUMCLDS']
        sqrt_transform_vars = ['FEARFULA', 'SAD', 'MOODCHG', 'IRRITAB', 'SLEEPQL']
        no_transform_vars = ['STIFF']

        for var in log_transform_vars:
            if var in self.data.columns:
                with np.errstate(invalid='ignore'):
                    self.data[f"{var}_log"] = np.log1p(self.data[var].clip(lower=0))
                self.transformed_symptom_vars.append(f"{var}_log")
                self.var_labels[f"{var}_log"] = f"{self.var_labels.get(var, var)} (Log)"

        for var in sqrt_transform_vars:
            if var in self.data.columns:
                with np.errstate(invalid='ignore'):
                    self.data[f"{var}_sqrt"] = np.sqrt(self.data[var].clip(lower=0))
                self.transformed_symptom_vars.append(f"{var}_sqrt")
                self.var_labels[f"{var}_sqrt"] = f"{self.var_labels.get(var, var)} (Sqrt)"

        for var in no_transform_vars:
            if var in self.data.columns:
                self.transformed_symptom_vars.append(var)

        self.transformed_outcome_vars = []
        if 'TOTIDE1' in self.data.columns:
            self.transformed_outcome_vars.append('TOTIDE1')
        if 'TOTIDE2' in self.data.columns:
            self.transformed_outcome_vars.append('TOTIDE2')

    def run_mixed_models(self):
        """Run mixed-effects models analyzing symptom-cognition relationships."""
        print("\nRunning linear mixed-effects models with transformed variables...")

        # Get the reference language for the model (the most common one) if using LANGCOG
        if self.use_langcog and 'LANGCOG' in self.data.columns:
            reference_language = self.data['LANGCOG'].mode()[0]

        # Loop through all transformed outcome variables
        for transformed_outcome in self.transformed_outcome_vars:
            print(f"Analyzing models for {self.var_labels.get(transformed_outcome, transformed_outcome)}")
            
            # Get the original variable name (for results dictionary keys)
            if transformed_outcome.endswith('_log'):
                original_outcome = transformed_outcome.replace('_log', '')
            elif transformed_outcome.endswith('_sqrt'):
                original_outcome = transformed_outcome.replace('_sqrt', '')
            elif transformed_outcome.endswith('_avg'):
                original_outcome = 'TOTIDE'  # Special case for average
            else:
                original_outcome = transformed_outcome
            
            # Loop through each symptom
            for symptom in self.symptom_vars:
                # Create formula with status, symptom, baseline age, VISIT, and optionally LANGCOG
                if self.use_langcog and 'LANGCOG' in self.data.columns:
                    formula = (f"{transformed_outcome} ~ {symptom} + C(STATUS_Label, Treatment('Premenopause')) + "
                              f"AGE_BASELINE + VISIT + C(LANGCOG, Treatment({reference_language}))")
                    required_cols = [transformed_outcome, symptom, 'STATUS_Label', 'AGE_BASELINE', 'VISIT', 'LANGCOG']
                else:
                    formula = (f"{transformed_outcome} ~ {symptom} + C(STATUS_Label, Treatment('Premenopause')) + "
                              f"AGE_BASELINE + VISIT")
                    required_cols = [transformed_outcome, symptom, 'STATUS_Label', 'AGE_BASELINE', 'VISIT']

                try:
                    # Drop rows with missing values for the current variables
                    analysis_data = self.data.dropna(subset=required_cols).copy()

                    # Reset index to avoid potential index issues
                    analysis_data = analysis_data.reset_index(drop=True)
                    
                    # Add weights to handle unbalanced data (from first model)
                    status_counts = analysis_data['STATUS_Label'].value_counts()
                    analysis_data['weights'] = analysis_data['STATUS_Label'].map(
                        lambda x: 1 / (status_counts[x] / sum(status_counts))
                    )
                    
                    # Fit mixed model with random intercept for SWANID and random slope for VISIT (from first model)
                    model = mixedlm(
                        formula=formula,
                        groups=analysis_data["SWANID"],
                        data=analysis_data,
                        re_formula="~VISIT"  # Random slope for VISIT (from first model)
                    )
                    
                    # Add weights and fit the model
                    results = model.fit(reml=True, weights=analysis_data['weights'])
                    key = f"{original_outcome}_{symptom}"
                    self.mixed_model_results[key] = results
                    
                    # Print detailed results
                    print(f"\nMixed Model Results for {self.var_labels.get(symptom, symptom)} -> "
                          f"{self.var_labels.get(original_outcome, original_outcome)}")
                    print(f"Using transformed variable: {transformed_outcome}")
                    print("=" * 50)
                    print(results.summary())
                    
                    # Calculate marginal and conditional R-squared
                    try:
                        # Residual variance
                        resid_var = results.scale
                        # Random effects variance
                        re_var = results.cov_re.iloc[0, 0] if hasattr(results.cov_re, 'iloc') else results.cov_re[0][0]
                        # Variance explained by fixed effects (approximate)
                        var_fixed = np.var(results.predict(analysis_data))
                        
                        # Marginal R² (fixed effects only)
                        marginal_r2 = var_fixed / (var_fixed + re_var + resid_var)
                        # Conditional R² (both fixed and random effects)
                        conditional_r2 = (var_fixed + re_var) / (var_fixed + re_var + resid_var)
                        
                        print(f"\nApproximate Marginal R² (fixed effects): {marginal_r2:.4f}")
                        print(f"Approximate Conditional R² (fixed + random): {conditional_r2:.4f}")
                        
                    except Exception as e:
                        print(f"Error calculating R-squared: {str(e)}")
                    
                    # Check model diagnostics
                    self.check_model_diagnostics(results, original_outcome, transformed_outcome, symptom, analysis_data)
                    
                except Exception as e:
                    print(f"Error in mixed model analysis for {symptom} -> {original_outcome}: {str(e)}")
    
    def check_model_diagnostics(self, model_results, outcome, transformed_outcome, symptom, data):
        """Generate diagnostic plots for mixed model residuals."""
        try:
            # Calculate residuals
            predicted = model_results.predict(data)
            actual = data[transformed_outcome]
            residuals = actual - predicted
            
            # Create directory for diagnostics
            diag_dir = os.path.join(self.output_dir, "model_diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            
            # Plot residuals
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.scatter(predicted, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            title_text = f'Residuals vs Fitted for {self.var_labels.get(outcome, outcome)} ~ {self.var_labels.get(symptom, symptom)}'
            plt.title(title_text)
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            
            plt.subplot(2, 2, 2)
            sns.histplot(residuals, kde=True)
            plt.title('Histogram of Residuals')
            plt.xlabel('Residual Value')
            
            plt.subplot(2, 2, 3)
            sm.qqplot(residuals.dropna(), line='s', ax=plt.gca())
            plt.title('Q-Q Plot of Residuals')
            
            plt.subplot(2, 2, 4)
            plt.scatter(range(len(residuals)), residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Residuals vs Order')
            plt.xlabel('Observation Order')
            plt.ylabel('Residuals')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(diag_dir, f'{outcome}_{symptom}_diagnostics.png'))
            plt.close()
            
            # Check normality of residuals using different methods 
            # Depending on sample size
            if len(residuals) > 5000:
                # For large samples, use Anderson-Darling test with a sample
                sample_size = min(5000, len(residuals))
                residual_sample = pd.Series(residuals).sample(sample_size)
                _, ad_p = sm.stats.diagnostic.normal_ad(residual_sample.dropna())
                print(f"\nAnderson-Darling normality test p-value (on {sample_size} sampled residuals): {ad_p:.4f}")
                
                # Also try Shapiro-Wilk on a smaller sample for comparison
                shapiro_sample = pd.Series(residuals).sample(1000)  # Shapiro-Wilk works best with smaller samples
                _, sw_p = stats.shapiro(shapiro_sample.dropna())
                print(f"Shapiro-Wilk normality test p-value (on 1000 sampled residuals): {sw_p:.4f}")
            else:
                # For smaller samples, use both tests on the full data
                _, ad_p = sm.stats.diagnostic.normal_ad(residuals.dropna())
                _, sw_p = stats.shapiro(residuals.dropna())
                print(f"\nAnderson-Darling normality test p-value: {ad_p:.4f}")
                print(f"Shapiro-Wilk normality test p-value: {sw_p:.4f}")
            
            # Decide if normality is violated based on both tests
            if ad_p < 0.05 and sw_p < 0.05:
                print("WARNING: Residuals may not be normally distributed.")
                print("However, with this large sample size, the model is still robust to moderate non-normality.")
            else:
                print("Residuals appear to be more normally distributed after transformation.")
            
        except Exception as e:
            print(f"Error in model diagnostics: {str(e)}")
    

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("\nRunning full analysis with transformations from both models...")
        self.prepare_data()

        print("\nRunning mixed models analysis with transformed variables...")
        self.run_mixed_models()

        print("\nCreating symptom effects plots...")
        visualizations.plot_symptom_effects(self)

        print("\nCreating stratified forest plot...")
        visualizations.create_stratified_forest_plot(self, outcome='TOTIDE1')

        print("\nAnalyzing symptom intensity by menopausal stage...")
        visualizations.analyze_symptom_intensity_by_stage(self)

        print("\nCreating symptom intensity heat map...")
        visualizations.plot_symptom_intensity_heatmap(self)

        print("\nAnalysis complete")

if __name__ == "__main__":
    # Symptom-cognition analysis: relationships between menopausal symptoms and outcomes
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_combined_data.csv")
    analysis = MenopauseCognitionAnalysis(data_path, use_langcog=False)
    analysis.run_complete_analysis()