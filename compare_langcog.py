"""
Script to compare model results with and without LANGCOG variable.

This script runs all three models (stages, symptoms, social) twice:
1. With LANGCOG included (use_langcog=True)
2. Without LANGCOG (use_langcog=False)

It then compares:
- Coefficient values and p-values
- Model fit statistics (R², RMSE, AIC, BIC if available)
- Changes in significance levels
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import re
import importlib.util

# Add project root to path for utilities
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Note: We'll import these dynamically to avoid naming conflicts

class LANGCOGComparator:
    def __init__(self, data_file):
        self.data_file = data_file
        self.results = {
            'stages': {'with_langcog': None, 'without_langcog': None},
            'symptoms': {'with_langcog': None, 'without_langcog': None},
            'social': {'with_langcog': None, 'without_langcog': None}
        }
        self.comparison_summary = []

    def run_stages_model(self, use_langcog):
        """Run stages model with or without LANGCOG."""
        print(f"\n{'='*80}")
        print(f"Running STAGES MODEL with LANGCOG={'INCLUDED' if use_langcog else 'EXCLUDED'}")
        print(f"{'='*80}\n")

        # Add model directory to sys.path for imports
        model_dir = os.path.join(os.path.dirname(__file__), '1_stages_model')
        sys.path.insert(0, model_dir)

        try:
            # Import using importlib to avoid caching issues
            module_path = os.path.join(model_dir, 'longitudinal.py')
            spec = importlib.util.spec_from_file_location("stages_longitudinal", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
 
            analyzer = module.MenopauseCognitionAnalysis(self.data_file, use_langcog=use_langcog)
            analyzer.filter_status()  # Create AGE_BASELINE
            analyzer.transform_variables()
            analyzer.run_mixed_models()

            return analyzer.mixed_model_results
        finally:
            # Clean up sys.path
            sys.path.remove(model_dir)

    def run_symptoms_model(self, use_langcog):
        """Run symptoms model with or without LANGCOG."""
        print(f"\n{'='*80}")
        print(f"Running SYMPTOMS MODEL with LANGCOG={'INCLUDED' if use_langcog else 'EXCLUDED'}")
        print(f"{'='*80}\n")

        # Add model directory to sys.path for imports
        model_dir = os.path.join(os.path.dirname(__file__), '2_symptoms_model')
        sys.path.insert(0, model_dir)

        try:
            # Import using importlib to avoid caching issues
            module_path = os.path.join(model_dir, 'longitudinal.py')
            spec = importlib.util.spec_from_file_location("symptoms_longitudinal", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.MenopauseCognitionAnalysis(self.data_file, use_langcog=use_langcog)
            analyzer.prepare_data()  # This calls transform_variables() internally and creates AGE_BASELINE
            analyzer.run_mixed_models()

            return analyzer.mixed_model_results
        finally:
            # Clean up sys.path
            sys.path.remove(model_dir)

    def run_social_model(self, use_langcog):
        """Run social model (note: this model doesn't use LANGCOG)."""
        print(f"\n{'='*80}")
        print(f"Running SOCIAL MODEL (Note: This model doesn't use LANGCOG)")
        print(f"{'='*80}\n")

        # Add model directory to sys.path for imports
        model_dir = os.path.join(os.path.dirname(__file__), '3_social_model')
        sys.path.insert(0, model_dir)

        try:
            # Import using importlib to avoid caching issues
            module_path = os.path.join(model_dir, 'longitudinal.py')
            spec = importlib.util.spec_from_file_location("social_longitudinal", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.MenopauseCognitionMixedModels(self.data_file)
            results = analyzer.run_analysis()

            return results
        finally:
            # Clean up sys.path
            sys.path.remove(model_dir)

    def extract_model_stats(self, model_result):
        """Extract key statistics from model results."""
        if model_result is None:
            return None

        stats = {}

        # Extract coefficients and p-values
        if hasattr(model_result, 'params'):
            stats['coefficients'] = model_result.params.to_dict()
        if hasattr(model_result, 'pvalues'):
            stats['pvalues'] = model_result.pvalues.to_dict()
        if hasattr(model_result, 'conf_int'):
            conf_int = model_result.conf_int()
            stats['conf_int_lower'] = conf_int[0].to_dict()
            stats['conf_int_upper'] = conf_int[1].to_dict()

        # Extract model fit statistics
        if hasattr(model_result, 'aic'):
            stats['aic'] = model_result.aic
        if hasattr(model_result, 'bic'):
            stats['bic'] = model_result.bic
        if hasattr(model_result, 'llf'):
            stats['log_likelihood'] = model_result.llf

        return stats

    def compare_coefficients(self, with_langcog, without_langcog, predictor_name):
        """Compare coefficients between models with and without LANGCOG."""
        comparisons = []

        if with_langcog is None or without_langcog is None:
            return comparisons

        # Get common predictors (excluding LANGCOG-specific ones)
        with_params = with_langcog.get('coefficients', {})
        without_params = without_langcog.get('coefficients', {})
        with_pvals = with_langcog.get('pvalues', {})
        without_pvals = without_langcog.get('pvalues', {})

        common_predictors = set(without_params.keys())

        for pred in common_predictors:
            if pred.startswith('C(LANGCOG'):
                continue

            coef_with = with_params.get(pred, np.nan)
            coef_without = without_params.get(pred, np.nan)
            pval_with = with_pvals.get(pred, np.nan)
            pval_without = without_pvals.get(pred, np.nan)

            # Check significance change
            sig_with = pval_with < 0.05 if not np.isnan(pval_with) else None
            sig_without = pval_without < 0.05 if not np.isnan(pval_without) else None

            sig_change = "No change"
            if sig_with is not None and sig_without is not None:
                if sig_with != sig_without:
                    if sig_with and not sig_without:
                        sig_change = "Lost significance"
                    else:
                        sig_change = "Gained significance"

            # Calculate percent change in coefficient
            if not np.isnan(coef_with) and not np.isnan(coef_without) and coef_with != 0:
                pct_change = ((coef_without - coef_with) / abs(coef_with)) * 100
            else:
                pct_change = np.nan

            comparisons.append({
                'Model': predictor_name,
                'Predictor': pred,
                'Coef_With_LANGCOG': coef_with,
                'Coef_Without_LANGCOG': coef_without,
                'Pct_Change': pct_change,
                'PValue_With_LANGCOG': pval_with,
                'PValue_Without_LANGCOG': pval_without,
                'Sig_Change': sig_change
            })

        return comparisons

    def compare_model_fit(self, with_langcog, without_langcog, model_name):
        """Compare model fit statistics."""
        comparison = {'Model': model_name}

        if with_langcog is None or without_langcog is None:
            return comparison

        # AIC comparison (lower is better)
        aic_with = with_langcog.get('aic', np.nan)
        aic_without = without_langcog.get('aic', np.nan)
        comparison['AIC_With_LANGCOG'] = aic_with
        comparison['AIC_Without_LANGCOG'] = aic_without
        comparison['AIC_Difference'] = aic_without - aic_with

        # BIC comparison (lower is better)
        bic_with = with_langcog.get('bic', np.nan)
        bic_without = without_langcog.get('bic', np.nan)
        comparison['BIC_With_LANGCOG'] = bic_with
        comparison['BIC_Without_LANGCOG'] = bic_without
        comparison['BIC_Difference'] = bic_without - bic_with

        # Log likelihood comparison (higher is better)
        ll_with = with_langcog.get('log_likelihood', np.nan)
        ll_without = without_langcog.get('log_likelihood', np.nan)
        comparison['LogLik_With_LANGCOG'] = ll_with
        comparison['LogLik_Without_LANGCOG'] = ll_without
        comparison['LogLik_Difference'] = ll_without - ll_with

        return comparison

    def run_comparison(self):
        """Run all models and compare results."""
        print(f"\n{'='*80}")
        print("LANGCOG COMPARISON ANALYSIS")
        print(f"Analysis Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        # Run stages model
        print("\n" + "="*80)
        print("MODEL 1: STAGES MODEL")
        print("="*80)
        self.results['stages']['with_langcog'] = self.run_stages_model(use_langcog=True)
        self.results['stages']['without_langcog'] = self.run_stages_model(use_langcog=False)

        # Run symptoms model
        print("\n" + "="*80)
        print("MODEL 2: SYMPTOMS MODEL")
        print("="*80)
        self.results['symptoms']['with_langcog'] = self.run_symptoms_model(use_langcog=True)
        self.results['symptoms']['without_langcog'] = self.run_symptoms_model(use_langcog=False)

        # Note: Social model doesn't use LANGCOG, so we skip it
        print("\n" + "="*80)
        print("MODEL 3: SOCIAL MODEL (Does not use LANGCOG - skipped)")
        print("="*80)

    def generate_comparison_report(self, output_dir="langcog_comparison_results"):
        """Generate comprehensive comparison report."""
        os.makedirs(output_dir, exist_ok=True)

        # Collect all coefficient comparisons
        all_coef_comparisons = []

        # Stages model comparisons
        if self.results['stages']['with_langcog'] is not None and self.results['stages']['without_langcog'] is not None:
            for outcome, model_result in self.results['stages']['with_langcog'].items():
                with_stats = self.extract_model_stats(model_result)
                without_stats = self.extract_model_stats(
                    self.results['stages']['without_langcog'][outcome]
                )
                comparisons = self.compare_coefficients(
                    with_stats, without_stats, f"Stages_{outcome}"
                )
                all_coef_comparisons.extend(comparisons)

        # Symptoms model comparisons
        if self.results['symptoms']['with_langcog'] is not None and self.results['symptoms']['without_langcog'] is not None:
            for outcome, outcome_results in self.results['symptoms']['with_langcog'].items():
                if isinstance(outcome_results, dict):
                    for symptom, model_result in outcome_results.items():
                        with_stats = self.extract_model_stats(model_result)
                        without_stats = self.extract_model_stats(
                            self.results['symptoms']['without_langcog'][outcome][symptom]
                        )
                        comparisons = self.compare_coefficients(
                            with_stats, without_stats, f"Symptoms_{outcome}_{symptom}"
                        )
                        all_coef_comparisons.extend(comparisons)

        # Create DataFrame and save
        coef_df = pd.DataFrame(all_coef_comparisons)

        if len(coef_df) == 0:
            print("\nWarning: No comparisons were generated. Check if models ran successfully.")
            return None

        coef_df = coef_df.round(4)

        # Save coefficient comparison
        coef_file = os.path.join(output_dir, "coefficient_comparison.csv")
        coef_df.to_csv(coef_file, index=False)
        print(f"\nCoefficient comparison saved to: {coef_file}")

        # Generate summary report
        summary_file = os.path.join(output_dir, "comparison_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LANGCOG COMPARISON SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Significance changes summary
            f.write("SIGNIFICANCE CHANGES\n")
            f.write("-"*80 + "\n")
            sig_changes = coef_df[coef_df['Sig_Change'] != 'No change'] if 'Sig_Change' in coef_df.columns else pd.DataFrame()
            if len(sig_changes) > 0:
                f.write(f"Total predictors with significance changes: {len(sig_changes)}\n\n")
                for _, row in sig_changes.iterrows():
                    f.write(f"Model: {row['Model']}\n")
                    f.write(f"  Predictor: {row['Predictor']}\n")
                    f.write(f"  Change: {row['Sig_Change']}\n")
                    f.write(f"  P-value with LANGCOG: {row['PValue_With_LANGCOG']:.4f}\n")
                    f.write(f"  P-value without LANGCOG: {row['PValue_Without_LANGCOG']:.4f}\n")
                    f.write(f"  Coefficient change: {row['Pct_Change']:.2f}%\n\n")
            else:
                f.write("No significance changes detected.\n\n")

            # Large coefficient changes (>10%)
            f.write("\nLARGE COEFFICIENT CHANGES (>10%)\n")
            f.write("-"*80 + "\n")
            large_changes = coef_df[abs(coef_df['Pct_Change']) > 10].sort_values(
                'Pct_Change', key=abs, ascending=False
            )
            if len(large_changes) > 0:
                f.write(f"Total predictors with >10% change: {len(large_changes)}\n\n")
                for _, row in large_changes.iterrows():
                    f.write(f"Model: {row['Model']}\n")
                    f.write(f"  Predictor: {row['Predictor']}\n")
                    f.write(f"  Coefficient with LANGCOG: {row['Coef_With_LANGCOG']:.4f}\n")
                    f.write(f"  Coefficient without LANGCOG: {row['Coef_Without_LANGCOG']:.4f}\n")
                    f.write(f"  Percent change: {row['Pct_Change']:.2f}%\n\n")
            else:
                f.write("No large coefficient changes detected.\n\n")

        print(f"Summary report saved to: {summary_file}")

        # Create simplified comparison for key predictors
        key_predictors = coef_df[coef_df['Predictor'].str.contains('STATUS|VISIT|AGE', na=False)]
        key_file = os.path.join(output_dir, "key_predictors_comparison.csv")
        key_predictors.to_csv(key_file, index=False)
        print(f"Key predictors comparison saved to: {key_file}")

        return coef_df


def main():
    # Path to processed data
    data_file = "processed_data.csv"

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # Create comparator and run analysis
    comparator = LANGCOGComparator(data_file)
    comparator.run_comparison()

    # Generate comparison report
    results_df = comparator.generate_comparison_report()

    print("\n" + "="*80)
    print("COMPARISON ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults saved to: langcog_comparison_results/")
    print("  - coefficient_comparison.csv: Full coefficient comparison")
    print("  - comparison_summary.txt: Summary of key changes")
    print("  - key_predictors_comparison.csv: Comparison for main predictors\n")


if __name__ == "__main__":
    main()
