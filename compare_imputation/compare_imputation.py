"""
Script to compare model results using data with imputation vs original data with missingness.

This script runs all three models twice:
1. With imputation (processed_data_with_imputation.csv - after KNN imputation)
2. Original data (processed_data_original.csv - before imputation, with missing values)

It then compares:
- Coefficient values and p-values
- Sample sizes
- Model fit statistics
- Changes in significance levels
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import importlib.util

# Add project root to path for utilities
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ImputationComparator:
    def __init__(self, with_imputation_file, original_file):
        self.with_imputation_file = with_imputation_file
        self.original_file = original_file
        self.results = {
            'stages': {'with_imputation': None, 'original': None},
            'symptoms': {'with_imputation': None, 'original': None},
            'social': {'with_imputation': None, 'original': None}
        }
        self.sample_sizes = {
            'stages': {'with_imputation': {}, 'original': {}},
            'symptoms': {'with_imputation': {}, 'original': {}},
            'social': {'with_imputation': {}, 'original': {}}
        }

    def run_stages_model(self, data_file, data_type):
        """Run stages model on specified data."""
        print(f"\n{'='*80}")
        print(f"Running STAGES MODEL with {data_type.upper()} DATA")
        print(f"{'='*80}\n")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, '1_stages_model')
        sys.path.insert(0, model_dir)

        try:
            module_path = os.path.join(model_dir, 'analysis.py')
            spec = importlib.util.spec_from_file_location(f"stages_{data_type}", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.MenopauseCognitionAnalysis(data_file, use_langcog=False)
            analyzer.filter_status()
            analyzer.transform_variables()

            # Store sample sizes before running models
            for outcome in analyzer.outcome_vars:
                self.sample_sizes['stages'][data_type][outcome] = {
                    'n_obs': len(analyzer.data.dropna(subset=[outcome])),
                    'n_subjects': analyzer.data.dropna(subset=[outcome])['SWANID'].nunique()
                }

            analyzer.run_mixed_models()
            return analyzer.mixed_model_results

        finally:
            sys.path.remove(model_dir)

    def run_symptoms_model(self, data_file, data_type):
        """Run symptoms model on specified data."""
        print(f"\n{'='*80}")
        print(f"Running SYMPTOMS MODEL with {data_type.upper()} DATA")
        print(f"{'='*80}\n")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, '2_symptoms_model')
        sys.path.insert(0, model_dir)

        try:
            module_path = os.path.join(model_dir, 'analysis.py')
            spec = importlib.util.spec_from_file_location(f"symptoms_{data_type}", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.MenopauseCognitionAnalysis(data_file, use_langcog=False)
            analyzer.prepare_data()

            # Store sample sizes
            for outcome in analyzer.transformed_outcome_vars:
                for symptom in analyzer.transformed_symptom_vars:
                    key = f"{outcome}_{symptom}"
                    if data_type not in self.sample_sizes['symptoms']:
                        self.sample_sizes['symptoms'][data_type] = {}
                    self.sample_sizes['symptoms'][data_type][key] = {
                        'n_obs': len(analyzer.data.dropna(subset=[outcome, symptom])),
                        'n_subjects': analyzer.data.dropna(subset=[outcome, symptom])['SWANID'].nunique()
                    }

            analyzer.run_mixed_models()
            return analyzer.mixed_model_results

        finally:
            sys.path.remove(model_dir)

    def run_social_model(self, data_file, data_type):
        """Run social model on specified data."""
        print(f"\n{'='*80}")
        print(f"Running SOCIAL MODEL with {data_type.upper()} DATA")
        print(f"{'='*80}\n")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, '3_social_model')
        sys.path.insert(0, model_dir)

        try:
            module_path = os.path.join(model_dir, 'analysis.py')
            spec = importlib.util.spec_from_file_location(f"social_{data_type}", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.ModerationAnalysis(data_file)
            analyzer.preprocess_data()
            analyzer.fit_moderation_models()

            # Store sample size (after preprocessing)
            self.sample_sizes['social'][data_type] = {
                'n_obs': len(analyzer.data),
                'n_subjects': analyzer.data['SWANID'].nunique() if 'SWANID' in analyzer.data.columns else 'N/A'
            }

            return analyzer.results

        finally:
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

    def compare_coefficients(self, with_imputation_stats, original_stats, predictor_name):
        """Compare coefficients between data with imputation and original data."""
        comparisons = []

        if with_imputation_stats is None or original_stats is None:
            return comparisons

        with_imputation_params = with_imputation_stats.get('coefficients', {})
        original_params = original_stats.get('coefficients', {})
        with_imputation_pvals = with_imputation_stats.get('pvalues', {})
        original_pvals = original_stats.get('pvalues', {})

        common_predictors = set(original_params.keys())

        for pred in common_predictors:
            coef_with_imputation = with_imputation_params.get(pred, np.nan)
            coef_original = original_params.get(pred, np.nan)
            pval_with_imputation = with_imputation_pvals.get(pred, np.nan)
            pval_original = original_pvals.get(pred, np.nan)

            # Check significance change
            sig_with_imputation = pval_with_imputation < 0.05 if not np.isnan(pval_with_imputation) else None
            sig_original = pval_original < 0.05 if not np.isnan(pval_original) else None

            sig_change = "No change"
            if sig_with_imputation is not None and sig_original is not None:
                if sig_with_imputation != sig_original:
                    if sig_with_imputation and not sig_original:
                        sig_change = "Gained significance (with imputation)"
                    else:
                        sig_change = "Lost significance (original data was sig)"

            # Calculate percent change in coefficient
            if not np.isnan(coef_original) and not np.isnan(coef_with_imputation) and coef_original != 0:
                pct_change = ((coef_with_imputation - coef_original) / abs(coef_original)) * 100
            else:
                pct_change = np.nan

            comparisons.append({
                'Model': predictor_name,
                'Predictor': pred,
                'Coef_With_Imputation': coef_with_imputation,
                'Coef_Original': coef_original,
                'Pct_Change': pct_change,
                'PValue_With_Imputation': pval_with_imputation,
                'PValue_Original': pval_original,
                'Sig_Change': sig_change
            })

        return comparisons

    def run_comparison(self):
        """Run all models and compare results."""
        print(f"\n{'='*80}")
        print("IMPUTATION COMPARISON ANALYSIS")
        print(f"Analysis Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        # Run stages model
        print("\n" + "="*80)
        print("MODEL 1: STAGES MODEL")
        print("="*80)
        self.results['stages']['with_imputation'] = self.run_stages_model(
            self.with_imputation_file, 'with_imputation'
        )
        self.results['stages']['original'] = self.run_stages_model(
            self.original_file, 'original'
        )

        # Run symptoms model
        print("\n" + "="*80)
        print("MODEL 2: SYMPTOMS MODEL")
        print("="*80)
        self.results['symptoms']['with_imputation'] = self.run_symptoms_model(
            self.with_imputation_file, 'with_imputation'
        )
        self.results['symptoms']['original'] = self.run_symptoms_model(
            self.original_file, 'original'
        )

        # Run social model
        print("\n" + "="*80)
        print("MODEL 3: SOCIAL MODEL")
        print("="*80)
        self.results['social']['with_imputation'] = self.run_social_model(
            self.with_imputation_file, 'with_imputation'
        )
        self.results['social']['original'] = self.run_social_model(
            self.original_file, 'original'
        )

    def generate_comparison_report(self, output_dir="output"):
        """Generate comprehensive comparison report."""
        os.makedirs(output_dir, exist_ok=True)

        # Collect all coefficient comparisons
        all_coef_comparisons = []

        # Stages model comparisons
        if (self.results['stages']['with_imputation'] is not None and
            self.results['stages']['original'] is not None):
            for outcome, model_result in self.results['stages']['with_imputation'].items():
                with_imputation_stats = self.extract_model_stats(model_result)
                original_stats = self.extract_model_stats(
                    self.results['stages']['original'][outcome]
                )
                comparisons = self.compare_coefficients(
                    with_imputation_stats, original_stats, f"Stages_{outcome}"
                )
                all_coef_comparisons.extend(comparisons)

        # Symptoms model comparisons
        if (self.results['symptoms']['with_imputation'] is not None and
            self.results['symptoms']['original'] is not None):
            for outcome, outcome_results in self.results['symptoms']['with_imputation'].items():
                if isinstance(outcome_results, dict):
                    for symptom, model_result in outcome_results.items():
                        with_imputation_stats = self.extract_model_stats(model_result)
                        original_stats = self.extract_model_stats(
                            self.results['symptoms']['original'][outcome][symptom]
                        )
                        comparisons = self.compare_coefficients(
                            with_imputation_stats, original_stats, f"Symptoms_{outcome}_{symptom}"
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

        # Save sample size comparison
        sample_size_file = os.path.join(output_dir, "sample_size_comparison.csv")
        self.save_sample_size_comparison(sample_size_file)

        # Generate summary report
        summary_file = os.path.join(output_dir, "comparison_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("IMPUTATION COMPARISON SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Sample size summary
            f.write("SAMPLE SIZE CHANGES\n")
            f.write("-"*80 + "\n")
            self.write_sample_size_summary(f)

            # Significance changes summary
            f.write("\n\nSIGNIFICANCE CHANGES\n")
            f.write("-"*80 + "\n")
            sig_changes = coef_df[coef_df['Sig_Change'].str.contains('sig', case=False, na=False)]

            if len(sig_changes) > 0:
                f.write(f"Total predictors with significance changes: {len(sig_changes)}\n\n")
                for _, row in sig_changes.iterrows():
                    f.write(f"Model: {row['Model']}\n")
                    f.write(f"  Predictor: {row['Predictor']}\n")
                    f.write(f"  Change: {row['Sig_Change']}\n")
                    f.write(f"  P-value (with imputation): {row['PValue_With_Imputation']:.4f}\n")
                    f.write(f"  P-value (original): {row['PValue_Original']:.4f}\n")
                    f.write(f"  Coefficient change: {row['Pct_Change']:.2f}%\n\n")
            else:
                f.write("No significance changes detected.\n\n")

            # Large coefficient changes (>20%)
            f.write("\nLARGE COEFFICIENT CHANGES (>20%)\n")
            f.write("-"*80 + "\n")
            large_changes = coef_df[abs(coef_df['Pct_Change']) > 20].sort_values(
                'Pct_Change', key=abs, ascending=False
            )

            if len(large_changes) > 0:
                f.write(f"Total predictors with >20% change: {len(large_changes)}\n\n")
                for _, row in large_changes.iterrows():
                    f.write(f"Model: {row['Model']}\n")
                    f.write(f"  Predictor: {row['Predictor']}\n")
                    f.write(f"  Coefficient (with imputation): {row['Coef_With_Imputation']:.4f}\n")
                    f.write(f"  Coefficient (original): {row['Coef_Original']:.4f}\n")
                    f.write(f"  Percent change: {row['Pct_Change']:.2f}%\n\n")
            else:
                f.write("No large coefficient changes detected.\n\n")

        print(f"Summary report saved to: {summary_file}")

        # Create key predictors comparison
        key_predictors = coef_df[coef_df['Predictor'].str.contains('STATUS|VISIT|AGE', na=False)]
        key_file = os.path.join(output_dir, "key_predictors_comparison.csv")
        key_predictors.to_csv(key_file, index=False)
        print(f"Key predictors comparison saved to: {key_file}")

        return coef_df

    def save_sample_size_comparison(self, output_file):
        """Save sample size comparison to CSV."""
        rows = []

        for model_name, model_data in self.sample_sizes.items():
            # Social model has different structure - no outcomes, just direct sample sizes
            if model_name == 'social':
                with_imputation_sizes = model_data.get('with_imputation', {})
                original_sizes = model_data.get('original', {})

                if with_imputation_sizes:  # Only add if we have data
                    rows.append({
                        'Model': model_name,
                        'Outcome': 'cognitive_function',  # Social model only has one outcome
                        'N_Obs_With_Imputation': with_imputation_sizes.get('n_obs', 'N/A'),
                        'N_Subjects_With_Imputation': with_imputation_sizes.get('n_subjects', 'N/A'),
                        'N_Obs_Original': original_sizes.get('n_obs', 'N/A'),
                        'N_Subjects_Original': original_sizes.get('n_subjects', 'N/A'),
                        'Obs_Gained': with_imputation_sizes.get('n_obs', 0) - original_sizes.get('n_obs', 0),
                        'Subjects_Gained': with_imputation_sizes.get('n_subjects', 0) - original_sizes.get('n_subjects', 0)
                    })
            else:
                # Stages and symptoms models have outcomes nested
                for outcome, sizes in model_data.get('with_imputation', {}).items():
                    original_sizes = model_data.get('original', {}).get(outcome, {})

                    rows.append({
                        'Model': model_name,
                        'Outcome': outcome,
                        'N_Obs_With_Imputation': sizes.get('n_obs', 'N/A'),
                        'N_Subjects_With_Imputation': sizes.get('n_subjects', 'N/A'),
                        'N_Obs_Original': original_sizes.get('n_obs', 'N/A'),
                        'N_Subjects_Original': original_sizes.get('n_subjects', 'N/A'),
                        'Obs_Gained': sizes.get('n_obs', 0) - original_sizes.get('n_obs', 0),
                        'Subjects_Gained': sizes.get('n_subjects', 0) - original_sizes.get('n_subjects', 0)
                    })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Sample size comparison saved to: {output_file}")

    def write_sample_size_summary(self, f):
        """Write sample size summary to file."""
        for model_name, model_data in self.sample_sizes.items():
            f.write(f"\n{model_name.upper()} MODEL:\n")

            with_imputation_data = model_data.get('with_imputation', {})
            original_data = model_data.get('original', {})

            if not with_imputation_data:
                f.write("  No data available\n")
                continue

            # Social model has different structure - no outcomes, just direct sample sizes
            if model_name == 'social':
                f.write(f"  cognitive_function:\n")
                f.write(f"    With imputation: {with_imputation_data.get('n_obs', 'N/A')} obs, "
                       f"{with_imputation_data.get('n_subjects', 'N/A')} subjects\n")
                f.write(f"    Original: {original_data.get('n_obs', 'N/A')} obs, "
                       f"{original_data.get('n_subjects', 'N/A')} subjects\n")

                if isinstance(with_imputation_data.get('n_obs'), (int, float)) and isinstance(original_data.get('n_obs'), (int, float)):
                    obs_gained = with_imputation_data['n_obs'] - original_data['n_obs']
                    pct_gained = (obs_gained / with_imputation_data['n_obs'] * 100) if with_imputation_data['n_obs'] > 0 else 0
                    f.write(f"    Gained: {obs_gained} obs ({pct_gained:.1f}%)\n")
            else:
                # Stages and symptoms models have outcomes nested
                for outcome in with_imputation_data.keys():
                    with_imp_sizes = with_imputation_data[outcome]
                    orig_sizes = original_data.get(outcome, {})

                    f.write(f"  {outcome}:\n")
                    f.write(f"    With imputation: {with_imp_sizes.get('n_obs', 'N/A')} obs, "
                           f"{with_imp_sizes.get('n_subjects', 'N/A')} subjects\n")
                    f.write(f"    Original: {orig_sizes.get('n_obs', 'N/A')} obs, "
                           f"{orig_sizes.get('n_subjects', 'N/A')} subjects\n")

                    if isinstance(with_imp_sizes.get('n_obs'), (int, float)) and isinstance(orig_sizes.get('n_obs'), (int, float)):
                        obs_gained = with_imp_sizes['n_obs'] - orig_sizes['n_obs']
                        pct_gained = (obs_gained / with_imp_sizes['n_obs'] * 100) if with_imp_sizes['n_obs'] > 0 else 0
                        f.write(f"    Gained: {obs_gained} obs ({pct_gained:.1f}%)\n")


def main():
    # Paths to data files (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with_imputation_file = os.path.join(project_root, "processed_data_with_imputation.csv")
    original_file = os.path.join(project_root, "processed_data_original.csv")

    if not os.path.exists(with_imputation_file):
        print(f"Error: Data file with imputation not found at {with_imputation_file}")
        return

    if not os.path.exists(original_file):
        print(f"Error: Original data file not found at {original_file}")
        return

    # Create comparator and run analysis
    comparator = ImputationComparator(with_imputation_file, original_file)
    comparator.run_comparison()

    # Generate comparison report
    results_df = comparator.generate_comparison_report()

    print("\n" + "="*80)
    print("COMPARISON ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults saved to: compare_imputation/output/")
    print("  - coefficient_comparison.csv: Full coefficient comparison")
    print("  - sample_size_comparison.csv: Sample size changes per model")
    print("  - comparison_summary.txt: Summary of key changes")
    print("  - key_predictors_comparison.csv: Comparison for main predictors\n")


if __name__ == "__main__":
    main()
