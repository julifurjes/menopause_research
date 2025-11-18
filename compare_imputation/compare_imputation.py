"""
Script to compare model results using imputed vs non-imputed (complete cases) data.

This script runs all three models twice:
1. With imputed data (current processed_data.csv)
2. With complete cases only (processed_combined_data.csv - before imputation)

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
    def __init__(self, imputed_file, complete_cases_file):
        self.imputed_file = imputed_file
        self.complete_cases_file = complete_cases_file
        self.results = {
            'stages': {'imputed': None, 'complete_cases': None},
            'symptoms': {'imputed': None, 'complete_cases': None},
            'social': {'imputed': None, 'complete_cases': None}
        }
        self.sample_sizes = {
            'stages': {'imputed': {}, 'complete_cases': {}},
            'symptoms': {'imputed': {}, 'complete_cases': {}},
            'social': {'imputed': {}, 'complete_cases': {}}
        }

    def run_stages_model(self, data_file, data_type):
        """Run stages model on specified data."""
        print(f"\n{'='*80}")
        print(f"Running STAGES MODEL with {data_type.upper()} DATA")
        print(f"{'='*80}\n")

        model_dir = os.path.join(os.path.dirname(__file__), '1_stages_model')
        sys.path.insert(0, model_dir)

        try:
            module_path = os.path.join(model_dir, 'longitudinal.py')
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

        model_dir = os.path.join(os.path.dirname(__file__), '2_symptoms_model')
        sys.path.insert(0, model_dir)

        try:
            module_path = os.path.join(model_dir, 'longitudinal.py')
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

        model_dir = os.path.join(os.path.dirname(__file__), '3_social_model')
        sys.path.insert(0, model_dir)

        try:
            module_path = os.path.join(model_dir, 'longitudinal.py')
            spec = importlib.util.spec_from_file_location(f"social_{data_type}", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.MenopauseCognitionMixedModels(data_file)
            results = analyzer.run_complete_analysis()

            # Store sample size (after preprocessing)
            self.sample_sizes['social'][data_type] = {
                'n_obs': len(analyzer.data),
                'n_subjects': analyzer.data['SWANID'].nunique() if 'SWANID' in analyzer.data.columns else 'N/A'
            }

            return results

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

    def compare_coefficients(self, imputed_stats, complete_stats, predictor_name):
        """Compare coefficients between imputed and complete cases."""
        comparisons = []

        if imputed_stats is None or complete_stats is None:
            return comparisons

        imputed_params = imputed_stats.get('coefficients', {})
        complete_params = complete_stats.get('coefficients', {})
        imputed_pvals = imputed_stats.get('pvalues', {})
        complete_pvals = complete_stats.get('pvalues', {})

        common_predictors = set(complete_params.keys())

        for pred in common_predictors:
            coef_imputed = imputed_params.get(pred, np.nan)
            coef_complete = complete_params.get(pred, np.nan)
            pval_imputed = imputed_pvals.get(pred, np.nan)
            pval_complete = complete_pvals.get(pred, np.nan)

            # Check significance change
            sig_imputed = pval_imputed < 0.05 if not np.isnan(pval_imputed) else None
            sig_complete = pval_complete < 0.05 if not np.isnan(pval_complete) else None

            sig_change = "No change"
            if sig_imputed is not None and sig_complete is not None:
                if sig_imputed != sig_complete:
                    if sig_imputed and not sig_complete:
                        sig_change = "Lost significance (imputed was sig)"
                    else:
                        sig_change = "Gained significance (complete cases sig)"

            # Calculate percent change in coefficient
            if not np.isnan(coef_imputed) and not np.isnan(coef_complete) and coef_imputed != 0:
                pct_change = ((coef_complete - coef_imputed) / abs(coef_imputed)) * 100
            else:
                pct_change = np.nan

            comparisons.append({
                'Model': predictor_name,
                'Predictor': pred,
                'Coef_Imputed': coef_imputed,
                'Coef_Complete_Cases': coef_complete,
                'Pct_Change': pct_change,
                'PValue_Imputed': pval_imputed,
                'PValue_Complete_Cases': pval_complete,
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
        self.results['stages']['imputed'] = self.run_stages_model(
            self.imputed_file, 'imputed'
        )
        self.results['stages']['complete_cases'] = self.run_stages_model(
            self.complete_cases_file, 'complete_cases'
        )

        # Run symptoms model
        print("\n" + "="*80)
        print("MODEL 2: SYMPTOMS MODEL")
        print("="*80)
        self.results['symptoms']['imputed'] = self.run_symptoms_model(
            self.imputed_file, 'imputed'
        )
        self.results['symptoms']['complete_cases'] = self.run_symptoms_model(
            self.complete_cases_file, 'complete_cases'
        )

        # Run social model
        print("\n" + "="*80)
        print("MODEL 3: SOCIAL MODEL")
        print("="*80)
        self.results['social']['imputed'] = self.run_social_model(
            self.imputed_file, 'imputed'
        )
        self.results['social']['complete_cases'] = self.run_social_model(
            self.complete_cases_file, 'complete_cases'
        )

    def generate_comparison_report(self, output_dir="output"):
        """Generate comprehensive comparison report."""
        os.makedirs(output_dir, exist_ok=True)

        # Collect all coefficient comparisons
        all_coef_comparisons = []

        # Stages model comparisons
        if (self.results['stages']['imputed'] is not None and
            self.results['stages']['complete_cases'] is not None):
            for outcome, model_result in self.results['stages']['imputed'].items():
                imputed_stats = self.extract_model_stats(model_result)
                complete_stats = self.extract_model_stats(
                    self.results['stages']['complete_cases'][outcome]
                )
                comparisons = self.compare_coefficients(
                    imputed_stats, complete_stats, f"Stages_{outcome}"
                )
                all_coef_comparisons.extend(comparisons)

        # Symptoms model comparisons
        if (self.results['symptoms']['imputed'] is not None and
            self.results['symptoms']['complete_cases'] is not None):
            for outcome, outcome_results in self.results['symptoms']['imputed'].items():
                if isinstance(outcome_results, dict):
                    for symptom, model_result in outcome_results.items():
                        imputed_stats = self.extract_model_stats(model_result)
                        complete_stats = self.extract_model_stats(
                            self.results['symptoms']['complete_cases'][outcome][symptom]
                        )
                        comparisons = self.compare_coefficients(
                            imputed_stats, complete_stats, f"Symptoms_{outcome}_{symptom}"
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
                    f.write(f"  P-value (imputed): {row['PValue_Imputed']:.4f}\n")
                    f.write(f"  P-value (complete cases): {row['PValue_Complete_Cases']:.4f}\n")
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
                    f.write(f"  Coefficient (imputed): {row['Coef_Imputed']:.4f}\n")
                    f.write(f"  Coefficient (complete): {row['Coef_Complete_Cases']:.4f}\n")
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
                imputed_sizes = model_data.get('imputed', {})
                complete_sizes = model_data.get('complete_cases', {})

                if imputed_sizes:  # Only add if we have data
                    rows.append({
                        'Model': model_name,
                        'Outcome': 'cognitive_function',  # Social model only has one outcome
                        'N_Obs_Imputed': imputed_sizes.get('n_obs', 'N/A'),
                        'N_Subjects_Imputed': imputed_sizes.get('n_subjects', 'N/A'),
                        'N_Obs_Complete': complete_sizes.get('n_obs', 'N/A'),
                        'N_Subjects_Complete': complete_sizes.get('n_subjects', 'N/A'),
                        'Obs_Lost': imputed_sizes.get('n_obs', 0) - complete_sizes.get('n_obs', 0),
                        'Subjects_Lost': imputed_sizes.get('n_subjects', 0) - complete_sizes.get('n_subjects', 0)
                    })
            else:
                # Stages and symptoms models have outcomes nested
                for outcome, sizes in model_data.get('imputed', {}).items():
                    complete_sizes = model_data.get('complete_cases', {}).get(outcome, {})

                    rows.append({
                        'Model': model_name,
                        'Outcome': outcome,
                        'N_Obs_Imputed': sizes.get('n_obs', 'N/A'),
                        'N_Subjects_Imputed': sizes.get('n_subjects', 'N/A'),
                        'N_Obs_Complete': complete_sizes.get('n_obs', 'N/A'),
                        'N_Subjects_Complete': complete_sizes.get('n_subjects', 'N/A'),
                        'Obs_Lost': sizes.get('n_obs', 0) - complete_sizes.get('n_obs', 0),
                        'Subjects_Lost': sizes.get('n_subjects', 0) - complete_sizes.get('n_subjects', 0)
                    })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Sample size comparison saved to: {output_file}")

    def write_sample_size_summary(self, f):
        """Write sample size summary to file."""
        for model_name, model_data in self.sample_sizes.items():
            f.write(f"\n{model_name.upper()} MODEL:\n")

            imputed_data = model_data.get('imputed', {})
            complete_data = model_data.get('complete_cases', {})

            if not imputed_data:
                f.write("  No data available\n")
                continue

            # Social model has different structure - no outcomes, just direct sample sizes
            if model_name == 'social':
                f.write(f"  cognitive_function:\n")
                f.write(f"    Imputed: {imputed_data.get('n_obs', 'N/A')} obs, "
                       f"{imputed_data.get('n_subjects', 'N/A')} subjects\n")
                f.write(f"    Complete: {complete_data.get('n_obs', 'N/A')} obs, "
                       f"{complete_data.get('n_subjects', 'N/A')} subjects\n")

                if isinstance(imputed_data.get('n_obs'), (int, float)) and isinstance(complete_data.get('n_obs'), (int, float)):
                    obs_lost = imputed_data['n_obs'] - complete_data['n_obs']
                    pct_lost = (obs_lost / imputed_data['n_obs'] * 100) if imputed_data['n_obs'] > 0 else 0
                    f.write(f"    Lost: {obs_lost} obs ({pct_lost:.1f}%)\n")
            else:
                # Stages and symptoms models have outcomes nested
                for outcome in imputed_data.keys():
                    imp_sizes = imputed_data[outcome]
                    comp_sizes = complete_data.get(outcome, {})

                    f.write(f"  {outcome}:\n")
                    f.write(f"    Imputed: {imp_sizes.get('n_obs', 'N/A')} obs, "
                           f"{imp_sizes.get('n_subjects', 'N/A')} subjects\n")
                    f.write(f"    Complete: {comp_sizes.get('n_obs', 'N/A')} obs, "
                           f"{comp_sizes.get('n_subjects', 'N/A')} subjects\n")

                    if isinstance(imp_sizes.get('n_obs'), (int, float)) and isinstance(comp_sizes.get('n_obs'), (int, float)):
                        obs_lost = imp_sizes['n_obs'] - comp_sizes['n_obs']
                        pct_lost = (obs_lost / imp_sizes['n_obs'] * 100) if imp_sizes['n_obs'] > 0 else 0
                        f.write(f"    Lost: {obs_lost} obs ({pct_lost:.1f}%)\n")


def main():
    # Paths to data files (relative to project root)
    imputed_file = os.path.join("..", "processed_data.csv")
    complete_cases_file = os.path.join("..", "processed_combined_data.csv")

    if not os.path.exists(imputed_file):
        print(f"Error: Imputed data file not found at {imputed_file}")
        return

    if not os.path.exists(complete_cases_file):
        print(f"Error: Complete cases file not found at {complete_cases_file}")
        return

    # Create comparator and run analysis
    comparator = ImputationComparator(imputed_file, complete_cases_file)
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
