import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.plot_config import STAGE_COLORS, set_apa_style
from utils.save_output import get_output_dir

class MenopauseDeclineAnalysis:
    """
    Analyze proportion of women experiencing decline in cognitive and emotional measures.
    Uses MCID (0.5 SD) thresholds for cognitive measures and any worsening for emotional measures.
    """

    def __init__(self, data):
        self.data = data
        self.output_dir = get_output_dir('1_stages_model')
        os.makedirs(self.output_dir, exist_ok=True)

        self.cognitive_measures = ['TOTIDE1', 'TOTIDE2']
        self.emotional_measures = ['NERVES', 'SAD', 'FEARFULA', 'NERVES_log', 'SAD_sqrt', 'FEARFULA_sqrt']
        self.status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']

        set_apa_style()
        self.mcid_thresholds = self._calculate_mcid_thresholds()

    def _calculate_mcid_thresholds(self):
        """Calculate MCID thresholds using 0.5 * SD from pre-menopause baseline."""
        mcid_thresholds = {}

        if 'STATUS_Label' in self.data.columns:
            baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause']
        else:
            baseline_data = self.data

        for measure in self.cognitive_measures:
            if measure in self.data.columns:
                measure_data = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
                if len(measure_data) > 0:
                    mcid_thresholds[measure] = 0.5 * measure_data.std()
                else:
                    measure_data = pd.to_numeric(self.data[measure], errors='coerce').dropna()
                    if len(measure_data) > 0:
                        mcid_thresholds[measure] = 0.5 * measure_data.std()

        return mcid_thresholds

    def calculate_decline_proportions(self):
        """Calculate proportion of women experiencing decline between visits."""
        if 'STATUS_Label' not in self.data.columns:
            return None

        for col in self.cognitive_measures + self.emotional_measures:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        print("\nMCID Thresholds for Cognitive Measures:")
        print("=" * 70)
        print(f"{'Measure':<15} {'Baseline SD':<15} {'MCID (0.5*SD)':<20}")
        print("-" * 70)

        baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause'] if 'STATUS_Label' in self.data.columns else self.data

        for measure, threshold in self.mcid_thresholds.items():
            measure_data = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
            if len(measure_data) > 0:
                print(f"{measure:<15} {measure_data.std():<15.3f} {threshold:<20.3f}")
        print("=" * 70)
        print("Note: Emotional measures use any worsening (no MCID threshold)")
        print("=" * 70)

        decline_data = []

        subjects = self.data['SWANID'].unique()
        for subject in subjects:
            subject_data = self.data[self.data['SWANID'] == subject].sort_values('VISIT')
            status = subject_data['STATUS_Label'].iloc[-1]

            for visit_idx in range(1, len(subject_data)):
                prev_visit = subject_data.iloc[visit_idx-1]
                curr_visit = subject_data.iloc[visit_idx]

                for measure in [m for m in self.cognitive_measures if m in self.data.columns]:
                    if not pd.isna(prev_visit[measure]) and not pd.isna(curr_visit[measure]):
                        change = curr_visit[measure] - prev_visit[measure]
                        mcid = self.mcid_thresholds.get(measure, 0)
                        decline_data.append({
                            'SWANID': subject,
                            'STATUS_Label': status,
                            'Measure': measure,
                            'Category': 'Cognitive',
                            'Change': change,
                            'Has_Decline': 1 if change < -mcid else 0
                        })

                for measure in [m for m in self.emotional_measures if m in self.data.columns]:
                    if not pd.isna(prev_visit[measure]) and not pd.isna(curr_visit[measure]):
                        change = curr_visit[measure] - prev_visit[measure]
                        decline_data.append({
                            'SWANID': subject,
                            'STATUS_Label': status,
                            'Measure': measure,
                            'Category': 'Emotional',
                            'Change': change,
                            'Has_Decline': 1 if change > 0 else 0
                        })

        decline_df = pd.DataFrame(decline_data)
        main_measures = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        filtered_decline_df = decline_df[decline_df['Measure'].isin(main_measures)]

        proportions = filtered_decline_df.groupby(['STATUS_Label', 'Measure', 'Category'])['Has_Decline'].mean()
        proportions_df = proportions.reset_index()

        category_proportions = filtered_decline_df.groupby(['STATUS_Label', 'Category'])['Has_Decline'].mean()
        category_proportions_df = category_proportions.reset_index()

        return proportions_df, category_proportions_df
    
    def plot_decline_proportions(self, proportions_df):
        """Create bar plots showing proportion with decline by stage."""
        if proportions_df is None or proportions_df.empty:
            return

        if 'STATUS_Label' in proportions_df.columns:
            proportions_df['STATUS_Label'] = pd.Categorical(
                proportions_df['STATUS_Label'],
                categories=self.status_order,
                ordered=True
            )
            proportions_df = proportions_df.sort_values('STATUS_Label')

        measure_order = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        measure_labels = {
            'TOTIDE1': 'Cognitive Function (Immediate Recall)',
            'TOTIDE2': 'Cognitive Function (Delayed Recall)',
            'NERVES': 'Nervousness',
            'SAD': 'Sadness',
            'FEARFULA': 'Fearfulness'
        }

        plot_data = proportions_df[proportions_df['Measure'].isin(measure_order)].copy()
        plot_data['Measure_Label'] = plot_data['Measure'].map(measure_labels)

        fig, axes = plt.subplots(5, 1, figsize=(14, 24))
        axes = axes.flatten()

        stage_colors_list = [STAGE_COLORS[stage] for stage in self.status_order]

        for idx, measure in enumerate(measure_order):
            ax = axes[idx]
            measure_data = plot_data[plot_data['Measure'] == measure]

            if measure_data.empty:
                continue

            bars = ax.bar(
                range(len(measure_data)),
                measure_data['Has_Decline'],
                color=stage_colors_list[:len(measure_data)],
                alpha=0.8, edgecolor='black', linewidth=0.5
            )

            for bar, value in zip(bars, measure_data['Has_Decline']):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{value*100:.1f}%',
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=18
                )

            ax.set_title(f'{measure_labels[measure]}', fontsize=14)
            sns.despine(ax=ax)

            ax.set_xticks(range(len(measure_data)))
            ax.set_xticklabels(measure_data['STATUS_Label'], rotation=45, ha='right', fontsize=18)
            ax.tick_params(axis='y', labelsize=18)

            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
            ax.set_ylim(0, max(0.3, measure_data['Has_Decline'].max() * 1.2))
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)

            if measure in ['TOTIDE1', 'TOTIDE2']:
                mcid_value = self.mcid_thresholds.get(measure, 0)
                if 'STATUS_Label' in self.data.columns:
                    baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause']
                    measure_data_for_sd = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
                    if len(measure_data_for_sd) > 0:
                        baseline_sd = measure_data_for_sd.std()
                        ax.text(
                            0.02, 0.98,
                            f'MCID threshold: {mcid_value:.2f} points\n(0.5 × baseline SD={baseline_sd:.2f})',
                            transform=ax.transAxes, va='top', ha='left', fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
                        )

        plt.tight_layout()
        fig.suptitle(
            'Proportion of Women with Decline by Menopausal Stage\n(Cognitive: MCID threshold; Emotional: Any worsening)',
            fontsize=15, fontweight='bold', y=0.998
        )
        plt.subplots_adjust(top=0.97)

        output_path = os.path.join(self.output_dir, 'menopausal_decline_proportions_mcid.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return fig
    
    def tabulate_results(self, proportions_df, category_proportions_df):
        """Print formatted tables of decline proportions and methodology."""
        proportions_df['Decline_Percentage'] = (proportions_df['Has_Decline'] * 100).round(1)
        category_proportions_df['Decline_Percentage'] = (category_proportions_df['Has_Decline'] * 100).round(1)

        women_counts = self.data.groupby('STATUS_Label')['SWANID'].nunique()

        print("\n" + "=" * 80)
        print("DECLINE PROPORTION ANALYSIS")
        print("=" * 80)
        print("\nMethodology:")
        print("  - Cognitive measures: MCID threshold (0.5 × baseline SD)")
        print("  - Emotional measures: Any worsening (no MCID threshold)")
        print("  - Baseline SD calculated from pre-menopause group")

        if 'STATUS_Label' in self.data.columns:
            baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause']
            for measure in ['TOTIDE1', 'TOTIDE2']:
                if measure in self.data.columns:
                    totide_data = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
                    if len(totide_data) > 0:
                        totide_sd = totide_data.std()
                        totide_mcid = self.mcid_thresholds.get(measure, 0)
                        measure_name = 'Immediate Recall' if measure == 'TOTIDE1' else 'Delayed Recall'
                        print(f"  - Cognitive ({measure} - {measure_name}): baseline SD={totide_sd:.3f}, MCID={totide_mcid:.3f} points")

        print("\nDecline Definition:")
        print("  - Cognitive: Decrease exceeding MCID threshold between visits")
        print("  - Emotional: Any increase between visits")
        print("=" * 80)

        print("\nProportion of Women Experiencing Decline:")
        print("(Cognitive: exceeding MCID; Emotional: any worsening)")
        print("=" * 80)
        print(f"{'Status':<15} {'Category':<10} {'Proportion (%)':<15} {'Sample Size':<12}")
        print("-" * 80)

        for status in self.status_order:
            if status in category_proportions_df['STATUS_Label'].values:
                for category in ['Cognitive', 'Emotional']:
                    row = category_proportions_df[
                        (category_proportions_df['STATUS_Label'] == status) &
                        (category_proportions_df['Category'] == category)
                    ]
                    if not row.empty:
                        sample_size = women_counts.get(status, 0)
                        print(f"{status:<15} {category:<10} {row['Decline_Percentage'].values[0]:<15.1f} {sample_size:<12}")

        print("=" * 80)
        print("\nDetailed Breakdown by Specific Measures:")
        print("=" * 80)
        print(f"{'Status':<15} {'Measure':<12} {'Category':<10} {'Threshold':<12} {'Proportion (%)':<15}")
        print("-" * 80)

        for status in self.status_order:
            status_rows = proportions_df[proportions_df['STATUS_Label'] == status]
            if not status_rows.empty:
                for _, row in status_rows.iterrows():
                    measure = row['Measure']
                    category = row['Category']
                    threshold_str = f"{self.mcid_thresholds.get(measure, 0):.3f}" if category == 'Cognitive' else "Any change"
                    print(f"{status:<15} {measure:<12} {category:<10} {threshold_str:<12} {row['Decline_Percentage']:<15.1f}")

        print("=" * 80)

        return proportions_df, category_proportions_df

    def run_analysis(self):
        """Run complete decline proportion analysis."""
        proportions_df, category_proportions_df = self.calculate_decline_proportions()

        if proportions_df is not None and category_proportions_df is not None:
            self.plot_decline_proportions(proportions_df)
            return self.tabulate_results(proportions_df, category_proportions_df)
        return None, None