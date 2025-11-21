import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.plot_config import (STAGE_COLORS, get_categorical_palette,
                               set_apa_style, CONSTRUCT_COLORS)

class MenopauseDeclineAnalysis:
    """
    Analyzes the proportion of women experiencing cognitive decline
    or emotional worsening across different menopausal stages.

    For cognitive measures: Uses a Minimally Clinically Important Difference (MCID)
    threshold based on the distribution-based 0.5 standard deviation (SD) method.

    For emotional measures: Uses any worsening (any increase in symptoms).
    """

    def __init__(self, data):
        self.data = data
        # Use centralized output directory
        from utils.save_output import get_output_dir
        self.output_dir = get_output_dir('1_stages_model')
        os.makedirs(self.output_dir, exist_ok=True)

        # Define cognitive and emotional measures
        self.cognitive_measures = ['TOTIDE1', 'TOTIDE2']
        self.emotional_measures = ['NERVES', 'SAD', 'FEARFULA', 'NERVES_log', 'SAD_sqrt', 'FEARFULA_sqrt']

        # Status order for visualization
        self.status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']

        # Apply APA style
        set_apa_style()

        # Calculate MCID thresholds
        self.mcid_thresholds = self._calculate_mcid_thresholds()

    def _calculate_mcid_thresholds(self):
        """
        Calculate Minimally Clinically Important Difference (MCID) thresholds
        using the distribution-based 0.5 standard deviation (SD) method.

        Calculates SD from baseline (pre-menopause) data in the current dataset.
        MCID is applied only to cognitive measures.

        Returns:
            dict: Dictionary mapping measure names to their MCID threshold values
        """
        mcid_thresholds = {}

        # Get baseline data from pre-menopause group
        if 'STATUS_Label' in self.data.columns:
            baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause']
        else:
            # Fallback to all data if STATUS_Label not available yet
            baseline_data = self.data

        # Calculate MCID only for cognitive measures
        # Using 0.5 * SD from the baseline (pre-menopause) data
        for measure in self.cognitive_measures:
            if measure in self.data.columns:
                # Convert to numeric and drop NaN values
                measure_data = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
                if len(measure_data) > 0:
                    sd = measure_data.std()
                    mcid_thresholds[measure] = 0.5 * sd
                else:
                    # Fallback to overall SD if no baseline data available
                    measure_data = pd.to_numeric(self.data[measure], errors='coerce').dropna()
                    if len(measure_data) > 0:
                        sd = measure_data.std()
                        mcid_thresholds[measure] = 0.5 * sd

        return mcid_thresholds

    def calculate_decline_proportions(self):
        """
        Calculate the proportion of women who experience decline in cognitive
        and emotional measures across visits, grouped by menopausal stage.

        Uses MCID (Minimally Clinically Important Difference) thresholds for cognitive
        measures to focus on meaningful changes. For emotional measures, uses any change.
        """
        # Ensure we have the STATUS_Label column
        if 'STATUS_Label' not in self.data.columns:
            print("Error: STATUS_Label column not found. Please ensure filter_status() has been run.")
            return None

        # Convert measures to numeric if needed
        for col in self.cognitive_measures + self.emotional_measures:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Print MCID thresholds for reference (cognitive measures only)
        print("\nMCID Thresholds for Cognitive Measures:")
        print("=" * 70)
        print(f"{'Measure':<15} {'Baseline SD':<15} {'MCID (0.5*SD)':<20}")
        print("-" * 70)

        # Get baseline data to calculate SDs for display
        if 'STATUS_Label' in self.data.columns:
            baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause']
        else:
            baseline_data = self.data

        for measure, threshold in self.mcid_thresholds.items():
            measure_data = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
            if len(measure_data) > 0:
                sd = measure_data.std()
                print(f"{measure:<15} {sd:<15.3f} {threshold:<20.3f}")
        print("=" * 70)
        print("Note: Emotional measures use any worsening (no MCID threshold)")
        print("=" * 70)

        # Group by woman ID and sort by visit to calculate changes
        decline_data = []

        # Process all subjects with multiple visits
        subjects = self.data['SWANID'].unique()
        for subject in subjects:
            subject_data = self.data[self.data['SWANID'] == subject].sort_values('VISIT')

            # Get the status label (using the most recent status)
            status = subject_data['STATUS_Label'].iloc[-1]

            # Calculate changes in measures
            for visit_idx in range(1, len(subject_data)):
                prev_visit = subject_data.iloc[visit_idx-1]
                curr_visit = subject_data.iloc[visit_idx]

                # Cognitive measures (higher is better, so decline is negative change exceeding MCID)
                for measure in [m for m in self.cognitive_measures if m in self.data.columns]:
                    if not pd.isna(prev_visit[measure]) and not pd.isna(curr_visit[measure]):
                        change = curr_visit[measure] - prev_visit[measure]
                        mcid = self.mcid_thresholds.get(measure, 0)
                        # Decline if change is negative and exceeds MCID threshold
                        has_decline = 1 if change < -mcid else 0
                        decline_data.append({
                            'SWANID': subject,
                            'STATUS_Label': status,
                            'Measure': measure,
                            'Category': 'Cognitive',
                            'Change': change,
                            'Has_Decline': has_decline
                        })

                # Emotional measures (higher is worse, so worsening is any positive change)
                for measure in [m for m in self.emotional_measures if m in self.data.columns]:
                    if not pd.isna(prev_visit[measure]) and not pd.isna(curr_visit[measure]):
                        change = curr_visit[measure] - prev_visit[measure]
                        # Worsening if change is positive (any increase)
                        has_decline = 1 if change > 0 else 0
                        decline_data.append({
                            'SWANID': subject,
                            'STATUS_Label': status,
                            'Measure': measure,
                            'Category': 'Emotional',
                            'Change': change,
                            'Has_Decline': has_decline
                        })
        
        # Convert to DataFrame
        decline_df = pd.DataFrame(decline_data)
        
        # Filter to main measures to avoid duplicates from transformed variables
        main_measures = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        filtered_decline_df = decline_df[decline_df['Measure'].isin(main_measures)]
        
        # Calculate proportion with decline by status and measure
        proportions = filtered_decline_df.groupby(['STATUS_Label', 'Measure', 'Category'])['Has_Decline'].mean()
        proportions_df = proportions.reset_index()
        
        # Calculate overall proportions by status and category
        category_proportions = filtered_decline_df.groupby(['STATUS_Label', 'Category'])['Has_Decline'].mean()
        category_proportions_df = category_proportions.reset_index()
        
        return proportions_df, category_proportions_df
    
    def plot_decline_proportions(self, proportions_df):
        """
        Create a visualization showing the proportion of women experiencing decline
        across all four measures by menopausal stage.

        Cognitive measures use MCID threshold; emotional measures use any worsening.
        """
        if proportions_df is None or proportions_df.empty:
            print("Error: No decline proportion data available.")
            return
        
        # Ensure STATUS_Label is in the correct order
        if 'STATUS_Label' in proportions_df.columns:
            proportions_df['STATUS_Label'] = pd.Categorical(
                proportions_df['STATUS_Label'],
                categories=self.status_order,
                ordered=True
            )
            proportions_df = proportions_df.sort_values('STATUS_Label')
        
        # Define measure order and labels
        measure_order = ['TOTIDE1', 'TOTIDE2', 'NERVES', 'SAD', 'FEARFULA']
        measure_labels = {
            'TOTIDE1': 'Cognitive Function (Immediate Recall)',
            'TOTIDE2': 'Cognitive Function (Delayed Recall)',
            'NERVES': 'Nervousness',
            'SAD': 'Sadness',
            'FEARFULA': 'Fearfulness'
        }

        # Filter to main measures
        plot_data = proportions_df[proportions_df['Measure'].isin(measure_order)].copy()

        # Create measure labels
        plot_data['Measure_Label'] = plot_data['Measure'].map(measure_labels)

        # Initialize the plot with subplots for each measure
        fig, axes = plt.subplots(5, 1, figsize=(14, 24))
        axes = axes.flatten()

        # Use colorblind-friendly colors - one color per stage
        stage_colors_list = [STAGE_COLORS[stage] for stage in self.status_order]
        
        for idx, measure in enumerate(measure_order):
            ax = axes[idx]
            
            # Filter data for this measure
            measure_data = plot_data[plot_data['Measure'] == measure]
            
            if measure_data.empty:
                continue
                
            # Create bar plot with stage-specific colors
            bars = ax.bar(
                range(len(measure_data)),
                measure_data['Has_Decline'],
                color=stage_colors_list[:len(measure_data)],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add value labels on top of each bar
            for i, (bar, value) in enumerate(zip(bars, measure_data['Has_Decline'])):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + 0.01,
                    f'{value*100:.1f}%', 
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=18
                )
            
            # Customize subplot (APA format)
            ax.set_title(f'{measure_labels[measure]}', fontsize=14)
            sns.despine(ax=ax)

            # Set x-axis labels
            ax.set_xticks(range(len(measure_data)))
            ax.set_xticklabels(measure_data['STATUS_Label'], rotation=45, ha='right', fontsize=18)
            ax.tick_params(axis='y', labelsize=18)

            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

            # Set consistent y-axis limits
            ax.set_ylim(0, max(0.3, measure_data['Has_Decline'].max() * 1.2))

            # Add grid for better readability
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)

            # Add MCID threshold as text annotation for cognitive measures
            if measure in ['TOTIDE1', 'TOTIDE2']:
                mcid_value = self.mcid_thresholds.get(measure, 0)
                # Calculate baseline SD for display
                if 'STATUS_Label' in self.data.columns:
                    baseline_data = self.data[self.data['STATUS_Label'] == 'Pre-menopause']
                    measure_data_for_sd = pd.to_numeric(baseline_data[measure], errors='coerce').dropna()
                    if len(measure_data_for_sd) > 0:
                        baseline_sd = measure_data_for_sd.std()
                        ax.text(
                            0.02, 0.98,
                            f'MCID threshold: {mcid_value:.2f} points\n(0.5 × baseline SD={baseline_sd:.2f})',
                            transform=ax.transAxes,
                            va='top',
                            ha='left',
                            fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
                        )
        
        # Adjust layout and add figure title
        plt.tight_layout()
        fig.suptitle(
            'Proportion of Women with Decline by Menopausal Stage\n(Cognitive: MCID threshold; Emotional: Any worsening)',
            fontsize=15,
            fontweight='bold',
            y=0.998
        )
        plt.subplots_adjust(top=0.97)

        # Save the figure
        output_path = os.path.join(self.output_dir, 'menopausal_decline_proportions_mcid.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {output_path}")
        
        return fig
    
    def tabulate_results(self, proportions_df, category_proportions_df):
        """
        Create tables showing the detailed results for reporting.
        Includes MCID threshold information.
        """
        # Format proportions as percentages
        proportions_df['Decline_Percentage'] = (proportions_df['Has_Decline'] * 100).round(1)
        category_proportions_df['Decline_Percentage'] = (category_proportions_df['Has_Decline'] * 100).round(1)

        # Count number of women in each stage who had multiple visits
        women_counts = self.data.groupby('STATUS_Label')['SWANID'].nunique()

        # Print analysis information header
        print("\n" + "=" * 80)
        print("DECLINE PROPORTION ANALYSIS")
        print("=" * 80)
        print("\nMethodology:")
        print("  - Cognitive measures: MCID threshold (0.5 × baseline SD)")
        print("  - Emotional measures: Any worsening (no MCID threshold)")
        print("  - Baseline SD calculated from pre-menopause group")

        # Calculate and display baseline SD for cognitive measures
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

        # Print detailed results
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
                    # Show MCID for cognitive, "Any change" for emotional
                    if category == 'Cognitive':
                        mcid_val = self.mcid_thresholds.get(measure, 0)
                        threshold_str = f"{mcid_val:.3f}"
                    else:
                        threshold_str = "Any change"
                    print(f"{status:<15} {measure:<12} {category:<10} {threshold_str:<12} {row['Decline_Percentage']:<15.1f}")
        
        print("=" * 80)
        
        # Return formatted dataframes for further analysis
        return proportions_df, category_proportions_df
    
    def run_analysis(self):
        """
        Run the complete decline proportion analysis.
        """
        print("\nAnalyzing proportion of women experiencing cognitive and emotional decline...")
        
        # Calculate decline proportions
        proportions_df, category_proportions_df = self.calculate_decline_proportions()
        
        if proportions_df is not None and category_proportions_df is not None:
            # Create visualization
            self.plot_decline_proportions(proportions_df)
            
            # Generate detailed tables
            formatted_prop_df, formatted_cat_df = self.tabulate_results(proportions_df, category_proportions_df)
            
            return formatted_prop_df, formatted_cat_df
        else:
            print("Error: Could not calculate decline proportions.")
            return None, None