import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils.plot_config import (SIGNIFICANCE_COLORS, get_significance_color,
                               set_apa_style, get_categorical_palette, CORRELATION_CMAP)


def plot_symptom_effects(analysis_obj):
    """Create forest plots showing symptom effects on cognitive outcomes."""
    if not analysis_obj.mixed_model_results:
        print("No mixed model results available. Run run_mixed_models() first.")
        return

    set_apa_style()
    print("\nCreating forest plots for transformed models...")

    # Get unique outcomes from the keys
    unique_outcomes = set()
    for key in analysis_obj.mixed_model_results.keys():
        outcome = key.split('_')[0]
        unique_outcomes.add(outcome)

    # Group results by outcome variable
    for outcome in unique_outcomes:
        # Get all models for this outcome
        outcome_models = {k: v for k, v in analysis_obj.mixed_model_results.items() if k.startswith(f"{outcome}_")}

        if not outcome_models:
            print(f"No models found for {outcome}")
            continue

        print(f"Creating forest plot for {outcome} with {len(outcome_models)} symptoms")

        # Create figure for this outcome
        plt.figure(figsize=(12, 16))

        # Initialize lists to store plot data
        symptoms = []
        coefs = []
        errors = []
        pvalues = []
        colors = []
        stars = []

        # Extract coefficients for each symptom
        for key, results in outcome_models.items():
            symptom = key.split('_')[1]

            # Get coefficient for symptom (it's the first predictor after the intercept)
            param_name = symptom

            if param_name in results.params.index:
                p_value = results.pvalues[param_name]
                color, star = get_significance_color(p_value)

                symptoms.append(analysis_obj.var_labels.get(symptom, symptom))
                coefs.append(results.params[param_name])
                errors.append(results.bse[param_name])
                pvalues.append(p_value)
                colors.append(color)
                stars.append(star)
            else:
                print(f"Warning: Coefficient for {symptom} not found in model results")

        if not symptoms:
            print(f"No valid coefficients found for {outcome}")
            continue

        # Sort by coefficient magnitude for better visualization
        sorted_indices = np.argsort(np.abs(coefs))[::-1]  # Sort by absolute value, descending
        symptoms = [symptoms[i] for i in sorted_indices]
        coefs = [coefs[i] for i in sorted_indices]
        errors = [errors[i] for i in sorted_indices]
        pvalues = [pvalues[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        stars = [stars[i] for i in sorted_indices]

        # Create forest plot with color-coded points
        y_pos = np.arange(len(symptoms))

        # Plot each point individually with its own color
        for i, (coef, error, color) in enumerate(zip(coefs, errors, colors)):
            plt.errorbar(
                coef, y_pos[i],
                xerr=1.96 * error,  # 95% CI
                fmt='o',
                color=color,
                capsize=5,
                markersize=8,
                elinewidth=2,
                capthick=2
            )

        # Add vertical line at zero
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Customize plot
        plt.yticks(y_pos, symptoms, fontsize=18)
        plt.xticks(fontsize=18)

        # Calculate reasonable x-axis limits based on coefficients and errors
        all_values = []
        for coef, error in zip(coefs, errors):
            all_values.extend([coef - 1.96 * error, coef + 1.96 * error])

        x_min, x_max = min(all_values), max(all_values)
        x_range = x_max - x_min
        plt.xlim(x_min - 0.1 * x_range, x_max + 0.3 * x_range)  # Extra space on right for labels

        # Add text labels with p-values - aligned to the right
        for i, (coef, error, p, star, color) in enumerate(zip(coefs, errors, pvalues, stars, colors)):
            # Calculate the right edge of the error bar
            upper_error = coef + 1.96 * error

            # Position label to the right of the error bar with some padding
            label_x = upper_error + 0.02 * x_range

            plt.text(
                label_x, y_pos[i],
                f'{coef:.3f} {star}',
                verticalalignment='center',
                horizontalalignment='left',
                fontsize=18,
                color=color,
                fontweight='bold' if p < 0.05 else 'normal',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none')
            )

        # Add gridlines
        plt.grid(True, axis='x', linestyle=':', alpha=0.6)

        # Create custom legend for significance levels and colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=SIGNIFICANCE_COLORS['p<0.001'], label='p < 0.001 (***)', edgecolor='black'),
            Patch(facecolor=SIGNIFICANCE_COLORS['p<0.01'], label='p < 0.01 (**)', edgecolor='black'),
            Patch(facecolor=SIGNIFICANCE_COLORS['p<0.05'], label='p < 0.05 (*)', edgecolor='black'),
            Patch(facecolor=SIGNIFICANCE_COLORS['ns'], label='p ≥ 0.05 (n.s.)', edgecolor='black')
        ]

        plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        file_name = os.path.join(analysis_obj.output_dir, f'{outcome}_forest_plot.png')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Forest plot saved to: 2_symptoms_model/output/{outcome}_forest_plot.png")


def create_stratified_forest_plot(analysis_obj, outcome='TOTIDE1'):
    """Create stratified forest plots showing symptom effects across demographic subgroups."""
    import statsmodels.formula.api as smf

    if not analysis_obj.mixed_model_results:
        print("No mixed model results available. Run run_mixed_models() first.")
        return

    print(f"\nCreating stratified forest plot for {outcome}...")

    # Simplified: Only use key stratifications
    stratifications = {
        'Overall': None,  # Overall effect
        'Menopausal Stage': 'STATUS_Label',
        'Age Group': 'AGE_GROUP'
    }

    # Create age groups if not already present
    if 'AGE_GROUP' not in analysis_obj.data.columns:
        analysis_obj.data['AGE_GROUP'] = pd.cut(analysis_obj.data['AGE'],
                                        bins=[0, 45, 55, 65, 100],
                                        labels=['<45', '45-54', '55-64', '65+'])

    # Collect all results for plotting
    plot_data = []

    # Use only top 3 symptoms for clarity
    symptoms_to_plot = ['NUMHOTF', 'NUMNITS', 'IRRITAB']  # Hot flashes, night sweats, irritability

    # For each stratification
    for strat_name, strat_var in stratifications.items():
        if strat_var is None:
            # Overall analysis (already have this from main model)
            key = f"{outcome}"
            if key in analysis_obj.mixed_model_results:
                result = analysis_obj.mixed_model_results[key]

                for symptom in symptoms_to_plot:
                    if symptom in result.params.index:
                        coef = result.params[symptom]
                        se = result.bse[symptom]
                        p_val = result.pvalues[symptom]

                        plot_data.append({
                            'Stratification': 'Overall',
                            'Group': 'All participants',
                            'Symptom': analysis_obj.var_labels.get(symptom, symptom),
                            'Coefficient': coef,
                            'SE': se,
                            'CI_lower': coef - 1.96 * se,
                            'CI_upper': coef + 1.96 * se,
                            'P_value': p_val
                        })
        else:
            # Stratified analysis
            if strat_var not in analysis_obj.data.columns:
                print(f"Warning: {strat_var} not in data, skipping {strat_name}")
                continue

            # Get unique categories
            categories = analysis_obj.data[strat_var].dropna().unique()

            for category in categories:
                # Subset data and reset index
                subset_data = analysis_obj.data[analysis_obj.data[strat_var] == category].copy()
                subset_data = subset_data.reset_index(drop=True)

                if len(subset_data) < 50:  # Minimum sample size
                    continue

                # Run a simple model for this subset
                try:
                    for symptom in symptoms_to_plot:
                        if symptom not in subset_data.columns:
                            continue

                        # Drop rows with missing values for this analysis
                        analysis_data = subset_data[[outcome, symptom, 'SWANID']].dropna()

                        if len(analysis_data) < 50:
                            continue

                        # Simple regression: outcome ~ symptom
                        formula = f"{outcome} ~ {symptom}"
                        model = smf.mixedlm(formula=formula,
                                          data=analysis_data,
                                          groups=analysis_data["SWANID"])
                        result = model.fit(method='lbfgs', maxiter=100)

                        if symptom in result.params.index:
                            coef = result.params[symptom]
                            se = result.bse[symptom]
                            p_val = result.pvalues[symptom]

                            plot_data.append({
                                'Stratification': strat_name,
                                'Group': str(category),
                                'Symptom': analysis_obj.var_labels.get(symptom, symptom),
                                'Coefficient': coef,
                                'SE': se,
                                'CI_lower': coef - 1.96 * se,
                                'CI_upper': coef + 1.96 * se,
                                'P_value': p_val
                            })
                except Exception as e:
                    print(f"Error in stratified analysis for {strat_name}-{category}: {str(e)}")
                    continue

    if not plot_data:
        print("No data available for stratified forest plot")
        return

    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)

    # Create separate subplots for each stratification
    stratification_groups = plot_df['Stratification'].unique()
    n_panels = len(stratification_groups)

    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 8), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for panel_idx, strat_group in enumerate(stratification_groups):
        ax = axes[panel_idx]

        # Get data for this stratification
        panel_data = plot_df[plot_df['Stratification'] == strat_group].copy()
        panel_data = panel_data.reset_index(drop=True)

        y_positions = np.arange(len(panel_data))

        # Plot each estimate with its CI
        for idx, row in panel_data.iterrows():
            color, stars = get_significance_color(row['P_value'])

            # Plot error bar
            ax.errorbar(row['Coefficient'], y_positions[idx],
                       xerr=[[row['Coefficient'] - row['CI_lower']],
                             [row['CI_upper'] - row['Coefficient']]],
                       fmt='o', color=color, capsize=3, markersize=6,
                       elinewidth=1.5, capthick=1.5)

            # Add coefficient text
            ax.text(row['CI_upper'] + 0.05, y_positions[idx],
                   f"{row['Coefficient']:.2f}{stars}",
                   va='center', ha='left', fontsize=9, color=color)

        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        # Create clean y-axis labels
        y_labels = [f"{row['Group']}: {row['Symptom']}" for _, row in panel_data.iterrows()]

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.set_xlabel('Estimate (95% CI)', fontsize=11)
        ax.set_title(f'{strat_group}', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)
        sns.despine(ax=ax)

    fig.suptitle(f'Symptom Effects on {analysis_obj.var_labels.get(outcome, outcome)}\nby Key Demographics',
                fontsize=14, fontweight='bold', y=0.98)

    # Add legend to the rightmost panel
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SIGNIFICANCE_COLORS['p<0.001'], label='p < 0.001 (***)', edgecolor='black'),
        Patch(facecolor=SIGNIFICANCE_COLORS['p<0.01'], label='p < 0.01 (**)', edgecolor='black'),
        Patch(facecolor=SIGNIFICANCE_COLORS['p<0.05'], label='p < 0.05 (*)', edgecolor='black'),
        Patch(facecolor=SIGNIFICANCE_COLORS['ns'], label='p ≥ 0.05 (n.s.)', edgecolor='black')
    ]
    axes[-1].legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=False)

    plt.tight_layout()

    # Save
    output_file = os.path.join(analysis_obj.output_dir, f'{outcome}_stratified_forest_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Stratified forest plot saved to: 2_symptoms_model/output/{outcome}_stratified_forest_plot.png")

    return plot_df


def analyze_symptom_intensity_by_stage(analysis_obj):
    """Analyze symptom intensity variation across menopausal stages."""
    print("\nAnalyzing symptom intensity variation across menopausal stages...")

    # Create output directory for these specific results
    intensity_dir = os.path.join(analysis_obj.output_dir, "symptom_intensity")
    os.makedirs(intensity_dir, exist_ok=True)

    # Define the symptom groups to analyze
    symptom_groups = {
        'Hot Flashes': ['NUMHOTF'],
        'Night Sweats': ['NUMNITS'],
        'Cold Sweats': ['NUMCLDS'],
        'Mood': ['IRRITAB', 'MOODCHG'],
        'Stiffness': ['STIFF'],
    }

    # Status order for plotting
    status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']

    # Initialize a DataFrame to store results
    results_data = []

    # Analyze each symptom group
    for symptom_group, symptom_vars in symptom_groups.items():
        print(f"\nAnalyzing {symptom_group} symptoms...")

        # Create figure for this symptom group
        fig, axes = plt.subplots(1, len(symptom_vars), figsize=(len(symptom_vars)*5, 6), squeeze=False)
        axes = axes.flatten()

        # For each measure of the symptom
        for i, symptom_var in enumerate(symptom_vars):
            if symptom_var not in analysis_obj.data.columns:
                print(f"Warning: {symptom_var} not found in data")
                continue

            # Compute summary statistics
            summary = analysis_obj.data.groupby('STATUS_Label', observed=True)[symptom_var].agg([
                'count', 'mean', 'std', 'median', 'min', 'max'
            ]).reset_index()

            # Convert summary to proper category type with correct order
            summary['STATUS_Label'] = pd.Categorical(
                summary['STATUS_Label'],
                categories=status_order,
                ordered=True
            )

            # Sort by the ordered category
            summary = summary.sort_values('STATUS_Label')

            # Store results for later reporting
            for _, row in summary.iterrows():
                results_data.append({
                    'Symptom Group': symptom_group,
                    'Symptom': analysis_obj.var_labels.get(symptom_var, symptom_var),
                    'Menopausal Stage': row['STATUS_Label'],
                    'Count': row['count'],
                    'Mean': row['mean'],
                    'StdDev': row['std'],
                    'Median': row['median'],
                    'Min': row['min'],
                    'Max': row['max']
                })

            # Create a bar plot
            ax = axes[i]

            # Calculate standard error for error bars
            summary['se'] = summary['std'] / np.sqrt(summary['count'])

            # Create the bar plot with error bars
            bars = ax.bar(
                x=np.arange(len(summary)),
                height=summary['mean'],
                yerr=summary['se'],
                capsize=4,
                width=0.7,
                color=sns.color_palette("YlGnBu", n_colors=len(summary))
            )

            # Add mean value labels on top of each bar
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1,
                    f'{summary["mean"].iloc[j]:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    rotation=0
                )

            # Set axis labels and title
            ax.set_title(analysis_obj.var_labels.get(symptom_var, symptom_var), fontsize=12)
            ax.set_ylabel('Mean Score (with SE)', fontsize=10)
            ax.set_xticks(np.arange(len(summary)))
            ax.set_xticklabels(summary['STATUS_Label'], rotation=45, ha='right', fontsize=9)

        # Add an overall title
        fig.suptitle(f'{symptom_group} Intensity by Menopausal Stage', fontsize=14, y=1.05)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(
            os.path.join(intensity_dir, f'{symptom_group.lower().replace(" ", "_")}_intensity.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # Create a comprehensive summary table as a DataFrame
    results_df = pd.DataFrame(results_data)

    # Create formatted tables for each symptom group
    for symptom_group in results_df['Symptom Group'].unique():
        group_data = results_df[results_df['Symptom Group'] == symptom_group]

        # Pivot the data to create a nicely formatted table
        pivot_mean = pd.pivot_table(
            group_data,
            values='Mean',
            index='Menopausal Stage',
            columns='Symptom',
            aggfunc='first'
        )

        # Add count information (average across symptoms)
        count_data = group_data.groupby('Menopausal Stage')['Count'].mean().astype(int)
        pivot_mean['Sample Size'] = count_data

        # Sort rows by menopausal stage order
        pivot_mean = pivot_mean.reindex(status_order)

        # Print the table
        print(f"\n{symptom_group} Intensity by Menopausal Stage:")
        print("=" * 80)
        print(pivot_mean.round(2).to_string())
        print("=" * 80)

        # Save the table to a CSV file
        pivot_mean.to_csv(
            os.path.join(intensity_dir, f'{symptom_group.lower().replace(" ", "_")}_intensity.csv')
        )

    # Plot overall symptom intensity pattern
    plot_overall_symptom_intensity_pattern(analysis_obj, results_df, intensity_dir)

    print("Symptom intensity analysis complete. Results saved to: 2_symptoms_model/output/symptom_intensity/")
    return results_df


def plot_overall_symptom_intensity_pattern(analysis_obj, results_df, output_dir):
    """Create visualization showing symptom intensity patterns across menopausal stages."""
    # Status order
    status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']

    # Create figure for the comprehensive visualization
    plt.figure(figsize=(14, 12))

    # Add a subplot for symptom intensity patterns
    plt.subplot(111)

    # Group by symptom group and menopausal stage, calculating mean intensity
    intensity_pattern = results_df.groupby(['Symptom Group', 'Menopausal Stage'])['Mean'].mean().reset_index()

    # Convert to proper categorical type
    intensity_pattern['Menopausal Stage'] = pd.Categorical(
        intensity_pattern['Menopausal Stage'],
        categories=status_order,
        ordered=True
    )

    # Sort by stage
    intensity_pattern = intensity_pattern.sort_values('Menopausal Stage')

    # Use colorblind-friendly colors for symptom groups
    symptom_groups = intensity_pattern['Symptom Group'].unique()
    symptom_group_colors = get_categorical_palette(len(symptom_groups))

    # Create a color dictionary
    color_dict = dict(zip(symptom_groups, symptom_group_colors))

    # Create line plot with custom colors
    sns.lineplot(
        data=intensity_pattern,
        x='Menopausal Stage',
        y='Mean',
        hue='Symptom Group',
        marker='o',
        markersize=10,
        linewidth=3,
        err_style='band',
        palette=color_dict
    )

    # Set title and labels (APA format)
    plt.xlabel('Menopausal Stage', fontsize=12)
    plt.ylabel('Mean Intensity Score', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    sns.despine()

    # Enhance the legend (APA format)
    plt.legend(
        title='Symptom Group',
        fontsize=11,
        title_fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_symptom_intensity_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_symptom_intensity_heatmap(analysis_obj):
    """Create heatmap showing symptom intensity across menopausal stages."""
    print("\nCreating symptom intensity heat map...")

    # Define symptom groups (matching your analyze_symptom_intensity_by_stage method)
    symptom_groups = {
        'Hot Flashes': ['NUMHOTF'],
        'Night Sweats': ['NUMNITS'],
        'Cold Sweats': ['NUMCLDS'],
        'Mood Symptoms': ['IRRITAB', 'MOODCHG'],
        'Stiffness': ['STIFF']
    }

    # Status order for the heat map (rows)
    status_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause', 'Surgical']

    # Initialize matrix to store z-scored intensities
    heatmap_data = pd.DataFrame(index=status_order, columns=list(symptom_groups.keys()))

    # Calculate mean intensity for each symptom group and menopausal stage
    for group_name, symptom_vars in symptom_groups.items():
        print(f"Processing {group_name}...")

        # Get available symptoms from this group
        available_symptoms = [var for var in symptom_vars if var in analysis_obj.data.columns]

        if not available_symptoms:
            print(f"Warning: No symptoms found for {group_name}")
            continue

        # Calculate mean intensity across symptoms in this group for each stage
        group_means = []
        for status in status_order:
            # Get data for this status
            status_data = analysis_obj.data[analysis_obj.data['STATUS_Label'] == status]

            if len(status_data) == 0:
                group_means.append(np.nan)
                continue

            # Calculate mean across all symptoms in this group for this status
            symptom_means = []
            for symptom in available_symptoms:
                symptom_mean = status_data[symptom].mean()
                if not np.isnan(symptom_mean):
                    symptom_means.append(symptom_mean)

            if symptom_means:
                group_mean = np.mean(symptom_means)
                group_means.append(group_mean)
            else:
                group_means.append(np.nan)

        # Store in heatmap data
        heatmap_data[group_name] = group_means

    # Convert to numeric and calculate z-scores
    heatmap_data = heatmap_data.astype(float)

    # Calculate z-scores for each symptom group (column-wise standardization)
    heatmap_data_z = heatmap_data.copy()
    for col in heatmap_data.columns:
        col_data = heatmap_data[col].dropna()
        if len(col_data) > 1:
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val > 0:
                heatmap_data_z[col] = (heatmap_data[col] - mean_val) / std_val
            else:
                heatmap_data_z[col] = 0

    print("Z-scored intensity matrix:")
    print(heatmap_data_z.round(2))

    # Create the Z-SCORED heat map with colorblind-friendly palette
    plt.figure(figsize=(12, 14))

    sns.heatmap(
        heatmap_data_z,
        annot=True,  # Show values in cells
        fmt='.2f',   # Format to 2 decimal places
        cmap=sns.light_palette('#228833', n_colors=10, as_cmap=True),  # Sequential green
        square=True, # Make cells square
        linewidths=0.5,
        cbar_kws={
            'label': 'Z-scored Symptom Intensity',
            'shrink': 0.8
        },
        annot_kws={'size': 11}
    )

    # Rotate x-axis labels for better readability (APA format)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    sns.despine()

    # Add a subtle grid
    plt.grid(False)  # Remove default grid from heatmap

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    file_name = os.path.join(analysis_obj.output_dir, 'symptom_intensity_heatmap_zscore.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Z-scored symptom intensity heat map saved to: 2_symptoms_model/output/symptom_intensity_heatmap_zscore.png")

    # Create the RAW VALUES heat map
    plt.figure(figsize=(12, 8))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap=sns.light_palette('#228833', n_colors=10, as_cmap=True),
        square=True,
        linewidths=0.5,
        cbar_kws={
            'label': 'Mean Symptom Intensity',
            'shrink': 0.8
        },
        annot_kws={'size': 11}
    )

    plt.title('Symptom Intensity Heat Map Across Menopausal Stages',
            fontsize=14, pad=20)
    plt.xlabel('Symptom Groups', fontsize=12)
    plt.ylabel('Menopausal Stage', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    sns.despine()

    plt.tight_layout()

    # Save the raw values version
    file_name_raw = os.path.join(analysis_obj.output_dir, 'symptom_intensity_heatmap_raw.png')
    plt.savefig(file_name_raw, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Raw values heat map saved to: 2_symptoms_model/output/symptom_intensity_heatmap_raw.png")

    # Return the data for further analysis if needed
    return heatmap_data_z, heatmap_data
