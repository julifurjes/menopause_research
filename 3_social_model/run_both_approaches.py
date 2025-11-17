"""
Master script to run both approaches for addressing reviewer concern about repeated measures

Approach 1: Stratified Network Analysis by Menopausal Stage
- Creates separate networks for each stage
- Similar to https://www.nature.com/articles/s44294-024-00045-9

Approach 2: Moderation Analysis using Mixed Models
- Tests if social support moderates stage-cognition relationship
- Properly handles repeated measures with random intercepts
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from stratified_network_analysis import StratifiedNetworkAnalysis
from moderation_analysis import ModerationAnalysis

def main():
    """Run both analyses"""

    print("=" * 80)
    print("RUNNING BOTH APPROACHES TO ADDRESS REPEATED MEASURES CONCERN")
    print("=" * 80)

    data_path = "processed_combined_data.csv"

    # Approach 1: Stratified Networks
    print("\n\n" + "=" * 80)
    print("APPROACH 1: STRATIFIED NETWORK ANALYSIS")
    print("=" * 80)
    print("\nCreating separate networks for each menopausal stage...")

    stratified_analysis = StratifiedNetworkAnalysis(data_path)
    stratified_results = stratified_analysis.run_analysis()

    print("\n✓ Stratified network analysis complete!")

    # Approach 2: Moderation Analysis
    print("\n\n" + "=" * 80)
    print("APPROACH 2: MODERATION ANALYSIS")
    print("=" * 80)
    print("\nTesting whether social support moderates the stage-cognition relationship...")

    moderation_analysis = ModerationAnalysis(data_path)
    moderation_results = moderation_analysis.run_analysis()

    print("\n✓ Moderation analysis complete!")

    # Summary
    print("\n\n" + "=" * 80)
    print("BOTH ANALYSES COMPLETE")
    print("=" * 80)
    print("\nApproach 1 (Stratified Networks):")
    print("  - Shows how social support-cognition networks differ across stages")
    print("  - Results: 3_social_model/output/stratified_networks/")
    print("\nApproach 2 (Moderation Analysis):")
    print("  - Tests if social support buffers cognitive decline across transition")
    print("  - Results: 3_social_model/output/moderation/")
    print("\nBoth approaches address the repeated measures concern:")
    print("  - Stratified: Analyzes each stage separately")
    print("  - Moderation: Uses mixed-effects models with random intercepts")
    print("=" * 80)

if __name__ == "__main__":
    main()
