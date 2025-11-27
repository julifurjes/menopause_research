"""
Shared data preparation functions used across all menopause analysis models.

This module contains common data preprocessing steps to avoid code duplication
across the three analysis models (stages, symptoms, social support).
"""

import pandas as pd
import numpy as np


def prepare_status_labels(data):
    """
    Filter and label menopausal status categories.

    This function is used by all three models to:
    - Convert STATUS to numeric
    - Filter to relevant status codes (1, 2, 3, 4, 5, 8)
    - Map status codes to descriptive labels
    - Create categorical STATUS_Label with proper ordering
    - Create Menopause_Type (Surgical vs Natural)

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing STATUS column

    Returns
    -------
    pd.DataFrame
        Dataframe with added columns: STATUS_Label, Menopause_Type
    """
    # Convert STATUS to numeric
    data['STATUS'] = pd.to_numeric(data['STATUS'], errors='coerce')

    # Filter to relevant menopausal stages
    data = data[data['STATUS'].isin([1, 2, 3, 4, 5, 8])].copy()

    # Map status codes to labels (consistent across all models)
    status_map = {
        1: 'Surgical',
        2: 'Post-menopause',
        3: 'Late Peri',
        4: 'Early Peri',
        5: 'Pre-menopause',
        8: 'Surgical'
    }
    data['STATUS_Label'] = data['STATUS'].map(status_map)

    # Create Menopause_Type (Surgical vs Natural)
    data['Menopause_Type'] = np.where(
        data['STATUS'].isin([1, 8]),
        'Surgical',
        'Natural'
    )

    # Create ordered categorical for STATUS_Label
    natural_order = ['Pre-menopause', 'Early Peri', 'Late Peri', 'Post-menopause']
    data['STATUS_Label'] = pd.Categorical(
        data['STATUS_Label'],
        categories=['Surgical'] + natural_order,
        ordered=True
    )

    return data


def calculate_baseline_age(data):
    """
    Calculate baseline age for each participant.

    Baseline age is defined as the age at the first visit (VISIT) for each
    participant (SWANID). This is used as a control variable in models.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing SWANID and AGE columns

    Returns
    -------
    pd.DataFrame
        Dataframe with added AGE_BASELINE column
    """
    if 'AGE' in data.columns and 'SWANID' in data.columns:
        data['AGE_BASELINE'] = data.groupby('SWANID')['AGE'].transform('first')

    return data


def sort_by_participant_visit(data):
    """
    Sort data by participant ID and visit number.

    Ensures data is in chronological order for each participant,
    which is important for longitudinal analyses.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing SWANID and VISIT columns

    Returns
    -------
    pd.DataFrame
        Sorted dataframe
    """
    if 'SWANID' in data.columns and 'VISIT' in data.columns:
        data = data.sort_values(['SWANID', 'VISIT'])

    return data


def convert_columns_to_numeric(data, columns):
    """
    Convert specified columns to numeric type.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    columns : list of str
        List of column names to convert to numeric

    Returns
    -------
    pd.DataFrame
        Dataframe with converted columns
    """
    for col in columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data


def apply_common_preprocessing(data, columns_to_convert=None):
    """
    Apply all common preprocessing steps in one function.

    This is a convenience function that applies all common preprocessing:
    1. Convert specified columns to numeric
    2. Prepare status labels
    3. Sort by participant and visit
    4. Calculate baseline age

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    columns_to_convert : list of str, optional
        List of column names to convert to numeric

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe
    """
    # Convert columns to numeric if specified
    if columns_to_convert:
        data = convert_columns_to_numeric(data, columns_to_convert)

    # Apply common preprocessing steps
    data = prepare_status_labels(data)
    data = sort_by_participant_visit(data)
    data = calculate_baseline_age(data)

    return data
