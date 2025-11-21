# Longitudinal Data Analysis Project

This repository contains a complete data analysis framework and the corresponding AI declaration.

## Setting Up Your Research Environment

### System Requirements

This project requires Python 3.x and several libraries. A virtual environment can be used to keep all dependencies organized.

### Getting the Repository

Start by downloading the repository to your computer:

```bash
git clone https://github.com/julifurjes/msc_thesis.git
cd msc_thesis
```

### Creating a Virtual Environment

To ensure your analysis runs consistently, create a Python environment:

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Installing Required Libraries

Install all necessary packages using the requirements file:

```bash
pip install -r requirements.txt
```

### Data Requirements and Setup

**Important**: This project requires external data that cannot be included in the repository due to GDPR.

You need to download the dataset from the ICPSR: https://www.icpsr.umich.edu/web/ICPSR/series/00253

After downloading the data:
1. Create a `data` folder in the main project directory
2. Place all downloaded files in this `data` folder

**Required data files** (rename downloaded files to match these names):
- `swan_cross_sectional.tsv` - Cross-sectional dataset
- `swan_timestamp_0.tsv` - Baseline dataset
- `swan_timestamp_1.tsv` - Follow-up visit 1
- `swan_timestamp_2.tsv` - Follow-up visit 2
- `swan_timestamp_3.tsv` - Follow-up visit 3
- `swan_timestamp_4.tsv` - Follow-up visit 4
- `swan_timestamp_5.tsv` - Follow-up visit 5
- `swan_timestamp_6.tsv` - Follow-up visit 6
- `swan_timestamp_7.tsv` - Follow-up visit 7
- `swan_timestamp_8.tsv` - Follow-up visit 8
- `swan_timestamp_9.tsv` - Follow-up visit 9
- `swan_timestamp_10.tsv` - Follow-up visit 10

**Note**: When downloading from ICPSR, files have technical names. Rename them to match the structure above - it's straightforward which file corresponds to which visit, except the "baseline" dataset which should be renamed to `swan_timestamp_0.tsv`.

Your project structure should look like this:

```
project-folder/
├── data/
│   ├── swan_cross_sectional.tsv
│   ├── swan_timestamp_0.tsv
│   ├── swan_timestamp_1.tsv
│   ├── swan_timestamp_2.tsv
│   └── ... (through timestamp_10.tsv)
├── preparations/              # Data processing scripts
├── output/                    # Generated analysis outputs
├── etc.
```

**Output files** (automatically generated):
- `processed_combined_data.csv` - Created by [create_dataframe.py](preparations/create_dataframe.py), used by all models

## Running the Analysis

**IMPORTANT**: All scripts must be run from the main project directory. Do NOT navigate into subdirectories before running scripts, as this will cause import errors.

### Step 1: Data Processing

The analysis requires specific data preparation steps that must be completed in order. Run these scripts from the main directory:

```bash
# First: Create the main data structure
python preparations/create_dataframe.py

# Second: Handle missing data
python preparations/impute_data.py
```

### Step 2: Optional Data Description

After preparing your data, you can run additional analysis (from the main directory):

```bash
# Descriptive statistics
python preparations/data_desc.py

# Validate data quality
python preparations/run_data_validation.py
```

### Step 3: Running the Models

The project includes three different modeling approaches. Each model can be run from the main directory:

```bash
# Model 1: Stages model
python 1_stages_model/analysis.py

# Model 2: Symptoms model
python 2_symptoms_model/analysis.py

# Model 3: Social model
python 3_social_model/analysis.py
```

## AI Declaration

Details about AI assistance in development can be found in the `AI_declaration.txt` file in the main directory.