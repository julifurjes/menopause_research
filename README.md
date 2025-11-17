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
2. Place all downloaded files in this `data` folder and follow the naming regulations in the code

Your project structure should look like this:

```
project-folder/
├── data/           # Your downloaded ICPSR data goes here
├── preparations/   # Data processing scripts
├── etc.
```

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
python 1_stages_model/longitudinal.py

# Model 2: Symptoms model
python 2_symptoms_model/longitudinal.py

# Model 3: Social model
python 3_social_model/longitudinal.py
```

## AI Declaration

Details about AI assistance in development can be found in the `AI_declaration.txt` file in the main directory.