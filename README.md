# Network Security - Phishing Data Detection

A machine learning pipeline for detecting and classifying phishing network data. This project fetches data from MongoDB, validates it, applies transformations, and trains ML models with experiment tracking.

## Overview

This project implements an end-to-end ML pipeline for phishing classification:
1. **Data Ingestion** - Retrieves phishing data from MongoDB and splits into train/test sets
2. **Data Validation** - Validates data schema and detects data drift
3. **Data Transformation** - Applies preprocessing and feature scaling
4. **Model Training** - Trains and evaluates multiple ML models
5. **Experiment Tracking** - Tracks experiments with MLflow

## Project Structure

```
NetworkSecurity/
├── networksecurity/
│   ├── components/
│   │   ├── data_ingestion.py       # Fetch data from MongoDB
│   │   ├── data_validation.py      # Validate schema and detect drift
│   │   ├── data_transformation.py  # Preprocess and scale features
│   │   └── model_trainer.py        # Train ML models
│   ├── entity/
│   │   ├── config_entity.py        # Configuration classes
│   │   └── artifact_entity.py      # Output artifact definitions
│   ├── exception/
│   │   └── exception.py            # Custom exception handling
│   ├── logging/
│   │   └── logger.py               # Logging setup
│   └── utils/
│       ├── main_utils/
│       │   └── utils.py
│       └── ml_utils/
│           ├── metric/
│           └── model/
├── data_schema/
│   └── schema.yaml                 # Data validation schema
├── Artifacts/                      # Pipeline outputs (timestamped)
├── logs/                           # Application logs
├── final_model/                    # Trained model storage
├── main.py                         # Run the complete pipeline
├── push_data.py                    # Upload data to MongoDB
├── test_mongodb.py                 # Test MongoDB connection
├── requirements.txt                # Dependencies
└── setup.py                        # Package setup
```

## Installation

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Set up MongoDB connection**
   Create a `.env` file in the project root:
   ```
   MONGO_DB_URL=mongodb://<username>:<password>@<host>:<port>/<database>
   ```

## Usage

**Run the complete training pipeline:**
```bash
python main.py
```

This executes all stages:
- Data Ingestion (MongoDB → train/test CSV)
- Data Validation (schema check, drift detection)
- Data Transformation (preprocessing & scaling)
- Model Training (train models, save best model)

**Upload data to MongoDB:**
```bash
python push_data.py
```

**Test MongoDB connection:**
```bash
python test_mongodb.py
```

## Components

**Data Ingestion** (`data_ingestion.py`)
- Connects to MongoDB and retrieves phishing data
- Splits data into train (80%) and test (20%) sets
- Saves CSV files to `Artifacts/data_ingestion/`

**Data Validation** (`data_validation.py`)
- Validates data schema against `data_schema/schema.yaml`
- Generates drift detection report
- Saves validated data to `Artifacts/Data_validation/`

**Data Transformation** (`data_transformation.py`)
- Preprocessing and feature scaling
- Converts data to NumPy arrays
- Saves transformed data to `Artifacts/Data_transformation/`

**Model Trainer** (`model_trainer.py`)
- Trains multiple ML models with GridSearchCV
- Evaluates performance metrics
- Saves best model to `final_model/`
- Tracks experiments in MLflow

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML algorithms
- `pymongo` - MongoDB connection
- `mlflow` - Experiment tracking
- `pyaml` - YAML handling
- `python-dotenv` - Environment variables

See `requirements.txt` for complete list.

## Output

Pipeline outputs are saved in timestamped directories under `Artifacts/`:

```
Artifacts/
└── 04_06_2026_21_32_25/
    ├── data_ingestion/
    │   └── train.csv, test.csv
    ├── Data_validation/
    │   ├── validated/
    │   └── drift_report/
    ├── Data_transformation/
    │   └── transformed/
    └── model_trainer/
        └── trained_model/
```

Trained model is also saved to `final_model/`

## Author

**Kumudini Bugde**  
Email: kumudinibugde@gmail.com

