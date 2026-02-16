# KALz Project 1

## 1. Software and Platform
- All packages installed located in `requirements.txt`
> Run `pip install -r requirements.txt` to install all at once
- Used Mac & Windows during development

## 2. Repository Map
```
[Project Folder]/
├── DATA/
│   ├── airport_clean.csv
│   ├── airport.csv
│   └── step2_preprocessed.csv
├── MODELS/
│   ├── .gitkeep
│   ├── linear_model.joblib
│   ├── logistic_model.joblib
│   └── tfidf_vectorizer.joblib
├── OUTPUT/
│   ├── [VARIOUS IMAGES]
│   ├──  ...
│   ├── [VARIOUS IMAGES]
│   ├── [VARIOUS .CSV FILES]
│   ├──  ...
│   └── [VARIOUS .CSV FILES]
├── SCRIPTS/
│   ├── 01_text_preprocess.py
│   ├── 02_logistic_model.py
│   ├── 03_generate_figures.py
│   ├── 04_linear_model.py
│   └── 05_feature_importance.py
├── venv/
├── .gitignore
├── LICENSE.md
├── README.md
└── requirements.txt
```
> [!NOTE]
> The models and venv/ were git ignored due to size; create them using directions below.
> You can delete everything in OUTPUT/ and recreate using the scripts if you would like.

## 3. How to reproduce our results
> [!NOTE]
> Ensure python is set up on your system.
> Run ALL terminal commands from the root directory.
> Use `python3` instead of `python` if commands aren't running.

### Create a virtual environment and install packages
Virtual environments isolate your packages to your current environment \
In your terminal:
- Create environment: `python -m venv venv`
- Activate it:
	- On macOS: `source venv/bin/activate`
	- On Windows: `source venv/Scripts/activate`
### Run python scripts
In your terminal:
- Text preprocessing: `python SCRIPTS/01_text_preprocess.py`
> [!NOTE]
> Delete `preprocessed.csv` before running this (the script creates this)
- Logistic Regression classifier: `python SCRIPTS/02_logistic_model.py`
	- Saves `.csv` files to `OUTPUT/`, model at `DATA/logistic_model.joblib`, and vectorizer at `.DATA/tfidf_vectorizer.joblib`
- Logistic model visual analysis: `python SCRIPTS/03_generate_figures.py`
	- Saves various images to `OUTPUT/`
- Linear Regression predictor: `python SCRIPTS/04_linear_model.py`
	- Saves model at `DATA/linear_model.joblib`
- Linear model visual analysis: `python SCRIPTS/05_feature_importance.py`
	- Saves various images and `.csv` files to `OUTPUT`