# KALz Project 1

## 1. Software and Platform
- All packages installed located in `requirements.txt`
> Run `pip install -r requirements.txt` to install all at once
- Used Mac & Windows during development

## 2. Repository Map
```
[Project Folder]/
├── DATA/
│   ├── airport.csv
│   ├── airport_clean.csv
│   └── step2_preprocessed.csv
├── OUTPUT/
│   └── logistic_results_[timestamp].txt
├── SCRIPTS/
│   ├── text_preprocess.py
│   └── logistic_classifier.py
├── .gitignore
├── LICENSE.md
├── README.md
└── requirements.txt
```

## 3. How to reproduce our results
> [!NOTE]
> Run ALL terminal commands from the root directory

### Create a virtual environment and install packages
Virtual environments isolate your packages to your current environment \
In your terminal:
- Create environment: `python -m venv venv`
	- May have to use `python3` instead of `python`
- Activate it:
	- On macOS: `source venv/bin/activate`
	- On Windows: `source venv/Scripts/activate`
### Run python scripts
In your terminal:
- Run preprocessing script: `python SCRIPTS/text_preprocess.py`
> [!NOTE]
> Delete `preprocessed.csv` before running this (the script creates this)
- Run logistic classifer script: `python SCRIPTS/logistic_classifier.py`
	- You will see the output in `OUTPUT/logistic_results_[timestamp].py` with the timestamp of when the script finished running