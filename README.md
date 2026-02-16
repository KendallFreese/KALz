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
├── MODELS/
│   ├── .gitkeep
│   ├── logistic_model.joblib
│   └── tfidf_vectorizer.joblib
├── OUTPUT/
│   ├── [VARIOUS IMAGES]
│   ├── 	  ...
│   ├── [VARIOUS IMAGES]
│   └── test_results.csv
├── SCRIPTS/
│   ├── 01_text_preprocess.py
│   ├── 02_train_model.py
│   └── 03_generate_figures.py
├── venv/
├── .gitignore
├── LICENSE.md
├── README.md
└── requirements.txt
```
> [!NOTE]
> The models and venv/ were git ignored due to size; create them using directions below.

## 3. How to reproduce our results
> [!NOTE]
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
- Run preprocessing script: `python SCRIPTS/01_text_preprocess.py`
> [!NOTE]
> Delete `preprocessed.csv` before running this (the script creates this)
- Run logistic classifer script: `python SCRIPTS/02_logistic_classifier.py`
	- You will see the output in `OUTPUT/logistic_results_[timestamp].py` with the timestamp of when the script finished running

 ## 4. References
 [1] J. LaMuraglia, “2025 North America Airport Satisfaction Study,” J.D. Power, Sep. 17, 2025. https://www.jdpower.com/business/press-releases/2025-north-america-airport-satisfaction-study
[2] S. Cheglatonyev, PhD, MSc, “Maximizing Non-Aeronautical Revenues: Key to Airport Financial Sustainability | ACI World Insights,” ACI World Insights, May 08, 2025. https://blog.aci.aero/airport-economics/maximizing-non-aeronautical-revenues-key-to-airport-financial-sustainability/
‌[3] S. Wang, F. Karanki, and Y. Gao, “Spatial spillovers in U.S. airport non-aeronautical revenue performance,” Journal of Transport Geography, vol. 130, p. 104452, Jan. 2026, doi: https://doi.org/10.1016/j.jtrangeo.2025.104452.
[4] D. Jurafsky and J. H. Martin, “Speech and Language Processing,” Stanford.edu, 3rd. ed. draft, Jan. 6, 2026. https://web.stanford.edu/~jurafsky/slp3/
[5] quankiquanki, “GitHub - quankiquanki/skytrax-reviews-dataset: An air travel dataset consisting of user reviews from Skytrax (www.airlinequality.com),” GitHub, 2025. https://github.com/quankiquanki/skytrax-reviews-dataset
[6]  “Top 5 Predictive Analytics Models and Algorithms,” insightsoftware, Jan. 01, 2022. https://insightsoftware.com/blog/top-5-predictive-analytics-models-and-algorithms/
