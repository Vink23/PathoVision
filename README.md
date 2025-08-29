# MRI Brain Tumor Image Multi-Class Classification 
MIDS Summer 2025 \
DATASCI 281 Computer Vision \
Section 1

**Authors**: Richard Yan, Jacqueline Yeung, Vinith Kuruppu, Eduardo Jose Villasenor

## Deliverables
Main Notebook: [2. model_building.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/2.%20model_building.ipynb)
- This is our main notebook, which contains our modeling framework, with references to supplementary notebooks for feature engineering.

Final Report: [Brain Tumor Classification Report.pdf](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/Brain%20Tumor%20Classification%20Report.pdf)

## Structure
### Primary Notebooks
We divided this project into separate notebooks based on individual stages of the pipleine:
- [0. preprocessing_and_eda.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/0.%20preprocessing_and_eda.ipynb)
- [1. feature_exploration.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/1.%20feature_exploration.ipynb)
- [2. model_building.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/2.%20model_building.ipynb) (main notebook, start here)
- [3. results_analysis.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/3.%20results_analysis.ipynb) (see here for results analysis)
### Supplementary Notebooks
These notebooks contain code for feature exploration and engineering.
- Canny Edge Feature Notebook: [canny_edges.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/canny_edges.ipynb)
- Blob Detection Features Notebook: [blob_detection.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/blob_detection.ipynb) (please download/pull to view)
- Complex Features Notebook: [complex_feat_extraction.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/complex_feat_extraction.ipynb) (please download/pull to view)
### Util Functions
Util Python File: [utils.py](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/utils.py)
- We encapsulated all of the main logic for feature generation, model training/testing, results generation, and EDA in this Python file and imported them into the various notebooks.
### Appendix
Ensemble Framework Exploration Notebook: [ensemble_pipeline.ipynb](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/main/ensemble_pipeline.ipynb)
- This notebook contains exploration of the ensemble framework.
