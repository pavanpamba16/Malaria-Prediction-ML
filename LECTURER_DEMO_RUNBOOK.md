# Lecturer Demo Runbook

## 1. Put these files in your GitHub project

- `app.py`
- `requirements-dashboard.txt`
- `Malaria-Data.csv` (already in the repository)

## 2. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dashboard.txt
```

## 3. Start the application

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 4. Recommended lecturer demonstration

1. Overview — explain the problem and ML pipeline.
2. Model Results — show the comparison table and ROC curves.
3. Patient Prediction — enter a sample patient and show the probability.
4. Explainability — show SHAP feature importance and LIME for one patient.
5. Viva Guide — use the questions as your speaking prompts.

## Important academic point

The current notebook reports relatively modest test performance. Do not claim 90%+ accuracy.
The dashboard deliberately reproduces the project's current experiment and makes the
selection criterion explicit.

The notebook uses 337 records, 17 input features, an 80/20 stratified split, StandardScaler,
five baseline models, and a Random Oversampling + Balanced Random Forest experiment.
