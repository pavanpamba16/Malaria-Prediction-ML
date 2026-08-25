
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from lime.lime_tabular import LimeTabularExplainer
import shap

st.set_page_config(
    page_title="Malaria Severity Prediction",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "age", "sex", "fever", "cold", "rigor", "fatigue", "headace",
    "bitter_tongue", "vomitting", "diarrhea", "Convulsion", "Anemia",
    "jundice", "cocacola_urine", "hypoglycemia", "prostraction",
    "hyperpyrexia"
]
TARGET = "severe_maleria"
CLASS_NAMES = ["Non-Severe Malaria", "Severe Malaria"]

FRIENDLY = {
    "age": "Age",
    "sex": "Sex",
    "fever": "Fever",
    "cold": "Cold",
    "rigor": "Rigor / Chills",
    "fatigue": "Fatigue",
    "headace": "Headache",
    "bitter_tongue": "Bitter Tongue",
    "vomitting": "Vomiting",
    "diarrhea": "Diarrhea",
    "Convulsion": "Convulsion",
    "Anemia": "Anemia",
    "jundice": "Jaundice",
    "cocacola_urine": "Coca-Cola Urine",
    "hypoglycemia": "Hypoglycemia",
    "prostraction": "Prostration",
    "hyperpyrexia": "Hyperpyrexia",
}

@st.cache_data
def load_data():
    df = pd.read_csv("Malaria-Data.csv")
    df = df.drop_duplicates().copy()
    numeric = df.select_dtypes(include=np.number).columns
    for col in numeric:
        df[col] = df[col].fillna(df[col].median())
    return df

@st.cache_resource
def train_pipeline():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss"
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            random_seed=42, verbose=False
        ),
    }

    fitted = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = np.asarray(model.predict(X_test_scaled)).ravel()
        prob = model.predict_proba(X_test_scaled)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        fitted[name] = model

    # Preserve the project's class-imbalance experiment.
    ros = RandomOverSampler(random_state=42)
    X_bal, y_bal = ros.fit_resample(X_train_scaled, y_train)
    balanced_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    balanced_rf.fit(X_bal, y_bal)
    brf_pred = balanced_rf.predict(X_test_scaled)
    brf_prob = balanced_rf.predict_proba(X_test_scaled)[:, 1]
    rows.append({
        "Model": "Balanced Random Forest",
        "Accuracy": accuracy_score(y_test, brf_pred),
        "Precision": precision_score(y_test, brf_pred, zero_division=0),
        "Recall": recall_score(y_test, brf_pred, zero_division=0),
        "F1 Score": f1_score(y_test, brf_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, brf_prob),
        "Balanced Accuracy": balanced_accuracy_score(y_test, brf_pred),
        "MCC": matthews_corrcoef(y_test, brf_pred),
    })
    fitted["Balanced Random Forest"] = balanced_rf

    metrics = pd.DataFrame(rows)

    # For a screening-oriented demo, select the model with the highest recall.
    # This is deliberately stated in the UI rather than calling it universally "best".
    selected_name = metrics.sort_values(
        ["Recall", "ROC-AUC", "F1 Score"], ascending=False
    ).iloc[0]["Model"]

    return {
        "df": df, "X": X, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "scaler": scaler, "models": fitted, "metrics": metrics,
        "selected_name": selected_name,
    }

def make_input():
    vals = {}
    vals["age"] = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
    vals["sex"] = st.selectbox("Sex", ["Female", "Male"])
    vals["sex"] = 0 if vals["sex"] == "Female" else 1

    symptoms = [f for f in FEATURES if f not in ["age", "sex"]]
    left, right = st.columns(2)
    for i, feature in enumerate(symptoms):
        with (left if i % 2 == 0 else right):
            vals[feature] = int(st.checkbox(FRIENDLY[feature], value=False))
    return vals

def input_frame(vals, columns):
    row = {f: 0 for f in columns}
    for f in FEATURES:
        if f in row:
            row[f] = vals[f]
    return pd.DataFrame([row], columns=columns)

def plot_metric_comparison(metrics):
    plot_df = metrics.copy()
    plot_df["Accuracy"] *= 100
    plot_df["F1 Score"] *= 100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(plot_df))
    width = 0.36
    ax.bar(x - width/2, plot_df["Accuracy"], width, label="Accuracy")
    ax.bar(x + width/2, plot_df["F1 Score"], width, label="F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_roc(bundle):
    fig, ax = plt.subplots(figsize=(8, 5))
    y_test = bundle["y_test"]
    for name, model in bundle["models"].items():
        prob = model.predict_proba(bundle["X_test_scaled"])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

st.markdown("""
<style>
.main-title {font-size: 2.35rem; font-weight: 800; margin-bottom: 0.2rem;}
.subtitle {font-size: 1.05rem; color: #64748b; margin-bottom: 1.5rem;}
.card {padding: 1.1rem; border: 1px solid #e2e8f0; border-radius: 14px;
       background: #ffffff; box-shadow: 0 4px 14px rgba(15,23,42,.05);}
.result {padding: 1.3rem; border-radius: 14px; text-align:center;
         border: 1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

bundle = train_pipeline()
metrics = bundle["metrics"]

st.sidebar.title("🦟 Malaria ML")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Patient Prediction", "Model Results", "Explainability", "Viva Guide"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Final-year project demonstration")
st.sidebar.caption("Educational / research use only")

if page == "Overview":
    st.markdown('<div class="main-title">Malaria Severity Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Machine Learning classification with model comparison and Explainable AI</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", len(bundle["df"]))
    c2.metric("Input Features", len(bundle["X"].columns))
    c3.metric("Test Records", len(bundle["X_test"]))
    c4.metric("Models Evaluated", len(bundle["models"]))

    st.markdown("### Project Pipeline")
    st.code(
        "Patient Data → Preprocessing → Train/Test Split → Scaling → "
        "Multiple ML Models → Evaluation → Severity Prediction → LIME / SHAP",
        language="text"
    )

    st.markdown("### Dataset & Methodology")
    st.write(
        "The notebook uses 337 records and 17 input features. "
        "The data is split 80/20 with stratification, the scaler is fitted "
        "only on training data, and Random Oversampling is applied only to "
        "the training set for the imbalance experiment."
    )

    st.markdown("### Model Selection for This Demo")
    selected = bundle["selected_name"]
    selected_row = metrics.loc[metrics["Model"] == selected].iloc[0]
    st.info(
        f"Screening-oriented demo selection: **{selected}** because it has the "
        f"highest severe-case recall in the current experiment "
        f"({selected_row['Recall']:.1%}). This is a project-demo criterion, "
        "not a clinical recommendation."
    )

elif page == "Patient Prediction":
    st.markdown('<div class="main-title">Patient Severity Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter a sample patient profile and demonstrate the end-to-end prediction workflow.</div>', unsafe_allow_html=True)

    vals = make_input()
    selected = bundle["selected_name"]
    model = bundle["models"][selected]

    if st.button("🔍 Predict Malaria Severity", type="primary", use_container_width=True):
        X_new = input_frame(vals, bundle["X"].columns)
        X_new_scaled = bundle["scaler"].transform(X_new)
        pred = int(model.predict(X_new_scaled)[0])
        prob = model.predict_proba(X_new_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.error("### ⚠️ Predicted Class: Severe Malaria")
        else:
            st.success("### ✓ Predicted Class: Non-Severe Malaria")

        a, b = st.columns(2)
        a.metric("Non-Severe Probability", f"{prob[0]*100:.1f}%")
        b.metric("Severe Probability", f"{prob[1]*100:.1f}%")

        chart = pd.DataFrame(
            {"Class": CLASS_NAMES, "Probability": prob * 100}
        ).set_index("Class")
        st.bar_chart(chart)

        st.caption(
            f"Model used: {selected}. The output is a machine-learning prediction "
            "for academic demonstration and must not be treated as a medical diagnosis."
        )

elif page == "Model Results":
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Results reproduced from the project methodology and current notebook pipeline.</div>', unsafe_allow_html=True)

    display_metrics = metrics.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Balanced Accuracy"]:
        display_metrics[col] = (display_metrics[col] * 100).round(2).astype(str) + "%"
    display_metrics["MCC"] = display_metrics["MCC"].round(4)
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    st.pyplot(plot_metric_comparison(metrics), clear_figure=True)
    st.pyplot(plot_roc(bundle), clear_figure=True)

    selected = bundle["selected_name"]
    st.markdown(f"### Confusion Matrix — {selected}")
    model = bundle["models"][selected]
    pred = model.predict(bundle["X_test_scaled"])
    cm = confusion_matrix(bundle["y_test"], pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Non-Severe", "Severe"],
        yticklabels=["Non-Severe", "Severe"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {selected}")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.warning(
        "Important observation for your viva: the current notebook results are modest, "
        "especially for severe-case recall. Present this honestly as a limitation and "
        "explain that the project demonstrates the full ML + explainability workflow."
    )

elif page == "Explainability":
    st.markdown('<div class="main-title">Explainable AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Show lecturers not only what the model predicts, but why.</div>', unsafe_allow_html=True)

    model_name = st.selectbox("Model to explain", list(bundle["models"].keys()), index=0)
    model = bundle["models"][model_name]

    tab1, tab2 = st.tabs(["SHAP — Global Importance", "LIME — Individual Prediction"])

    with tab1:
        st.write(
            "SHAP estimates each feature's contribution to the model output. "
            "For this demonstration, we visualize feature importance for the selected tree model."
        )
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(bundle["X_test_scaled"])
            if isinstance(shap_values, list):
                vals = shap_values[1]
            else:
                arr = np.asarray(shap_values)
                vals = arr[:, :, 1] if arr.ndim == 3 else arr
            importance = np.abs(vals).mean(axis=0)
            imp = pd.DataFrame({
                "Feature": bundle["X_test_scaled"].columns,
                "Mean |SHAP|": importance
            }).sort_values("Mean |SHAP|", ascending=True).tail(12)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(imp["Feature"], imp["Mean |SHAP|"])
            ax.set_xlabel("Mean absolute SHAP value")
            ax.set_title(f"Top Feature Contributions — {model_name}")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"SHAP visualization could not be generated for {model_name}: {e}")

    with tab2:
        patient_index = st.slider(
            "Test patient index",
            min_value=0,
            max_value=len(bundle["X_test_scaled"]) - 1,
            value=0
        )
        patient = bundle["X_test_scaled"].iloc[patient_index].values
        actual = int(bundle["y_test"].iloc[patient_index])
        prediction = int(model.predict(patient.reshape(1, -1))[0])
        probability = model.predict_proba(patient.reshape(1, -1))[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual", CLASS_NAMES[actual])
        c2.metric("Predicted", CLASS_NAMES[prediction])
        c3.metric("Severe Probability", f"{probability[1]*100:.1f}%")

        try:
            lime_explainer = LimeTabularExplainer(
                training_data=bundle["X_train_scaled"].values,
                feature_names=bundle["X_train_scaled"].columns.tolist(),
                class_names=CLASS_NAMES,
                mode="classification",
                discretize_continuous=True,
                random_state=42
            )
            explanation = lime_explainer.explain_instance(
                patient, model.predict_proba, num_features=10
            )
            exp_df = pd.DataFrame(
                explanation.as_list(label=prediction),
                columns=["Feature condition", "Contribution"]
            ).sort_values("Contribution")
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(exp_df["Feature condition"], exp_df["Contribution"])
            ax.axvline(0, linewidth=1)
            ax.set_title("LIME Local Explanation")
            ax.set_xlabel("Contribution to selected class")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"LIME explanation could not be generated: {e}")

elif page == "Viva Guide":
    st.markdown('<div class="main-title">Lecturer / Viva Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Use this page as your speaking sequence during the demonstration.</div>', unsafe_allow_html=True)

    questions = [
        ("What is the problem?", "Predict whether a patient's malaria case is non-severe or severe using clinical features."),
        ("Why multiple models?", "To compare different classification approaches instead of relying on a single algorithm."),
        ("Why train/test split?", "To evaluate the model on unseen data and reduce the risk of reporting training performance as generalization."),
        ("Why scaling?", "The project standardizes the input features and applies the same fitted scaler to test data."),
        ("Why class imbalance handling?", "The training set contains fewer severe cases, so Random Oversampling was tested only on training data."),
        ("Why Recall matters?", "For severe-case screening, missing a true severe case is important, so recall is a useful metric to discuss."),
        ("Why LIME?", "LIME gives a local explanation for one individual prediction."),
        ("Why SHAP?", "SHAP provides feature-contribution explanations and can be used for global importance."),
        ("Is this a medical diagnosis?", "No. It is an academic ML demonstration and should not replace clinical assessment or diagnostic testing."),
    ]
    for q, a in questions:
        with st.expander(q):
            st.write(a)

    st.markdown("### 60-second opening")
    st.write(
        "“My project develops a machine-learning based malaria severity prediction "
        "system. I preprocess the clinical dataset, compare several classification "
        "models, evaluate them using multiple metrics, investigate class imbalance, "
        "and use Explainable AI techniques such as LIME and SHAP. The final interface "
        "allows a lecturer to enter a sample patient profile, see the predicted severity "
        "and probability, and then inspect the factors contributing to the prediction.”"
    )
