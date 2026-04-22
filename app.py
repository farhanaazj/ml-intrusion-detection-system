# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IDS · Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — compact, dark cyber-terminal aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@600;700;800&display=swap');

:root {
    --bg: #080c18;
    --card: #0e1520;
    --card2: #111d2e;
    --border: #1a2a3a;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --accent3: #7c3aed;
    --success: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #ffffff;
    --muted: #ffffff;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg);
    color: var(--text);
    font-size: 12px;
}
.stApp { background-color: var(--bg); }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--card);
    border-radius: 8px 8px 0 0;
    border: 1px solid var(--border);
    padding: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 6px 16px;
    color: var(--muted);
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#00d4ff18,#7c3aed18) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── METRICS ── */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 8px 12px;
}
div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.1rem !important;
    font-weight: 700;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg,#00d4ff,#0099bb);
    color: #000 !important;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
    width: 100%;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── DATAFRAME ── */
.stDataFrame { font-size: 0.68rem !important; }
.stDataFrame table { border-collapse: collapse; }
.stDataFrame th {
    background: var(--card2) !important;
    color: var(--accent) !important;
    font-size: 0.65rem !important;
    padding: 4px 8px !important;
}
.stDataFrame td { padding: 3px 8px !important; font-size: 0.67rem !important; }

/* ── SECTION HEADERS ── */
.sh {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
    margin: 18px 0 10px;
}

/* ── HERO ── */
.hero {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg,#00d4ff,#7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 2px;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── BADGE ── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.62rem;
    font-weight: 700;
    background: rgba(0,212,255,0.1);
    color: var(--accent);
    border: 1px solid rgba(0,212,255,0.25);
}
.badge-g { background:rgba(16,185,129,0.1); color:#10b981; border-color:rgba(16,185,129,0.25); }
.badge-r { background:rgba(239,68,68,0.1); color:#ef4444; border-color:rgba(239,68,68,0.25); }
.badge-o { background:rgba(255,107,53,0.1); color:#ff6b35; border-color:rgba(255,107,53,0.25); }

/* ── PREDICTION BOX ── */
.pred-box {
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 18px 22px;
    text-align: center;
    margin-top: 10px;
}
.pred-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.pred-conf {
    font-size: 0.7rem;
    color: var(--muted);
}

/* ── SELECTBOX / SLIDER ── */
.stSelectbox label, .stSlider label, .stFileUploader label {
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── CODE BLOCK ── */
.stCodeBlock { font-size: 0.65rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0e1520',
    'axes.facecolor': '#0e1520',
    'axes.edgecolor': '#1a2a3a',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    'text.color': '#ffffff',
    'grid.color': '#1a2a3a',
    'grid.alpha': 0.7,
    'font.family': 'monospace',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

PAL = ['#00d4ff', '#ff6b35', '#7c3aed', '#10b981', '#f59e0b', '#ec4899']

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for k in ['trained_models', 'results', 'X_test', 'y_test',
          'feature_names', 'df_raw', 'X', 'y', 'scaler',
          'tuned_model', 'best_params', 'base_metrics', 'tuned_metrics',
          'models_trained', 'X_train', 'y_train']:
    if k not in st.session_state:
        st.session_state[k] = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)
    drop_cols = [c for c in df.columns if c.lower() in ['time', 'timestamp', 'date']]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col.lower() != 'label':
            df[col] = le.fit_transform(df[col].astype(str))
    target_col = 'label' if 'label' in df.columns else df.columns[-1]
    df.dropna(inplace=True)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == object:
        y = pd.Series(le.fit_transform(y), name=target_col)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, df, X.columns.tolist(), scaler

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
    return models

def tune_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2'],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    cm = confusion_matrix(y_test, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'F1 Score':  f1_score(y_test, y_pred, average='weighted'),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'FPR':       fp / (tn + fp + 1e-9),
    }, y_pred, y_prob

# ─────────────────────────────────────────────
# PLOT FUNCTIONS  (compact figsize)
# ─────────────────────────────────────────────
def fig_class_dist(y):
    vals = y.value_counts()
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    wedges, texts, at = ax.pie(
        vals.values, labels=[f'Class {l}' for l in vals.index],
        autopct='%1.1f%%', colors=PAL[:len(vals)],
        wedgeprops={'edgecolor': '#080c18', 'linewidth': 1.5},
        textprops={'color': '#cbd5e1', 'fontsize': 7}
    )
    for a in at:
        a.set_color('#080c18'); a.set_fontsize(7); a.set_fontweight('bold')
    ax.set_title('Class Distribution', fontsize=9, color='#00d4ff')
    plt.tight_layout(pad=0.5)
    return fig

def fig_corr(df, top_n=9):
    num = df.select_dtypes(include=np.number)
    corr = num.corr()
    if 'label' in corr.columns:
        top = corr['label'].abs().nlargest(top_n + 1).index.tolist()
        corr = corr.loc[top, top]
    fig, ax = plt.subplots(figsize=(5, 4))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                linewidths=0.3, linecolor='#080c18',
                annot=True, fmt='.2f',
                annot_kws={'fontsize': 6, 'color': 'white'}, ax=ax)
    ax.set_title('Correlation Matrix (Top Features)', fontsize=9, color='#00d4ff')
    plt.tight_layout(pad=0.4)
    return fig

def fig_boxplot(df, feature_names):
    # EDA extra: boxplot of top 6 numeric features by label
    cols = [c for c in feature_names if c in df.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.5))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        groups = [df[df['label'] == lbl][col].dropna().values
                  for lbl in sorted(df['label'].unique())]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.45,
                        medianprops={'color': '#080c18', 'linewidth': 1.5},
                        whiskerprops={'color': '#475569'},
                        capprops={'color': '#475569'},
                        flierprops={'marker': 'o', 'markersize': 2,
                                    'markerfacecolor': '#ff6b35', 'alpha': 0.5})
        for patch, color in zip(bp['boxes'], PAL):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(col[:14], fontsize=7, color='#94a3b8')
        ax.set_xticklabels([f'C{lbl}' for lbl in sorted(df['label'].unique())], fontsize=6)
        ax.grid(True, axis='y', alpha=0.3)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Feature Distribution by Class (Boxplots)', fontsize=9,
                 color='#00d4ff', y=1.01)
    plt.tight_layout(pad=0.4)
    return fig

def fig_metric_bar(results_df):
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC']
    df = results_df[metrics].T
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(len(metrics))
    w = 0.22
    for i, (col, color) in enumerate(zip(df.columns, PAL)):
        bars = ax.bar(x + i * w, df[col].values, w, label=col,
                      color=color, alpha=0.85, edgecolor='#080c18', linewidth=0.4)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6, color='#94a3b8')
    ax.set_xticks(x + w)
    ax.set_xticklabels(metrics, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_title('Model Performance Comparison', fontsize=9, color='#00d4ff')
    ax.legend(fontsize=6.5, facecolor='#0e1520', edgecolor='#1a2a3a', ncol=3)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(pad=0.4)
    return fig

def fig_fpr(results_df):
    fig, ax = plt.subplots(figsize=(4, 2.8))
    models = results_df.index.tolist()
    fprs = results_df['FPR'].tolist()
    bars = ax.bar(models, fprs, color=PAL[:len(models)],
                  edgecolor='#080c18', linewidth=0.4, width=0.5)
    for bar, v in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=6.5, color='#94a3b8')
    ax.set_title('False Positive Rate', fontsize=9, color='#ff6b35')
    ax.set_ylabel('FPR', fontsize=7)
    ax.set_ylim(0, max(fprs + [0.01]) * 1.35)
    ax.set_xticklabels(models, fontsize=6.5, rotation=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(pad=0.4)
    return fig

def fig_roc(models_dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot([0, 1], [0, 1], '--', color='#334155', lw=1)
    for i, (name, model) in enumerate(models_dict.items()):
        if not hasattr(model, 'predict_proba'):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=PAL[i], lw=1.5,
                label=f'{name[:12]} AUC={auc:.3f}')
    ax.set_xlabel('FPR', fontsize=7)
    ax.set_ylabel('TPR', fontsize=7)
    ax.set_title('ROC Curves', fontsize=9, color='#00d4ff')
    ax.legend(fontsize=6.5, facecolor='#0e1520', edgecolor='#1a2a3a')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.4)
    return fig

def fig_cm(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3, 2.5))
    heat = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                       linewidths=0.5, linecolor='#080c18',
                       ax=ax, cbar=False)

    # Draw text with high-contrast color per cell for best readability
    max_val = cm.max() if cm.size > 0 else 0
    threshold = max_val / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = 'white' if value > threshold else 'black'
            ax.text(j + 0.5, i + 0.5, f'{value:d}',
                    ha='center', va='center', fontsize=10,
                    color=text_color, fontfamily='monospace', fontweight='bold')

    ax.set_title(name[:20], fontsize=8, color='#00d4ff', pad=6)
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_ylabel('Actual', fontsize=7)
    plt.tight_layout(pad=0.4)
    return fig

def fig_feature_imp(model, feature_names, top_n=12):
    if not hasattr(model, 'feature_importances_'):
        return None
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(5, max(3, top_n * 0.28)))
    colors = [PAL[0] if v > np.median(imp[idx]) else PAL[2] for v in imp[idx]]
    ax.barh([feature_names[i][:18] for i in idx], imp[idx],
            color=colors, edgecolor='#080c18', linewidth=0.3)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=9, color='#00d4ff')
    ax.set_xlabel('Importance', fontsize=7)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout(pad=0.4)
    return fig

def fig_tune_compare(base, tuned):
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC']
    bv = [base.get(m) or 0 for m in metrics]
    tv = [tuned.get(m) or 0 for m in metrics]
    x = np.arange(len(metrics))
    w = 0.33
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(x - w / 2, bv, w, label='Base RF', color=PAL[1], alpha=0.85)
    ax.bar(x + w / 2, tv, w, label='Tuned RF', color=PAL[0], alpha=0.85)
    for i, (b, t) in enumerate(zip(bv, tv)):
        ax.text(i - w / 2, b + 0.005, f'{b:.3f}', ha='center', fontsize=6, color='#94a3b8')
        ax.text(i + w / 2, t + 0.005, f'{t:.3f}', ha='center', fontsize=6, color='#94a3b8')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_title('Base vs Tuned Random Forest', fontsize=9, color='#00d4ff')
    ax.legend(fontsize=7, facecolor='#0e1520')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(pad=0.4)
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero">🛡️ IDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Network Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload network_traffic.csv", type=['csv'])
    test_size = st.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05)
    run_tuning = st.checkbox("Hyperparameter Tuning", value=True)
    run_btn = st.button("▶  Run Analysis")

    st.markdown("---")
    st.markdown("""<div style='font-size:0.65rem;color:#334155;line-height:1.7'>
    <b style='color:#475569'>Models</b><br>
    · Logistic Regression<br>· Random Forest<br>· Gradient Boosting<br><br>
    <b style='color:#475569'>Tuning</b><br>· GridSearchCV · 3-fold CV
    </div>""", unsafe_allow_html=True)

    if st.session_state.models_trained:
        st.markdown("---")
        st.markdown('<div style="font-size:0.65rem;color:#10b981">✓ Models trained &amp; ready</div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero">ML-Based Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Network Traffic · Anomaly Detection · Multi-Model Evaluation</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GATE: need file
# ─────────────────────────────────────────────
if not uploaded_file:
    st.info("⬅️  Upload **network_traffic.csv** in the sidebar and click **▶ Run Analysis**.")
    st.stop()

# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("Preprocessing data…"):
        X, y, df_raw, feature_names, scaler = load_and_preprocess(uploaded_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        st.session_state.X = X
        st.session_state.y = y
        st.session_state.df_raw = df_raw
        st.session_state.feature_names = feature_names
        st.session_state.scaler = scaler
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    with st.spinner("Training models…"):
        trained = train_models(X_train, y_train)
        st.session_state.trained_models = trained

    results = {}
    for name, model in trained.items():
        m, _, _ = evaluate(model, X_test, y_test)
        results[name] = m
    st.session_state.results = results

    if run_tuning:
        with st.spinner("Tuning best model (GridSearchCV)…"):
            best_name = pd.DataFrame(results).T['F1 Score'].idxmax()
            base_m = results[best_name]
            tuned_model, best_params = tune_rf(X_train, y_train)
            tuned_m, _, _ = evaluate(tuned_model, X_test, y_test)
            st.session_state.tuned_model = tuned_model
            st.session_state.best_params = best_params
            st.session_state.base_metrics = base_m
            st.session_state.tuned_metrics = tuned_m

    st.session_state.models_trained = True
    st.success("✓ Analysis complete — explore the tabs below.")

# ─────────────────────────────────────────────
# GUARD: models not yet trained
# ─────────────────────────────────────────────
if not st.session_state.models_trained:
    st.info("Configure settings and click **▶ Run Analysis** to begin.")
    st.stop()

# ─────────────────────────────────────────────
# PULL STATE
# ─────────────────────────────────────────────
df_raw       = st.session_state.df_raw
X            = st.session_state.X
y            = st.session_state.y
feature_names = st.session_state.feature_names
X_train      = st.session_state.X_train
X_test       = st.session_state.X_test
y_train      = st.session_state.y_train
y_test       = st.session_state.y_test
trained      = st.session_state.trained_models
results      = st.session_state.results
scaler       = st.session_state.scaler
results_df   = pd.DataFrame(results).T.apply(pd.to_numeric, errors='coerce')
best_model_name = results_df['F1 Score'].idxmax()
best_model   = trained[best_model_name]

# ═════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  Upload & EDA",
    "🤖  Model Training & Evaluation",
    "⚡  Live Prediction Dashboard"
])

# ─────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────
with tab1:
    # Dataset metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df_raw):,}")
    c2.metric("Features", len(feature_names))
    c3.metric("Train", f"{len(X_train):,}")
    c4.metric("Test", f"{len(X_test):,}")
    c5.metric("Classes", y.nunique())

    st.markdown('<div class="sh">📋 Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(6), use_container_width=True, height=180)

    st.markdown('<div class="sh">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe().round(3), use_container_width=True, height=180)

    # Row 1: class dist + corr
    col_a, col_b = st.columns([1, 1.8])
    with col_a:
        st.markdown('<div class="sh">🎯 Class Distribution</div>', unsafe_allow_html=True)
        st.pyplot(fig_class_dist(y), use_container_width=True)
        plt.close()
    with col_b:
        st.markdown('<div class="sh">🔗 Correlation Matrix</div>', unsafe_allow_html=True)
        st.pyplot(fig_corr(df_raw), use_container_width=True)
        plt.close()

    # EDA extra — Boxplots per class
    st.markdown('<div class="sh">📦 Feature Distribution by Class</div>', unsafe_allow_html=True)
    st.pyplot(fig_boxplot(df_raw, feature_names), use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────
# TAB 2 — MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sh">📊 Evaluation Metrics</div>', unsafe_allow_html=True)

    display_cols = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC', 'FPR']
    styled = results_df[display_cols].style\
        .format("{:.4f}")\
        .background_gradient(cmap='Blues', subset=['Accuracy', 'F1 Score', 'ROC-AUC'])\
        .background_gradient(cmap='Reds_r', subset=['FPR'])
    st.dataframe(styled, use_container_width=True, height=140)

    st.markdown(
        f'<div style="margin:6px 0 12px">Best model: '
        f'<span class="badge">{best_model_name}</span> '
        f'<span style="color:#475569;font-size:0.65rem"> · F1 = '
        f'<span style="color:#00d4ff">{results_df.loc[best_model_name,"F1 Score"]:.4f}</span></span></div>',
        unsafe_allow_html=True
    )

    # Charts row 1
    col_a, col_b = st.columns([1.6, 1])
    with col_a:
        st.markdown('<div class="sh">📈 Metric Comparison</div>', unsafe_allow_html=True)
        st.pyplot(fig_metric_bar(results_df), use_container_width=True)
        plt.close()
    with col_b:
        st.markdown('<div class="sh">⚠️ False Positive Rate</div>', unsafe_allow_html=True)
        st.pyplot(fig_fpr(results_df), use_container_width=True)
        plt.close()

    # ROC + Feature Importance
    col_c, col_d = st.columns([1, 1.2])
    with col_c:
        st.markdown('<div class="sh">📉 ROC Curves</div>', unsafe_allow_html=True)
        st.pyplot(fig_roc(trained, X_test, y_test), use_container_width=True)
        plt.close()
    with col_d:
        st.markdown('<div class="sh">🔍 Feature Importance</div>', unsafe_allow_html=True)
        fi = fig_feature_imp(best_model, feature_names)
        if fi:
            st.pyplot(fi, use_container_width=True)
            plt.close()
        else:
            st.info("Feature importance not available for this model type.")

    # Confusion Matrices
    st.markdown('<div class="sh">🔢 Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for col, (name, model) in zip(cm_cols, trained.items()):
        _, y_pred, _ = evaluate(model, X_test, y_test)
        with col:
            st.markdown(f'<div style="text-align:center"><span class="badge">{name}</span></div>',
                        unsafe_allow_html=True)
            st.pyplot(fig_cm(y_test, y_pred, name), use_container_width=True)
            plt.close()

    # Classification Report
    st.markdown('<div class="sh">📝 Classification Report — Best Model</div>', unsafe_allow_html=True)
    _, y_pred_best, _ = evaluate(best_model, X_test, y_test)
    st.code(classification_report(y_test, y_pred_best), language='text')

    # ── TUNING ──────────────────────────────
    if st.session_state.tuned_model is not None:
        st.markdown('<div class="sh">⚡ Hyperparameter Tuning Results</div>', unsafe_allow_html=True)

        bp = st.session_state.best_params
        p_cols = st.columns(len(bp))
        for col, (k, v) in zip(p_cols, bp.items()):
            col.metric(k, str(v))

        col_t1, col_t2 = st.columns([1.4, 1])
        with col_t1:
            st.pyplot(
                fig_tune_compare(st.session_state.base_metrics,
                                 st.session_state.tuned_metrics),
                use_container_width=True
            )
            plt.close()
        with col_t2:
            # Delta table
            st.markdown('<div class="sh">Δ Performance Delta</div>', unsafe_allow_html=True)
            delta_rows = []
            for m in display_cols:
                b = st.session_state.base_metrics.get(m) or 0
                t = st.session_state.tuned_metrics.get(m) or 0
                delta_rows.append({'Metric': m,
                                   'Base RF': f'{b:.4f}',
                                   'Tuned RF': f'{t:.4f}',
                                   'Δ': f'{t-b:+.4f}'})
            st.dataframe(pd.DataFrame(delta_rows).set_index('Metric'),
                         use_container_width=True, height=220)

        # Tuned feature importance
        fi_tuned = fig_feature_imp(st.session_state.tuned_model, feature_names)
        if fi_tuned:
            st.markdown('<div class="sh">🔍 Tuned Model Feature Importance</div>',
                        unsafe_allow_html=True)
            st.pyplot(fi_tuned, use_container_width=True)
            plt.close()

# ─────────────────────────────────────────────
# TAB 3 — LIVE PREDICTION DASHBOARD
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sh">⚡ Live Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""<div style='font-size:0.68rem;color:#475569;margin-bottom:12px'>
    Select a record from the dataset — all three models predict in real time.
    </div>""", unsafe_allow_html=True)

    # Build display labels for the dropdown
    raw_df = df_raw.reset_index(drop=True)
    row_labels = [
        f"Row {i:04d}  |  proto={int(raw_df.loc[i,'protocol']) if 'protocol' in raw_df.columns else '?'}"
        f"  |  pkts={int(raw_df.loc[i,'packet_count']) if 'packet_count' in raw_df.columns else '?'}"
        f"  |  label={int(raw_df.loc[i,'label'])}"
        for i in range(min(len(raw_df), 300))
    ]

    sel_label = st.selectbox("Select a network traffic record", row_labels)
    sel_idx = int(sel_label.split("  |  ")[0].replace("Row ", ""))

    # Build feature vector
    row_raw = raw_df.iloc[[sel_idx]][feature_names]
    row_scaled = pd.DataFrame(scaler.transform(row_raw), columns=feature_names)

    # Ground truth
    true_label = int(raw_df.iloc[sel_idx]['label'])

    # Predict with all models (+tuned if available)
    predict_models = dict(trained)
    if st.session_state.tuned_model is not None:
        predict_models["Tuned RF"] = st.session_state.tuned_model

    st.markdown('<div class="sh">🔎 Selected Record Features</div>', unsafe_allow_html=True)
    st.dataframe(row_raw.T.rename(columns={row_raw.index[0]: 'Value'}).round(4),
                 use_container_width=True, height=220)

    st.markdown('<div class="sh">🤖 Model Predictions</div>', unsafe_allow_html=True)

    pred_cols = st.columns(len(predict_models))
    for col, (name, model) in zip(pred_cols, predict_models.items()):
        pred = int(model.predict(row_scaled)[0])
        prob = model.predict_proba(row_scaled)[0] if hasattr(model, 'predict_proba') else None
        conf = max(prob) if prob is not None else None
        is_attack = pred == 1
        correct = pred == true_label

        color = '#ef4444' if is_attack else '#10b981'
        label_txt = '🚨 ATTACK' if is_attack else '✅ NORMAL'
        badge_cls = 'badge-r' if is_attack else 'badge-g'
        match_txt = '✓ Correct' if correct else '✗ Wrong'
        match_color = '#10b981' if correct else '#ef4444'

        with col:
            st.markdown(f"""
            <div class="pred-box" style="border-color:{color}22;background:linear-gradient(135deg,{color}08,{color}04)">
              <div style="font-size:0.62rem;color:#475569;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.1em">{name}</div>
              <div class="pred-val" style="color:{color}">{label_txt}</div>
              {'<div class="pred-conf">Confidence: <b>' + f'{conf:.1%}' + '</b></div>' if conf is not None else ''}
              <div style="margin-top:8px;font-size:0.65rem;color:{match_color}">{match_txt}</div>
            </div>
            """, unsafe_allow_html=True)

    # Ground truth callout
    gt_color = '#ef4444' if true_label == 1 else '#10b981'
    gt_txt = '🚨 ATTACK' if true_label == 1 else '✅ NORMAL'
    st.markdown(f"""
    <div style="margin-top:14px;padding:10px 16px;border-radius:8px;
                border:1px solid {gt_color}33;background:{gt_color}0a;
                font-size:0.72rem;display:flex;align-items:center;gap:12px">
      <span style="color:#475569;text-transform:uppercase;letter-spacing:0.1em;font-size:0.62rem">Ground Truth</span>
      <span style="color:{gt_color};font-weight:700;font-family:'Syne',sans-serif">{gt_txt}</span>
      <span style="color:#334155;font-size:0.6rem">· Class label = {true_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    if any(hasattr(m, 'predict_proba') for m in predict_models.values()):
        st.markdown('<div class="sh">📊 Prediction Probability Breakdown</div>', unsafe_allow_html=True)
        prob_data = {}
        for name, model in predict_models.items():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(row_scaled)[0]
                for i, p in enumerate(probs):
                    prob_data.setdefault(f'Class {i}', {})[name] = p

        fig_prob, ax = plt.subplots(figsize=(6, 2.4))
        classes = list(prob_data.keys())
        model_names = list(predict_models.keys())
        x = np.arange(len(model_names))
        w = 0.35 / max(len(classes), 1)
        for j, cls in enumerate(classes):
            vals = [prob_data[cls].get(n, 0) for n in model_names]
            color = '#ef4444' if '1' in cls else '#10b981'
            bars = ax.bar(x + j * w - (len(classes) - 1) * w / 2,
                          vals, w, label=cls, color=color, alpha=0.8,
                          edgecolor='#080c18', linewidth=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f'{v:.2f}', ha='center', fontsize=6.5, color='#94a3b8')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=7)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel('Probability', fontsize=7)
        ax.set_title('Class Probabilities per Model', fontsize=9, color='#00d4ff')
        ax.legend(fontsize=7, facecolor='#0e1520', edgecolor='#1a2a3a')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig_prob, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#1e2d40;font-size:0.6rem;font-family:monospace'>
ML-Based IDS · Anomaly Detection · Academic Research Dashboard
</div>""", unsafe_allow_html=True)