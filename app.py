import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
DATA_PATH = "winequality_combined.csv"  # replace with your path
REMOVE_OUTLIERS = True
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- Helper Functions ----------
def safe_load_data(path):
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, numeric_cols

def remove_outliers_iqr(df, numeric_cols):
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df[numeric_cols] < (Q1 - 1.5*IQR)) | (df[numeric_cols] > (Q3 + 1.5*IQR))).any(axis=1)]
    return df_clean

# ---------- Load & Preprocess ----------
df, numeric_cols = safe_load_data(DATA_PATH)
target_col = numeric_cols[-1]           # assume last column = target
feature_cols = numeric_cols[:-1]

if REMOVE_OUTLIERS:
    df = remove_outliers_iqr(df, numeric_cols)

X = df[feature_cols].values.astype(float)
y = df[target_col].values.astype(int)  # classification target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality with PCA
pca = PCA(n_components=min(5, X_scaled.shape[1]))
X_pca = pca.fit_transform(X_scaled)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ---------- Classification Model ----------
model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ---------- Gradio Function ----------
def predict_class(*user_inputs):
    user_arr = np.array([user_inputs], dtype=float)
    user_scaled = scaler.transform(user_arr)
    user_pca = pca.transform(user_scaled)
    pred = model.predict(user_pca)[0]
    return int(pred), f"Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}"

# ---------- Build Gradio App ----------
with gr.Blocks() as demo:
    gr.Markdown("## 🍷 Wine Quality Prediction (Classification + PCA)")
    gr.Markdown("### Enter Numeric Features:")
    input_fields = [gr.Number(label=col) for col in feature_cols]

    predict_btn = gr.Button("Predict Quality")
    pred_out = gr.Textbox(label="Predicted Quality")
    metrics_out = gr.Textbox(label="Model Metrics")

    predict_btn.click(fn=predict_class, inputs=input_fields, outputs=[pred_out, metrics_out])

# ---------- Launch ----------
if __name__ == "__main__":
    demo.launch(share=True)
