import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Cache model loading
@st.cache_resource
def load_model(model_file):
    return joblib.load(model_file)

# Cache data loading
@st.cache_data
def load_data(data_file):
    return pd.read_csv(data_file)

# Sidebar: Dataset/Model selector
st.sidebar.title("Settings")
dataset_choice = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine"])

# Map dataset choice to files and class names
if dataset_choice == "Iris":
    model_file = "iris_model.pkl"
    data_file = "data/iris_dataset.csv"
    class_names = ["Setosa", "Versicolor", "Virginica"]
elif dataset_choice == "Wine":
    model_file = "wine_model.pkl"
    data_file = "data/wine_dataset.csv"
    class_names = ["Class 0", "Class 1", "Class 2"]

# Load model and data
model = load_model(model_file)
df = load_data(data_file)

# Sidebar: Page navigation
pages = ["Data", "Visualisations", "Prediction", "Model Performance"]
page_choice = st.sidebar.radio("Go to", pages)

# --- Page 1: Data ---
if page_choice == "Data":
    st.title(f"{dataset_choice} Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write(df.describe())

# --- Page 2: Visualisations ---
elif page_choice == "Visualisations":
    st.title(f"{dataset_choice} Data Visualisations")
    col = st.selectbox("Select column", df.columns[:-1])  # exclude target
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

# --- Page 3: Prediction ---
elif page_choice == "Prediction":
    st.title(f"{dataset_choice} Prediction")
    
    input_data = []
    for feature in df.columns[:-1]:  # exclude target
        val = st.number_input(feature, value=float(df[feature].mean()))
        input_data.append(val)
    
    if st.button("Predict"):
        pred = model.predict([input_data])[0]
        pred_label = class_names[pred] if pred < len(class_names) else str(pred)
        st.success(f"Predicted Class: **{pred_label}**")
        
        # Prediction probabilities
        probs = model.predict_proba([input_data])[0]
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": (probs * 100).round(2)
        })
        st.table(prob_df)

# --- Page 4: Model Performance ---
elif page_choice == "Model Performance":
    st.title(f"{dataset_choice} Model Performance")

    # Prepare X and y
    X = df.iloc[:, :-1]
    y_true = df.iloc[:, -1]

    # Predictions
    y_pred = model.predict(X)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
