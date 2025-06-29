import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit.components.v1 as components

# ===== 页面设置 =====
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("""
This app uses a trained XGBoost model to classify the overall healthiness of a ready-to-eat food into ten levels (Grades 0.5 to 5.0).  
**Input variables explanation**:
- `Protein`, `Sodium`, `Total fat`, `Energy`: Nutrient values per 100g  
- `weight`: Total package weight (g)  
- `procef_4`: 1 = ultra-processed, 0 = not  
- `ifclaim`: Whether any nutrition/health/other claim exists (1/0)  
- `ifnurclaim`: Whether a nutrition claim is present  
- `nutclaim3`: Specific type of nutrient claim
""")

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    try:
        return joblib.load("XGBoost_final_model_selected_9.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please upload 'XGBoost_final_model_selected_9.pkl'.")
        st.stop()

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler2.pkl")
    except FileNotFoundError:
        st.error("❌ Scaler file not found. Please upload 'scaler2.pkl'.")
        st.stop()

@st.cache_resource
def load_background_data():
    try:
        return np.load("background_data.npy")
    except FileNotFoundError:
        st.error("❌ Background data not found. Please upload 'background_data.npy'.")
        st.stop()

model = load_model()
scaler = load_scaler()
background_data = load_background_data()
explainer = shap.Explainer(model, background_data)

# ===== HSR 风格星图绘制函数 =====
def draw_hsr_star_plot(score):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2)
    ax.axis('off')

    box = plt.Rectangle((0.2, 0.4), 5.6, 1.2, linewidth=2,
                        edgecolor='black', facecolor='white')
    ax.add_patch(box)

    ax.text(3.0, 1.65, "HEALTH STAR RATING", fontsize=14, ha='center', fontweight='bold')

    full_stars = int(score)
    half_star = (score - full_stars) >= 0.5

    for i in range(5):
        x = 0.9 + i
        if i < full_stars:
            ax.text(x, 1.0, '★', fontsize=32, ha='center', va='center', color='black')
        elif i == full_stars and half_star:
            ax.text(x, 1.0, '⯨', fontsize=32, ha='center', va='center', color='black')
        else:
            ax.text(x, 1.0, '☆', fontsize=32, ha='center', va='center', color='gray')

    ax.text(5.45, 1.05, f"{score:.1f}", fontsize=17, fontweight='bold', va='center', ha='left', color='#222')
    plt.tight_layout()
    return fig

# ===== 输入栏 =====
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)
total_fat = st.sidebar.number_input("Total Fat (g/100g)", min_value=0.0, step=0.1)
weight = st.sidebar.number_input("Weight (g)", min_value=0.0, step=1.0)
procef_4 = st.sidebar.selectbox("Ultra-Processed? (procef_4)", [0, 1])
ifclaim = st.sidebar.selectbox("Any Claim Present? (ifclaim)", [0, 1])
ifnurclaim = st.sidebar.selectbox("Nutrition Claim Present? (ifnurclaim)", [0, 1])
nutclaim3 = st.sidebar.selectbox("Specific Nutrient Claim (nutclaim3)", [0, 1])

# ===== 预测逻辑 =====
if st.sidebar.button("🧮 Predict"):
    scaled_columns = ['Sodium', 'Protein', 'Energy', 'Total fat', 'weight',
                      'ifclaim', 'ifnurclaim', 'nutclaim3']
    final_columns = scaled_columns + ['procef_4']

    input_dict = {
        "Sodium": sodium,
        "Protein": protein,
        "Energy": energy,
        "Total fat": total_fat,
        "weight": weight,
        "ifclaim": ifclaim,
        "ifnurclaim": ifnurclaim,
        "nutclaim3": nutclaim3
    }

    user_input_for_scaler = pd.DataFrame([[input_dict[feat] for feat in scaled_columns]], columns=scaled_columns)
    user_scaled_part = scaler.transform(user_input_for_scaler)
    user_scaled_df = pd.DataFrame(user_scaled_part, columns=scaled_columns)
    user_scaled_df["procef_4"] = procef_4
    user_scaled_df = user_scaled_df[final_columns]

    prediction = model.predict(user_scaled_df)[0]
    prob_array = model.predict_proba(user_scaled_df)[0]

    # 使用模型真实类别建立 label_map
    label_map = {cls: round(0.5 + 0.5 * cls, 1) for cls in model.classes_}
    predicted_label = label_map.get(prediction, f"Class {prediction}")

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Prediction:** `{predicted_label}`")
    st.pyplot(draw_hsr_star_plot(predicted_label))

    st.subheader("📊 Probability Table")
    prob_df = pd.DataFrame({
        "HSR Class": [label_map[cls] for cls in model.classes_],
        "Probability": [f"{p:.2f}" for p in prob_array]
    })
    st.dataframe(prob_df, use_container_width=True)

    st.subheader("📈 SHAP Force Plot (All Classes)")
    shap_values = explainer(user_scaled_df)
    sample_index = 0
    for i, cls in enumerate(model.classes_):
        with st.expander(f"🔍 SHAP for Class {label_map[cls]}"):
            force_plot_html = shap.plots.force(
                base_value=explainer.expected_value[i],
                shap_values=shap_values.values[sample_index, :, i],
                features=user_scaled_df.iloc[sample_index],
                matplotlib=False,
                show=False
            )
            components.html(shap.getjs() + force_plot_html.html(), height=400)

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
