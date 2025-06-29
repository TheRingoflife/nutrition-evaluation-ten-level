# 修改说明：将原五分类代码修改为十分类版本，标签为0.5到5.0（步长0.5），并用HSR星图代替Nutri-score图

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ===== 页面设置 =====
st.set_page_config(page_title="HSR Classifier (10-Level)", layout="wide")
st.title("🌟 Predicting Health Star Rating (0.5 - 5.0)")

# ===== 加载资源 =====
@st.cache_resource
def load_model():
    return joblib.load("XGBoost_HSR_10class.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler2.pkl")

@st.cache_resource
def load_background_data():
    return np.load("background_data_10class.npy")

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

    # 背景框
    box = plt.Rectangle((0.2, 0.4), 5.6, 1.2, linewidth=2,
                        edgecolor='black', facecolor='white')
    ax.add_patch(box)

    # 标签
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

    input_dict = {feat: eval(feat.replace(' ', '_')) for feat in scaled_columns}
    user_input_for_scaler = pd.DataFrame([[input_dict[feat] for feat in scaled_columns]], columns=scaled_columns)
    user_scaled_part = scaler.transform(user_input_for_scaler)
    user_scaled_df = pd.DataFrame(user_scaled_part, columns=scaled_columns)
    user_scaled_df['procef_4'] = procef_4
    user_scaled_df = user_scaled_df[final_columns]

    prediction = model.predict(user_scaled_df)[0]  # 0–9
    prob_array = model.predict_proba(user_scaled_df)[0]  # shape: [10]
    label_map = {i: round(0.5 + 0.5*i, 1) for i in range(10)}
    predicted_score = label_map.get(prediction, prediction)

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Predicted HSR Score:** `{predicted_score}`")
    st.pyplot(draw_hsr_star_plot(predicted_score))

    st.subheader("📊 Probability Table")
    prob_df = pd.DataFrame({
        "HSR Score": [f"{label_map[i]}" for i in range(len(prob_array))],
        "Probability": [f"{p:.2f}" for p in prob_array]
    })
    st.dataframe(prob_df, use_container_width=True)

    st.subheader("📈 SHAP Force Plot (All Classes)")
    shap_values = explainer(user_scaled_df)
    for class_index in range(shap_values.values.shape[2]):
        with st.expander(f"🔍 SHAP for HSR Score {label_map[class_index]}"):
            force_plot_html = shap.plots.force(
                base_value=explainer.expected_value[class_index],
                shap_values=shap_values.values[0, :, class_index],
                features=user_scaled_df.iloc[0],
                matplotlib=False,
                show=False
            )
            components.html(shap.getjs() + force_plot_html.html(), height=400)

st.markdown("---")
st.markdown("Developed for 10-level HSR prediction · Research use only.")
