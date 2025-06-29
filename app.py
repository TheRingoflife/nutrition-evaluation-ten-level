import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
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

# ===== HSR 风格图绘制函数 =====
def draw_basic_hsr_template(score=4.5):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 主圆
    main_circle = Circle((5, 5.5), 4, facecolor='#005eb8', edgecolor='black', lw=2)
    ax.add_patch(main_circle)

    # 分数文本
    score_text = ax.text(5, 5.5, f"{score}", fontsize=38, color='white',
                         ha='center', va='center', weight='bold')
    score_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

    # 下方说明
    ax.text(5, 2.0, "HEALTH STAR", fontsize=14, ha='center', va='center', color='black', weight='bold')
    ax.text(5, 1.0, "RATING", fontsize=14, ha='center', va='center', color='black', weight='bold')
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

    # ===== 展示预测结果 =====
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Predicted HSR Score:** `{predicted_score}`")
    st.pyplot(draw_basic_hsr_template(predicted_score))

    st.subheader("📊 Probability Table")
    prob_df = pd.DataFrame({
        "HSR Score": [f"{label_map[i]}" for i in range(len(prob_array))],
        "Probability": [f"{p:.2f}" for p in prob_array]
    })
    st.dataframe(prob_df, use_container_width=True)

    # ===== SHAP 力图（全部类别） =====
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

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed for 10-level HSR prediction · Research use only.")
