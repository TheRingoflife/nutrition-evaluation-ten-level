import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})

LANGUAGES = {
    "English": "en",
    "中文": "zh",
}

TEXTS = {
    "en": {
        "title": "🍱 HSR-Derived Front-of-Pack Labelling Prototype",
        "subtitle": "Nine-Category Star-Rating Classification (0.5–4.5)",
        "description": (
            "This prototype uses a locked XGBoost model and three routinely "
            "available on-pack nutrients to generate an approximate nine-category "
            "HSR-derived classification for prepackaged ready foods. The nine "
            "categories correspond to the observed HSR values from 0.5 to 4.5 "
            "stars; no products with an HSR of 5.0 were observed in the study sample."
        ),
        "target_audience": "🎯 Intended Users",
        "audience_desc": (
            "Designed for researchers and public-health practitioners exploring "
            "food-supply monitoring in settings where complete HSR inputs are unavailable."
        ),
        "problem_statement": "📊 Problem Statement",
        "problem_desc": (
            "Direct HSR calculation requires nutritional and compositional inputs "
            "that may not be routinely reported on food labels."
        ),
        "solution": "💡 Prototype Approach",
        "solution_desc": (
            "The locked model uses sodium, energy and protein to approximate one "
            "of the nine observed HSR-derived categories and provides SHAP-based "
            "explanations of individual predictions."
        ),
        "mission": "🚀 Intended Application",
        "mission_desc": (
            "This prototype demonstrates a reduced-input approach for exploratory "
            "food-supply monitoring and comparisons among products within the same "
            "ready-food category when complete inputs for direct HSR calculation "
            "are unavailable. The output is a predicted HSR-derived category and "
            "should not be interpreted as a formally calculated Health Star Rating "
            "or an independent assessment of overall product healthfulness."
        ),
        "input_variables": "🔢 Available On-Pack Input Variables",
        "sodium_label": "Sodium (mg/100 g)",
        "energy_label": "Energy (kJ/100 g)",
        "protein_label": "Protein (g/100 g)",
        "help_sodium": "Sodium content per 100 g of the product",
        "help_energy": "Energy content per 100 g of the product (kJ, not kcal)",
        "help_protein": "Protein content per 100 g of the product",
        "predict_button": "🧮 Generate Predicted HSR-Derived Category",
        "prediction_result": "🔍 Prediction Result",
        "health_categories": {
            0: {
                "name": "0.5",
                "stars": "0.5⭐",
                "color": "#dc3545",
                "description": "Predicted HSR-derived category corresponding to 0.5 stars",
            },
            1: {
                "name": "1.0",
                "stars": "1.0⭐",
                "color": "#fd7e14",
                "description": "Predicted HSR-derived category corresponding to 1.0 star",
            },
            2: {
                "name": "1.5",
                "stars": "1.5⭐",
                "color": "#ffc107",
                "description": "Predicted HSR-derived category corresponding to 1.5 stars",
            },
            3: {
                "name": "2.0",
                "stars": "2.0⭐",
                "color": "#ffc107",
                "description": "Predicted HSR-derived category corresponding to 2.0 stars",
            },
            4: {
                "name": "2.5",
                "stars": "2.5⭐",
                "color": "#ffc107",
                "description": "Predicted HSR-derived category corresponding to 2.5 stars",
            },
            5: {
                "name": "3.0",
                "stars": "3.0⭐",
                "color": "#ffc107",
                "description": "Predicted HSR-derived category corresponding to 3.0 stars",
            },
            6: {
                "name": "3.5",
                "stars": "3.5⭐",
                "color": "#28a745",
                "description": "Predicted HSR-derived category corresponding to 3.5 stars",
            },
            7: {
                "name": "4.0",
                "stars": "4.0⭐",
                "color": "#28a745",
                "description": "Predicted HSR-derived category corresponding to 4.0 stars",
            },
            8: {
                "name": "4.5",
                "stars": "4.5⭐",
                "color": "#20c997",
                "description": "Predicted HSR-derived category corresponding to 4.5 stars",
            },
        },
        "confidence": "Model-Estimated Probability",
        "feature_importance": "📊 Feature Importance",
        "shap_plot": "📊 SHAP Force Plot",
        "base_value": "Baseline Model Output",
        "final_prediction": "Final Model Output",
        "expand_shap": "Click to view the SHAP force plot",
        "shap_success": "✅ SHAP force plot created!",
        "shap_html_success": "✅ SHAP force plot created!",
        "shap_custom_success": "✅ SHAP force plot created!",
        "shap_table": "📊 SHAP Values Table",
        "shap_table_info": (
            "💡 SHAP values show how each input contributes to the model output "
            "for the selected category."
        ),
        "prediction_probabilities": "Model-Estimated Category Probabilities",
        "positive_impact": (
            "Positive contribution toward this HSR-derived category"
        ),
        "negative_impact": (
            "Negative contribution away from this HSR-derived category"
        ),
        "warning_input": (
            "⚠️ Please enter the required input values before generating a prediction."
        ),
        "input_tip": (
            "💡 Enter the nutrient values exactly as declared per 100 g on the "
            "product package."
        ),
        "model_error": (
            "❌ Prediction cannot proceed because the required model or "
            "preprocessing files are unavailable."
        ),
        "prediction_failed": "Prediction failed",
        "shap_failed": "SHAP analysis failed",
        "shap_unavailable": (
            "💡 A SHAP explanation is unavailable, but feature importance is "
            "shown above."
        ),
        "footer": (
            "Developed using Streamlit and XGBoost · "
            "For exploratory research use only."
        ),
        "feature_names": ["Sodium", "Energy", "Protein"],
        "chart_feature_names": ["Sodium", "Energy", "Protein"],
    },

    "zh": {
        "title": "🍱 HSR衍生正面标签原型",
        "subtitle": "九分类星级标签（0.5–4.5）",
        "description": (
            "本原型使用已锁定的XGBoost模型，根据包装上常规标示的三项营养素，"
            "为预包装即食食品生成近似的九分类HSR衍生结果。九个类别对应研究样本"
            "中观察到的0.5至4.5星HSR值；样本中未观察到HSR为5.0星的产品。"
        ),
        "target_audience": "🎯 预期使用者",
        "audience_desc": (
            "面向在缺少完整HSR计算所需信息的情况下，开展食品供应监测探索的"
            "研究人员和公共卫生实践人员。"
        ),
        "problem_statement": "📊 问题陈述",
        "problem_desc": (
            "直接计算HSR需要多项营养和食品组成信息，而这些信息未必都会在"
            "食品标签上常规标示。"
        ),
        "solution": "💡 原型方法",
        "solution_desc": (
            "已锁定的模型使用钠、能量和蛋白质，近似预测九个已观察到的"
            "HSR衍生类别之一，并使用SHAP解释各输入变量对单次预测的贡献。"
        ),
        "mission": "🚀 预期应用",
        "mission_desc": (
            "当无法获得直接计算HSR所需的完整信息时，本原型展示了一种基于较少"
            "输入的探索性食品供应监测方法，并可辅助比较同一即食食品类别内产品"
            "的相对差异。模型输出是预测的HSR衍生类别，不代表正式计算得到的健康"
            "星级评分，也不构成对产品整体健康程度的独立评价。"
        ),
        "input_variables": "🔢 包装上可获得的输入变量",
        "sodium_label": "钠（mg/100 g）",
        "energy_label": "能量（kJ/100 g）",
        "protein_label": "蛋白质（g/100 g）",
        "help_sodium": "产品每100 g中的钠含量",
        "help_energy": "产品每100 g中的能量含量（单位为kJ，而非kcal）",
        "help_protein": "产品每100 g中的蛋白质含量",
        "predict_button": "🧮 生成预测的HSR衍生类别",
        "prediction_result": "🔍 预测结果",
        "health_categories": {
            0: {
                "name": "0.5",
                "stars": "0.5⭐",
                "color": "#dc3545",
                "description": "预测的HSR衍生类别，对应0.5星",
            },
            1: {
                "name": "1.0",
                "stars": "1.0⭐",
                "color": "#fd7e14",
                "description": "预测的HSR衍生类别，对应1.0星",
            },
            2: {
                "name": "1.5",
                "stars": "1.5⭐",
                "color": "#ffc107",
                "description": "预测的HSR衍生类别，对应1.5星",
            },
            3: {
                "name": "2.0",
                "stars": "2.0⭐",
                "color": "#ffc107",
                "description": "预测的HSR衍生类别，对应2.0星",
            },
            4: {
                "name": "2.5",
                "stars": "2.5⭐",
                "color": "#ffc107",
                "description": "预测的HSR衍生类别，对应2.5星",
            },
            5: {
                "name": "3.0",
                "stars": "3.0⭐",
                "color": "#ffc107",
                "description": "预测的HSR衍生类别，对应3.0星",
            },
            6: {
                "name": "3.5",
                "stars": "3.5⭐",
                "color": "#28a745",
                "description": "预测的HSR衍生类别，对应3.5星",
            },
            7: {
                "name": "4.0",
                "stars": "4.0⭐",
                "color": "#28a745",
                "description": "预测的HSR衍生类别，对应4.0星",
            },
            8: {
                "name": "4.5",
                "stars": "4.5⭐",
                "color": "#20c997",
                "description": "预测的HSR衍生类别，对应4.5星",
            },
        },
        "confidence": "模型估计概率",
        "feature_importance": "📊 特征重要性",
        "shap_plot": "📊 SHAP力图",
        "base_value": "模型基准输出",
        "final_prediction": "最终模型输出",
        "expand_shap": "点击查看SHAP力图",
        "shap_success": "✅ SHAP力图已生成！",
        "shap_html_success": "✅ SHAP力图已生成！",
        "shap_custom_success": "✅ SHAP力图已生成！",
        "shap_table": "📊 SHAP值表格",
        "shap_table_info": (
            "💡 SHAP值表示各输入变量对所选类别模型输出的贡献。"
        ),
        "prediction_probabilities": "各类别的模型估计概率",
        "positive_impact": "推动模型输出趋向该HSR衍生类别的正向贡献",
        "negative_impact": "推动模型输出远离该HSR衍生类别的负向贡献",
        "warning_input": "⚠️ 请填写所需的输入变量后再进行预测。",
        "input_tip": "💡 请按照产品包装标示填写每100 g的营养素数值。",
        "model_error": "❌ 缺少所需的模型或预处理文件，无法进行预测。",
        "prediction_failed": "预测失败",
        "shap_failed": "SHAP分析失败",
        "shap_unavailable": (
            "💡 当前无法生成SHAP解释，但上方仍显示特征重要性。"
        ),
        "footer": (
            "使用Streamlit和XGBoost开发 · 仅供探索性研究使用。"
        ),
        "feature_names": ["钠", "能量", "蛋白质"],
        "chart_feature_names": ["Sodium", "Energy", "Protein"],
    },
}

st.set_page_config(
    page_title="Nutritional Quality Classifier (9-Class 0.5-4.5)",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_language():
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        lang_choice = st.selectbox("🌐 Language", list(LANGUAGES.keys()))
    return TEXTS[LANGUAGES[lang_choice]]


texts = get_language()

st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">{texts['title']}</h1>
    <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin-bottom: 2rem;">
    <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{texts['description']}</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #1976d2; margin: 0 0 0.5rem 0;">{texts['target_audience']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['audience_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="background: #f3e5f5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #7b1fa2; margin: 0 0 0.5rem 0;">{texts['problem_statement']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['problem_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    st.markdown(f"""
    <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #2e7d32; margin: 0 0 0.5rem 0;">{texts['solution']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['solution_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #f57c00; margin: 0 0 0.5rem 0;">{texts['mission']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['mission_desc']}</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load("XGBoost_retrained_model.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler2.pkl")
    except Exception as e:
        st.error(f"Scaler loading failed: {e}")
        return None


@st.cache_resource
def load_background():
    try:
        return np.load("background_data.npy")
    except Exception as e:
        st.error(f"background_data loading failed: {e}")
        return None


@st.cache_resource
def load_feature_names():
    try:
        names = np.load("feature_names.npy", allow_pickle=True)
        return [str(x) for x in names.tolist()]
    except Exception:
        return ["Sodium", "Energy", "Protein"]


model = load_model()
scaler = load_scaler()
background_data = load_background()
chart_names = load_feature_names()

if model is None or scaler is None or background_data is None:
    st.error(texts["model_error"])
    st.stop()

st.sidebar.markdown(f"## {texts['input_variables']}")
st.sidebar.markdown(f"""
<div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <p style="margin: 0; font-size: 0.9rem; color: #1976d2;">
        <strong>{texts['input_tip']}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

sodium = st.sidebar.number_input(texts["sodium_label"], min_value=0.0, step=1.0, help=texts["help_sodium"])
energy = st.sidebar.number_input(texts["energy_label"], min_value=0.0, step=1.0, help=texts["help_energy"])
protein = st.sidebar.number_input(texts["protein_label"], min_value=0.0, step=0.1, help=texts["help_protein"])

if st.sidebar.button(texts["predict_button"], type="primary", use_container_width=True):
    if sodium == 0 and energy == 0 and protein == 0:
        st.warning(texts["warning_input"])
        st.stop()
    try:
        input_data = np.array([[sodium, energy, protein]], dtype=float)
        input_scaled = scaler.transform(input_data)
        user_scaled_df = pd.DataFrame(input_scaled, columns=chart_names)

        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]
        n_class = len(texts["health_categories"])
        if prediction < 0 or prediction >= n_class:
            st.error(f"Prediction index {prediction} is out of range. Expected 0-{n_class - 1}")
            st.stop()
        category_info = texts["health_categories"][prediction]
        confidence = float(probabilities[prediction])

        st.markdown(f"## {texts['prediction_result']}")
        st.markdown(f"""
        <div style="background: {category_info['color']}; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="margin: 0; font-size: 2.5rem;">{category_info['stars']}</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">{category_info['description']}</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['confidence']}: <strong>{confidence:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### 📊 {texts['prediction_probabilities']} (0.5-4.5)")
        items = list(texts["health_categories"].items())
        prob_cols = st.columns(5)
        for col, (cat_id, cat_info) in zip(prob_cols, items[:5]):
            with col:
                st.metric(f"{cat_info['stars']} {cat_info['name']}", f"{float(probabilities[cat_id]):.3f}")
        prob_cols2 = st.columns(5)
        for col, (cat_id, cat_info) in zip(prob_cols2, items[5:]):
            with col:
                st.metric(f"{cat_info['stars']} {cat_info['name']}", f"{float(probabilities[cat_id]):.3f}")

        st.markdown(f"## {texts['feature_importance']}")
        final_model = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(final_model, "feature_importances_"):
            feature_importance = final_model.feature_importances_
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(chart_names, feature_importance, color=["#ff6b6b", "#4ecdc4", "#45b7d1"])
            ax.set_xlabel("Importance", fontsize=12)
            ax.set_title("Feature Importance Analysis", fontsize=14, fontweight="bold")
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", ha="left", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(f"## {texts['shap_plot']}")
        if not SHAP_AVAILABLE:
            st.info(texts["shap_unavailable"])
        else:
            try:
                explainer = shap.Explainer(model.predict_proba, background_data)
                shap_values = explainer(user_scaled_df)
                expected_value = model.predict_proba(background_data).mean(axis=0)
                if hasattr(shap_values, "values"):
                    if len(shap_values.values.shape) == 3:
                        shap_vals = shap_values.values[0, :, prediction]
                        base_val = expected_value[prediction]
                    else:
                        shap_vals = shap_values.values[0, :]
                        base_val = expected_value[0]
                else:
                    shap_vals = shap_values[0, :]
                    base_val = expected_value[0]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(texts["base_value"], f"{base_val:.4f}")
                with col_b:
                    st.metric(texts["final_prediction"], f"{base_val + shap_vals.sum():.4f}")

                with st.expander(texts["expand_shap"], expanded=True):
                    try:
                        plt.figure(figsize=(16, 6))
                        shap.force_plot(
                            base_val, shap_vals, user_scaled_df.iloc[0],
                            feature_names=chart_names, matplotlib=True, show=False,
                        )
                        plt.tight_layout()
                        st.pyplot(plt)
                        plt.close()
                        st.success(texts["shap_success"])
                    except Exception as e:
                        st.warning(f"Matplotlib version failed: {e}")
                        try:
                            force_plot = shap.force_plot(
                                base_val, shap_vals, user_scaled_df.iloc[0],
                                feature_names=chart_names, matplotlib=False,
                            )
                            components.html(shap.getjs() + force_plot.html(), height=400)
                            st.success(texts["shap_html_success"])
                        except Exception as e2:
                            st.warning(f"HTML version also failed: {e2}")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            colors = ["#ff6b6b" if x < 0 else "#4ecdc4" for x in shap_vals]
                            bars = ax.barh(chart_names, shap_vals, color=colors, alpha=0.8, height=0.6)
                            for bar, shap_val, feature_val, feature_name in zip(bars, shap_vals, user_scaled_df.iloc[0].values, chart_names):
                                width = bar.get_width()
                                y_pos = bar.get_y() + bar.get_height() / 2
                                ax.text(width / 2, y_pos, f"{shap_val:.3f}", ha="center", va="center", color="white", fontweight="bold")
                                ax.text(width + 0.05 if width > 0 else width - 0.05, y_pos, f"{feature_name}: {feature_val:.2f}",
                                        ha="left" if width > 0 else "right", va="center", fontsize=11, fontweight="bold")
                            ax.axvline(x=0, color="black", linestyle="-", alpha=0.5, linewidth=2)
                            ax.set_xlabel("SHAP Value")
                            ax.set_title(f"SHAP Force Plot - {category_info['stars']}")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            st.success(texts["shap_custom_success"])
            except Exception as e:
                st.error(f"{texts['shap_failed']}: {e}")
                st.info(texts["shap_unavailable"])
    except Exception as e:
        st.error(f"{texts['prediction_failed']}: {e}")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p style="margin: 0;">{texts['footer']}</p>
</div>
""", unsafe_allow_html=True)
