import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 设置matplotlib参数，避免重叠
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# ===== 多语言支持 =====
LANGUAGES = {
    "English": "en",
    "中文": "zh"
}

TEXTS = {
    "en": {
        "title": "🍱 Nutritional Quality Classifier",
        "subtitle": "ML-Powered Ready Food Health Assessment (9-Class 0.5-4.5)",
        "description": "This advanced machine learning application uses XGBoost to predict the nutritional healthiness of ready foods based on 4 key nutritional features with 9 health grades (0.5-4.5).",
        "target_audience": "🎯 Target Audience",
        "audience_desc": "Designed for countries with limited nutritional information and consumers seeking quick, reliable food health assessments.",
        "problem_statement": "📊 Problem Statement",
        "problem_desc": "Many countries lack comprehensive nutritional labeling systems, making it difficult to implement generalized positive labeling for food products.",
        "solution": "💡 Our Solution",
        "solution_desc": "Advanced ML model analyzes 4 key nutritional features to provide instant, accurate health predictions with detailed explanations across 9 health grades (0.5-4.5).",
        "mission": "🚀 Mission",
        "mission_desc": "Providing a practical approach for countries with incomplete nutritional information to implement effective food health assessment systems.",
        "input_variables": "🔢 Input Variables",
        "protein_label": "Protein (g/100g)",
        "sodium_label": "Sodium (mg/100g)",
        "energy_label": "Energy (kJ/100g)",
        "processed_label": "Is Ultra-Processed? (procef_4)",
        "help_sodium": "Sodium content per 100g of food",
        "help_energy": "Energy content per 100g of food",
        "help_protein": "Protein content per 100g of food",
        "help_procef_4": "0=Not ultra-processed, 1=Ultra-processed",
        "predict_button": "🧮 Predict Healthiness",
        "prediction_result": "🔍 Prediction Result",
        "health_categories": {
            0: {"name": "0.5", "stars": "0.5⭐", "color": "#dc3545", "description": "Very Unhealthy"},
            1: {"name": "1.0", "stars": "1.0⭐", "color": "#fd7e14", "description": "Unhealthy"},
            2: {"name": "1.5", "stars": "1.5⭐", "color": "#ffc107", "description": "Poor"},
            3: {"name": "2.0", "stars": "2.0⭐", "color": "#ffc107", "description": "Below Average"},
            4: {"name": "2.5", "stars": "2.5⭐", "color": "#ffc107", "description": "Average"},
            5: {"name": "3.0", "stars": "3.0⭐", "color": "#ffc107", "description": "Above Average"},
            6: {"name": "3.5", "stars": "3.5⭐", "color": "#28a745", "description": "Good"},
            7: {"name": "4.0", "stars": "4.0⭐", "color": "#28a745", "description": "Very Good"},
            8: {"name": "4.5", "stars": "4.5⭐", "color": "#20c997", "description": "Excellent"}
        },
        "confidence": "Confidence",
        "feature_importance": "📊 Feature Importance",
        "shap_plot": "📊 SHAP Force Plot",
        "base_value": "Base value",
        "final_prediction": "Final prediction",
        "expand_shap": "Click to view SHAP force plot",
        "shap_success": "✅ SHAP force plot created!",
        "shap_html_success": "✅ SHAP force plot created!",
        "shap_custom_success": "✅ SHAP force plot created!",
        "shap_table": "📊 SHAP Values Table",
        "shap_table_info": "💡 SHAP values displayed as table",
        "prediction_probabilities": "Prediction Probabilities",
        "positive_impact": "Positive Impact (Higher Health)",
        "negative_impact": "Negative Impact (Lower Health)",
        "warning_input": "⚠️ Please enter values for at least one feature before predicting.",
        "input_tip": "💡 Tip: Please enter the nutritional information of the food, and the system will predict its healthiness across 9 grades (0.5-4.5).",
        "model_error": "❌ Cannot proceed without model and scaler files",
        "prediction_failed": "Prediction failed",
        "shap_failed": "SHAP analysis failed",
        "shap_unavailable": "💡 SHAP explanation is not available, but feature importance is shown above.",
        "footer": "Developed using Streamlit and XGBoost · For research use only.",
        "feature_names": ["Energy", "Protein", "Sodium", "procef_4"],
        "chart_feature_names": ["Energy", "Protein", "Sodium", "procef_4"]
    },
    "zh": {
        "title": "🍱 营养质量分类器",
        "subtitle": "ML驱动的即食食品健康评估（九分类0.5-4.5等级）",
        "description": "这个先进的机器学习应用程序使用XGBoost根据4个关键营养特征预测即食食品的营养健康性，分为9个健康等级（0.5-4.5）。",
        "target_audience": "🎯 目标用户",
        "audience_desc": "专为营养信息有限的国家和寻求快速、可靠食品健康评估的消费者设计。",
        "problem_statement": "📊 问题陈述",
        "problem_desc": "许多国家缺乏全面的营养标签系统，难以实施食品的概括性正面标签。",
        "solution": "💡 我们的解决方案",
        "solution_desc": "先进的ML模型分析4个关键营养特征，提供即时、准确的健康预测和详细解释，涵盖9个健康等级（0.5-4.5）。",
        "mission": "🚀 使命",
        "mission_desc": "为营养信息纰漏不全导致无法使用概括性正面标签的国家提供一个使用思路。",
        "input_variables": "🔢 输入变量",
        "protein_label": "蛋白质 (g/100g)",
        "sodium_label": "钠 (mg/100g)",
        "energy_label": "能量 (kJ/100g)",
        "processed_label": "是否超加工？(procef_4)",
        "help_sodium": "每100g食品中的钠含量",
        "help_energy": "每100g食品中的能量含量",
        "help_procef_4": "0=非超加工, 1=超加工",
        "help_protein": "每100g食品中的蛋白质含量",
        "predict_button": "🧮 预测健康性",
        "prediction_result": "🔍 预测结果",
        "health_categories": {
            0: {"name": "0.5", "stars": "0.5⭐", "color": "#dc3545", "description": "非常不健康"},
            1: {"name": "1.0", "stars": "1.0⭐", "color": "#fd7e14", "description": "不健康"},
            2: {"name": "1.5", "stars": "1.5⭐", "color": "#ffc107", "description": "较差"},
            3: {"name": "2.0", "stars": "2.0⭐", "color": "#ffc107", "description": "低于平均"},
            4: {"name": "2.5", "stars": "2.5⭐", "color": "#ffc107", "description": "平均"},
            5: {"name": "3.0", "stars": "3.0⭐", "color": "#ffc107", "description": "高于平均"},
            6: {"name": "3.5", "stars": "3.5⭐", "color": "#28a745", "description": "良好"},
            7: {"name": "4.0", "stars": "4.0⭐", "color": "#28a745", "description": "很好"},
            8: {"name": "4.5", "stars": "4.5⭐", "color": "#20c997", "description": "优秀"}
        },
        "confidence": "置信度",
        "feature_importance": "📊 特征重要性",
        "shap_plot": "📊 SHAP力图",
        "base_value": "基准值",
        "final_prediction": "最终预测",
        "expand_shap": "点击查看SHAP力图",
        "shap_success": "✅ SHAP力图创建成功!",
        "shap_html_success": "✅ SHAP力图创建成功!",
        "shap_custom_success": "✅ SHAP力图创建成功!",
        "shap_table": "📊 SHAP值表格",
        "shap_table_info": "💡 SHAP值以表格形式显示",
        "prediction_probabilities": "预测概率",
        "positive_impact": "积极影响 (更高健康性)",
        "negative_impact": "消极影响 (更低健康性)",
        "warning_input": "⚠️ 请在预测前至少输入一个特征的值。",
        "input_tip": "💡 提示: 请输入食品的营养成分信息，系统将预测其健康性（9个等级0.5-4.5）。",
        "model_error": "❌ 没有模型和标准化器文件无法继续",
        "prediction_failed": "预测失败",
        "shap_failed": "SHAP分析失败",
        "shap_unavailable": "💡 SHAP解释不可用，但上面显示了特征重要性。",
        "footer": "使用Streamlit和XGBoost开发 · 仅供研究使用。",
        "feature_names": ["能量", "蛋白质", "钠", "procef_4"],
        "chart_feature_names": ["Energy", "Protein", "Sodium", "procef_4"]
    }
}

# ===== 页面设置 =====
st.set_page_config(
    page_title="Nutritional Quality Classifier (9-Class 0.5-4.5)",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 语言选择器 =====
def get_language():
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        lang_choice = st.selectbox("🌐 Language", list(LANGUAGES.keys()))
    return TEXTS[LANGUAGES[lang_choice]]

# 获取当前语言文本
texts = get_language()

# ===== 主标题区域 =====
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">{texts['title']}</h1>
    <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ===== 应用描述 =====
st.markdown(f"""
<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin-bottom: 2rem;">
    <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{texts['description']}</p>
</div>
""", unsafe_allow_html=True)

# ===== 信息卡片 =====
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

# ===== 加载模型 =====
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

model = load_model()
scaler = load_scaler()

if model is None or scaler is None:
    st.error(texts['model_error'])
    st.stop()

# ===== 侧边栏输入 =====
st.sidebar.markdown(f"## {texts['input_variables']}")

# 添加输入说明
st.sidebar.markdown(f"""
<div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <p style="margin: 0; font-size: 0.9rem; color: #1976d2;">
        <strong>{texts['input_tip']}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# 4个输入特征，按照指定顺序：能量、蛋白质、钠、procef_4，默认为空
energy = st.sidebar.number_input(texts['energy_label'], min_value=0.0, step=1.0, value=None, help=texts['help_energy'])
protein = st.sidebar.number_input(texts['protein_label'], min_value=0.0, step=0.1, value=None, help=texts['help_protein'])
sodium = st.sidebar.number_input(texts['sodium_label'], min_value=0.0, step=1.0, value=None, help=texts['help_sodium'])
procef_4 = st.sidebar.selectbox(texts['processed_label'], [0, 1], help=texts['help_procef_4'])

# 添加预测按钮样式
if st.sidebar.button(texts['predict_button'], type="primary", use_container_width=True):
    # 将None值转换为0进行处理
    energy = energy if energy is not None else 0.0
    protein = protein if protein is not None else 0.0
    sodium = sodium if sodium is not None else 0.0
    
    # 检查输入是否为零
    if energy == 0 and protein == 0 and sodium == 0:
        st.warning(texts['warning_input'])
        st.stop()
    
    try:
        # 1. 准备输入数据 - 按照指定顺序：能量、蛋白质、钠、procef_4
        input_data = np.array([[energy, protein, sodium, procef_4]], dtype=float)
        input_scaled = scaler.transform(input_data)
        user_scaled_df = pd.DataFrame(input_scaled, columns=texts['chart_feature_names'])
        
        # 2. 预测
        prediction = model.predict(user_scaled_df)[0]
        probabilities = model.predict_proba(user_scaled_df)[0]
        
        # 检查预测结果是否在有效范围内
        if prediction >= len(texts['health_categories']):
            st.error(f"Prediction index {prediction} is out of range. Expected 0-{len(texts['health_categories'])-1}")
            st.stop()
        
        # 3. 展示结果 - 美化
        st.markdown(f"## {texts['prediction_result']}")
        
        # 获取预测类别的信息
        category_info = texts['health_categories'][prediction]
        confidence = probabilities[prediction]
        
        # 结果卡片 - 使用数字+星星显示
        st.markdown(f"""
        <div style="background: {category_info['color']}; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="margin: 0; font-size: 2.5rem;">{category_info['stars']}</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">{category_info['description']}</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['confidence']}: <strong>{confidence:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示所有类别的概率 - 使用两行显示9个等级，用数字+星星表示
        st.markdown(f"### 📊 {texts['prediction_probabilities']} (0.5-4.5 Grades)")
        prob_cols = st.columns(5)
        for i, (col, (cat_id, cat_info)) in enumerate(zip(prob_cols, list(texts['health_categories'].items())[:5])):
            with col:
                prob_value = probabilities[cat_id]
                delta_value = f"{prob_value-0.1:.3f}" if prob_value > 0.1 else None
                st.metric(
                    f"{cat_info['stars']} {cat_info['name']}", 
                    f"{prob_value:.3f}",
                    delta=delta_value
                )
        
        # 第二行显示剩余4个等级
        prob_cols2 = st.columns(5)
        for i, (col, (cat_id, cat_info)) in enumerate(zip(prob_cols2, list(texts['health_categories'].items())[5:])):
            with col:
                prob_value = probabilities[cat_id]
                delta_value = f"{prob_value-0.1:.3f}" if prob_value > 0.1 else None
                st.metric(
                    f"{cat_info['stars']} {cat_info['name']}", 
                    f"{prob_value:.3f}",
                    delta=delta_value
                )
        
        # 4. 特征重要性
        st.markdown(f"## {texts['feature_importance']}")
        
        if hasattr(model, 'steps'):
            final_model = model.steps[-1][1]
            if hasattr(final_model, 'feature_importances_'):
                feature_importance = final_model.feature_importances_
                features = texts['chart_feature_names']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(features, feature_importance, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # 5. SHAP力图
        st.markdown(f"## {texts['shap_plot']}")
        
        try:
            # 创建背景数据
            np.random.seed(42)
            background_data = np.random.normal(0, 1, (100, 4)).astype(float)
            
            # 使用 Explainer
            explainer = shap.Explainer(model.predict_proba, background_data)
            shap_values = explainer(user_scaled_df)
            
            # 计算期望值
            background_predictions = model.predict_proba(background_data)
            expected_value = background_predictions.mean(axis=0)
            
            # 获取 SHAP 值 - 对于多分类，我们需要选择预测类别的SHAP值
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    shap_vals = shap_values.values[0, :, prediction]  # 选择预测类别的SHAP值
                    base_val = expected_value[prediction]
                else:
                    shap_vals = shap_values.values[0, :]
                    base_val = expected_value[0]
            else:
                shap_vals = shap_values[0, :]
                base_val = expected_value[0]
            
            # 显示 SHAP 值信息
            col1, col2 = st.columns(2)
            with col1:
                st.metric(texts['base_value'], f"{base_val:.4f}")
            with col2:
                st.metric(texts['final_prediction'], f"{base_val + shap_vals.sum():.4f}")
            
            # 创建 SHAP 力图
            with st.expander(texts['expand_shap'], expanded=True):
                # 方法1：优先使用matplotlib版本
                try:
                    plt.figure(figsize=(16, 6))
                    
                    # 创建自定义的DataFrame，特征值保留两位小数
                    custom_df = user_scaled_df.copy()
                    for col in custom_df.columns:
                        custom_df[col] = custom_df[col].round(2)
                    
                    shap.force_plot(base_val, shap_vals,
                                   custom_df.iloc[0], 
                                   feature_names=texts['chart_feature_names'],
                                   matplotlib=True, show=False)
                    
                    # 修改标题为纯英文，移除description部分
                    # plt.title(f'SHAP Force Plot - {category_info["stars"]} Prediction', fontsize=16, fontweight='bold', pad=30)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                    st.success(texts['shap_success'])
                    
                except Exception as e:
                    st.warning(f"Matplotlib version failed: {e}")
                    
                    # 方法2：使用 HTML 版本作为备用
                    try:
                        # 创建自定义的DataFrame，特征值保留两位小数
                        custom_df = user_scaled_df.copy()
                        for col in custom_df.columns:
                            custom_df[col] = custom_df[col].round(2)
                        
                        force_plot = shap.force_plot(
                            base_val,
                            shap_vals,
                            custom_df.iloc[0],
                            feature_names=texts['chart_feature_names'],
                            matplotlib=False
                        )
                        
                        force_html = force_plot.html()
                        components.html(shap.getjs() + force_html, height=400)
                        st.success(texts['shap_html_success'])
                        
                    except Exception as e2:
                        st.warning(f"HTML version also failed: {e2}")
                        
                        # 方法3：自定义清晰的条形图（优化版本，按SHAP值排序，特征值保留两位小数）
                        try:
                            # 按SHAP值从大到小排序
                            sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
                            sorted_features = [texts['chart_feature_names'][i] for i in sorted_indices]
                            sorted_shap_vals = shap_vals[sorted_indices]
                            sorted_feature_values = [custom_df.iloc[0, i] for i in sorted_indices]
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in sorted_shap_vals]
                            bars = ax.barh(sorted_features, sorted_shap_vals, color=colors, alpha=0.8, height=0.6)
                            
                            # 优化标签显示，避免重叠
                            for i, (bar, shap_val, feature_val, feature_name) in enumerate(zip(bars, sorted_shap_vals, sorted_feature_values, sorted_features)):
                                width = bar.get_width()
                                y_pos = bar.get_y() + bar.get_height()/2
                                
                                # 在条形图内部显示SHAP值
                                ax.text(width/2, y_pos, f'{shap_val:.3f}', 
                                       ha='center', va='center', color='white', fontweight='bold', fontsize=11)
                                
                                # 在条形图外部显示特征名称和值，使用更好的布局
                                if width > 0:
                                    # 右侧显示
                                    ax.text(width + 0.1, y_pos, f'{feature_name}', 
                                           ha='left', va='center', fontsize=12, fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                                    ax.text(width + 0.1, y_pos - 0.15, f'Val: {feature_val:.2f}', 
                                           ha='left', va='center', fontsize=10, style='italic',
                                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcyan", alpha=0.6))
                                else:
                                    # 左侧显示
                                    ax.text(width - 0.1, y_pos, f'{feature_name}', 
                                           ha='right', va='center', fontsize=12, fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
                                    ax.text(width - 0.1, y_pos - 0.15, f'Val: {feature_val:.2f}', 
                                           ha='right', va='center', fontsize=10, style='italic',
                                           bbox=dict(boxstyle="round,pad=0.2", facecolor="mistyrose", alpha=0.6))
                            
                            # 添加零线
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
                            ax.set_xlabel('SHAP Value', fontsize=14, fontweight='bold')
                            ax.set_ylabel('Features', fontsize=14, fontweight='bold')
                            ax.set_title(f'SHAP Force Plot - {category_info["stars"]} Prediction (Sorted by Impact)', fontsize=16, fontweight='bold', pad=20)
                            ax.grid(True, alpha=0.3)
                            
                            # 添加图例
                            legend_elements = [
                                plt.Rectangle((0,0),1,1, facecolor='#4ecdc4', alpha=0.8, label=texts['positive_impact']),
                                plt.Rectangle((0,0),1,1, facecolor='#ff6b6b', alpha=0.8, label=texts['negative_impact'])
                            ]
                            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
                            
                            # 调整布局，确保标签不被截断
                            plt.tight_layout()
                            plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
                            st.pyplot(fig)
                            plt.close()
                            st.success(texts['shap_custom_success'])
                            
                        except Exception as e3:
                            st.error(f"All SHAP plots failed: {e3}")
                            
                            # 方法4：显示详细表格
                            st.markdown(f"### {texts['shap_table']}")
                            shap_df = pd.DataFrame({
                                'Feature': sorted_features,
                                'Feature Value': [f"{val:.2f}" for val in sorted_feature_values],
                                'SHAP Value': [f"{val:.3f}" for val in sorted_shap_vals],
                                'Impact': [texts['negative_impact'] if x < 0 else texts['positive_impact'] for x in sorted_shap_vals]
                            })
                            st.dataframe(shap_df, use_container_width=True)
                            st.info(texts['shap_table_info'])
            
        except Exception as e:
            st.error(f"{texts['shap_failed']}: {e}")
            st.info(texts['shap_unavailable'])
            
    except Exception as e:
        st.error(f"{texts['prediction_failed']}: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p style="margin: 0;">{texts['footer']}</p>
</div>
""", unsafe_allow_html=True)
