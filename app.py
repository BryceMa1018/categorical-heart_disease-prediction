import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_shap import st_shap

# -----------------------------------
# 1. Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 Heart Disease Risk Prediction System")
st.write(
    "Input patient clinical indicators below to estimate the probability of heart disease and view the SHAP feature contributions."
)
st.markdown("---")


# -----------------------------------
# 2. Load Model, Scaler & SHAP Explainer (Cached)
# -----------------------------------
@st.cache_resource
def load_ml_resources():
    # 加载模型、Scaler 和特征基准
    model = joblib.load("heart_model3.pkl")
    background = pd.read_csv("background_data.csv")
    scaler = joblib.load("scaler.pkl")

    # 显式指定 TreeExplainer
    try:
        explainer = shap.TreeExplainer(model, background)
    except Exception:
        explainer = shap.Explainer(model, background)

    return model, background, scaler, explainer


try:
    model, background, scaler, explainer = load_ml_resources()
except Exception as e:
    st.error(f"Error loading model, scaler or background data: {e}")
    st.stop()

# -----------------------------------
# 3. Patient Feature Inputs
# -----------------------------------
st.header("📋 Patient Clinical Information")

col1, col2, col3 = st.columns(3)
input_data = {}

with col1:
    input_data["sex"] = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    input_data["ca"] = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])

with col2:
    input_data["thalach"] = st.slider("Maximum Heart Rate (thalach)", 60, 220, 150)
    input_data["exang"] = st.selectbox(
        "Exercise-induced Angina (exang)",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col3:
    input_data["cp"] = st.selectbox(
        "Chest Pain Type (cp)",
        [1, 2, 3, 4],
        format_func=lambda x: f"Type {x}"
    )
    input_data["thal"] = st.selectbox(
        "Thalassemia (thal)",
        [3, 6, 7],
        format_func=lambda x: {3: "Normal (3)", 6: "Fixed Defect (6)", 7: "Reversible Defect (7)"}[x]
    )

# -----------------------------------
# 4. Feature Pipeline & One-Hot Realignment
# -----------------------------------
# 获取训练集完整的特征列名列表（对应你的 model3 只有那 9 个特征）
final_features = background.columns.tolist()

# 用全 0 初始化单行 DataFrame，彻底解决 NaN 丢失问题
X_input = pd.DataFrame(0, index=[0], columns=final_features)

# 填充非独热编码特征
X_input["sex"] = input_data["sex"]
X_input["ca"] = input_data["ca"]
X_input["exang"] = input_data["exang"]
X_input["thalach"] = input_data["thalach"]

# 动态安全匹配胸痛 CP 哑变量
cp_val = float(input_data["cp"])
for col in final_features:
    if col.startswith("cp_"):
        try:
            if float(col.split("_")[1]) == cp_val:
                X_input[col] = 1
        except (ValueError, IndexError):
            pass

# 动态安全匹配地中海贫血 Thal 哑变量
thal_val = float(input_data["thal"])
for col in final_features:
    if col.startswith("thal_"):
        try:
            if float(col.split("_")[1]) == thal_val:
                X_input[col] = 1
        except (ValueError, IndexError):
            pass

# 确保列顺序与训练集（background）完全、严格一致
X_input = X_input[final_features]

# -----------------------------------
# 【关键修复】精准单变量特征转换 (免特征名检验)
# -----------------------------------
try:
    # 尝试在 scaler 的特征列表中精准找到 'thalach' 的位置索引
    if hasattr(scaler, "feature_names_in_"):
        thalach_idx = list(scaler.feature_names_in_).index("thalach")
        mean_val = scaler.mean_[thalach_idx]
        scale_val = scaler.scale_[thalach_idx]
    else:
        # 如果训练时没有保存特征名，通过常见的心率均值区间自动捕捉位置
        matched_idx = np.where((scaler.mean_ > 100) & (scaler.mean_ < 180))[0]
        if len(matched_idx) > 0:
            idx = matched_idx[0]
            mean_val = scaler.mean_[idx]
            scale_val = scaler.scale_[idx]
        else:
            # 最后的兜底假设
            mean_val = scaler.mean_[-1]
            scale_val = scaler.scale_[-1]

    # 利用底层公式直接对单列数据洗澡：(X - mean) / std
    X_input["thalach"] = (X_input["thalach"] - mean_val) / scale_val

except Exception as e:
    # 极端异常下的兜底方案，如果以上寻找失败，打印警告并直接对单列使用自带 transform
    X_input[["thalach"]] = scaler.transform(X_input[["thalach"]])

# -----------------------------------
# 5. Prediction Execution
# -----------------------------------
st.markdown("---")
if st.button("🚀 Predict Heart Disease Risk", type="primary"):

    # 计算预测概率
    prob = model.predict_proba(X_input)[0, 1]

    # 展示结果指标
    st.subheader("📊 Prediction Result")
    st.metric(label="Predicted Heart Disease Risk", value=f"{prob:.1%}")

    # 分层临床风险提示
    if prob < 0.30:
        st.success("🟢 Low predicted cardiovascular risk.")
    elif prob < 0.70:
        st.warning("🟡 Moderate predicted cardiovascular risk.")
    else:
        st.error("🔴 High predicted cardiovascular risk.")

    # -----------------------------------
    # 6. SHAP Explanations
    # -----------------------------------
    st.subheader("💡 SHAP Feature Explanation")

    # 计算当前样本的 SHAP 值
    shap_values = explainer(X_input)

    # 6a. 交互式 Force Plot
    st.markdown("**1. Global Decision Push (Force Plot)**")
    st_shap(shap.plots.force(shap_values[0], matplotlib=False), height=140)

    # 6b. 静态条形图
    st.markdown("**2. Local Feature Contribution (Bar Plot)**")

    # 提取 SHAP 值并按绝对值大小排序
    shap_df = pd.DataFrame({
        "Feature": final_features,
        "SHAP": shap_values.values[0]
    })
    shap_df["abs_SHAP"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("abs_SHAP", ascending=True)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["tomato" if x > 0 else "steelblue" for x in shap_df["SHAP"]]

    sns.barplot(
        x="SHAP",
        y="Feature",
        data=shap_df,
        palette=colors,
        ax=ax
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("SHAP Value (Impact on Model Output)")
    ax.set_ylabel("Clinical Features")
    sns.despine(ax=ax, top=True, right=True)

    st.pyplot(fig)

    # -----------------------------------
    # 7. Clinical Recommendations
    # -----------------------------------
    st.subheader("📋 Clinical Recommendation")
    recommendations = []

    if input_data["thalach"] < 120:
        recommendations.append("- Reduced maximum heart rate may indicate impaired cardiac reserve.")
    if input_data["exang"] == 1:
        recommendations.append("- Exercise-induced angina suggests myocardial ischemia.")
    if input_data["ca"] >= 2:
        recommendations.append(
            "- Multiple major vessels involvement indicates elevated coronary artery disease burden.")
    if input_data["cp"] == 4:
        recommendations.append("- Asymptomatic chest pain pattern is strongly associated with cardiovascular risk.")
    if input_data["thal"] == 7:
        recommendations.append("- Reversible thalassemia defect may indicate ischemic myocardial abnormalities.")
    if prob >= 0.7:
        recommendations.append(
            "- **Urgent:** Recommend further cardiology evaluation and possible coronary angiography.")

    if len(recommendations) == 0:
        recommendations.append("- No major high-risk indicators detected. Maintain routine clinical follow-up.")

    for rec in recommendations:
        st.write(rec)

# -----------------------------------
# 8. Footer
# -----------------------------------
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This system is intended for clinical research and educational assistance purposes only. "
    "Final diagnostic and treatment decisions must be made by qualified healthcare professionals."
)