"""
=============================================================
  Teen Depression Prediction - Streamlit Web App
  รันผ่าน CMD: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="🧠 Teen Depression Predictor",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-safe {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        font-size: 1.5rem;
        font-weight: bold;
        color: #155724;
        text-align: center;
    }
    .risk-depression {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        font-size: 1.5rem;
        font-weight: bold;
        color: #721c24;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = "saved_model/svm_model.pkl"
    scaler_path = "saved_model/scaler.pkl"
    encoders_path = "saved_model/encoders.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    artifacts = joblib.load(encoders_path)
    return model, scaler, artifacts


model, scaler, artifacts = load_model()

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=80)
    st.title("📋 ข้อมูลโมเดล")
    st.markdown("---")

    if artifacts:
        st.markdown("**🔧 Algorithm:** SVM (Support Vector Machine)")
        st.markdown(f"**⚙️ Best Params:**")
        for k, v in artifacts.get("best_params", {}).items():
            st.code(f"{k}: {v}")
        st.markdown(f"**📝 Features:** {len(artifacts.get('feature_names', []))}")
        st.markdown(f"**🎯 Classes:** {artifacts.get('target_classes', [])}")
    else:
        st.error("❌ ไม่พบโมเดล! กรุณารัน `python train_model.py` ก่อน")

# MAIN
st.markdown('<h1 class="main-header">🧠 Teen Depression Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ระบบทำนายความเสี่ยงภาวะซึมเศร้าในวัยรุ่น ด้วย SVM Machine Learning</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ **ไม่พบไฟล์โมเดล!** กรุณารัน `python train_model.py` ก่อน")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 ทำนายผล", "📊 วิเคราะห์หลายราย", "ℹ️ เกี่ยวกับโมเดล"])

# ====================== TAB 1 ======================
with tab1:
    st.markdown("### 📝 กรอกข้อมูลเพื่อทำนาย")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 ข้อมูลส่วนบุคคล")
        age = st.slider("🎂 อายุ (ปี)", 13, 19, 15)
        gender = st.selectbox("⚧ เพศ", ["male", "female"])
        platform = st.selectbox("📱 แพลตฟอร์มที่ใช้", ["TikTok", "Instagram", "Both"])
        social_interaction = st.selectbox(
            "🤝 ระดับปฏิสัมพันธ์ทางสังคม",
            ["low", "medium", "high"]
        )

        st.markdown("#### ⏰ พฤติกรรมการใช้")
        social_media_hours = st.slider("📱 ชั่วโมงใช้โซเชียลมีเดีย/วัน", 0.0, 10.0, 5.0, 0.1)
        screen_before_sleep = st.slider("📺 ใช้หน้าจอก่อนนอน (ชม.)", 0.0, 3.0, 1.5, 0.1)
        sleep_hours = st.slider("😴 ชั่วโมงการนอน/วัน", 3.0, 10.0, 7.0, 0.1)
        physical_activity = st.slider("🏃 กิจกรรมทางกาย (ชม./วัน)", 0.0, 3.0, 1.0, 0.1)

    with col2:
        st.markdown("#### 📈 ผลการเรียนและอารมณ์")
        academic_perf = st.slider("📚 ผลการเรียน (1-5)", 1.0, 5.0, 3.0, 0.1)

        st.markdown("#### 💭 ระดับความเครียดและอารมณ์")
        stress_level = st.slider("😰 ระดับความเครียด (1-10)", 1, 10, 5)
        anxiety_level = st.slider("😟 ระดับความกังวล (1-10)", 1, 10, 5)
        addiction_level = st.slider("📱 ระดับการติดโซเชียล (1-10)", 1, 10, 5)

    st.markdown("---")

    if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "daily_social_media_hours": social_media_hours,
            "platform_usage": platform,
            "sleep_hours": sleep_hours,
            "screen_time_before_sleep": screen_before_sleep,
            "academic_performance": academic_perf,
            "physical_activity": physical_activity,
            "social_interaction_level": social_interaction,
            "stress_level": stress_level,
            "anxiety_level": anxiety_level,
            "addiction_level": addiction_level,
        }])

        input_processed = input_data.copy()
        for col, le in artifacts["feature_encoders"].items():
            if col in input_processed.columns:
                input_processed[col] = le.transform(input_processed[col].astype(str))

        input_scaled = scaler.transform(input_processed)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("### 🎯 ผลการทำนาย")

        if prediction == 1:
            st.markdown(
                '<div class="risk-depression">🔴 มีความเสี่ยงภาวะซึมเศร้า (Depression Risk)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="risk-safe">🟢 ไม่มีความเสี่ยงภาวะซึมเศร้า (No Depression Risk)</div>',
                unsafe_allow_html=True
            )

        # Probability
        st.markdown("#### 📊 ความน่าจะเป็น")
        prob_df = pd.DataFrame({
            "Class": ["No Depression", "Depression"],
            "Probability": probability
        })
        fig = px.bar(
            prob_df, x="Class", y="Probability",
            color="Class",
            color_discrete_map={"No Depression": "#28a745", "Depression": "#dc3545"},
            text_auto=".1%"
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Radar Chart
        st.markdown("#### 🕸️ Radar Chart - ภาพรวมปัจจัย")
        radar_data = {
            "ปัจจัย": [
                "Social Media", "Screen Before Sleep", "Sleep",
                "Physical Activity", "Academic",
                "Stress", "Anxiety", "Addiction"
            ],
            "คะแนน": [
                social_media_hours / 10, screen_before_sleep / 3, sleep_hours / 10,
                physical_activity / 3, academic_perf / 5,
                stress_level / 10, anxiety_level / 10, addiction_level / 10
            ]
        }
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data["คะแนน"] + [radar_data["คะแนน"][0]],
            theta=radar_data["ปัจจัย"] + [radar_data["ปัจจัย"][0]],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.3)",
            line=dict(color="#1f77b4")
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)

        # คำแนะนำ
        st.markdown("### 💡 คำแนะนำ")
        if prediction == 1:
            st.warning("🚨 **ควรพบผู้เชี่ยวชาญด้านสุขภาพจิตโดยเร็ว**")
            st.markdown("""
            - ลดเวลาใช้โซเชียลมีเดียลงอย่างจริงจัง
            - เพิ่มชั่วโมงการนอนหลับให้ได้อย่างน้อย 7-8 ชม.
            - ออกกำลังกายสม่ำเสมอ
            - พูดคุยกับคนที่ไว้ใจเกี่ยวกับความรู้สึก
            - **สายด่วนสุขภาพจิต: 1323 (กรมสุขภาพจิต)**
            """)
        else:
            st.success("✅ สุขภาพจิตอยู่ในเกณฑ์ดี รักษากิจวัตรที่ดีไว้")
            st.markdown("""
            - หมั่นออกกำลังกายและนอนหลับให้เพียงพอ
            - รักษาความสัมพันธ์ที่ดีกับครอบครัวและเพื่อน
            - ใช้โซเชียลมีเดียอย่างมีสติ
            """)

# ====================== TAB 2 ======================
with tab2:
    st.markdown("### 📊 อัปโหลดไฟล์ CSV เพื่อทำนายหลายราย")

    st.markdown("""
    **รูปแบบ CSV ที่ต้องมีคอลัมน์:**
    `age, gender, daily_social_media_hours, platform_usage, sleep_hours,
    screen_time_before_sleep, academic_performance, physical_activity,
    social_interaction_level, stress_level, anxiety_level, addiction_level`
    """)

    uploaded_file = st.file_uploader("📁 เลือกไฟล์ CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown("#### 📋 ตัวอย่างข้อมูลที่อัปโหลด")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🔮 ทำนายทั้งหมด", type="primary"):
            try:
                batch_processed = batch_df.copy()

                for col, le in artifacts["feature_encoders"].items():
                    if col in batch_processed.columns:
                        batch_processed[col] = le.transform(batch_processed[col].astype(str))

                batch_scaled = scaler.transform(batch_processed)
                predictions = model.predict(batch_scaled)
                probabilities = model.predict_proba(batch_scaled)

                batch_df["Predicted_Label"] = predictions
                batch_df["Predicted_Class"] = np.where(
                    predictions == 1, "Depression", "No Depression"
                )
                batch_df["Depression_Probability"] = probabilities[:, 1]

                st.success(f"✅ ทำนายสำเร็จ {len(batch_df)} ราย")
                st.dataframe(batch_df, use_container_width=True)

                # สรุป
                st.markdown("#### 📊 สรุปผลการทำนาย")
                summary = batch_df["Predicted_Class"].value_counts().reset_index()
                summary.columns = ["Class", "Count"]

                fig_pie = px.pie(
                    summary, values="Count", names="Class",
                    color="Class",
                    color_discrete_map={"No Depression": "#28a745", "Depression": "#dc3545"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                csv_result = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 ดาวน์โหลดผลทำนาย (CSV)",
                    data=csv_result,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                st.exception(e)

# ====================== TAB 3 ======================
with tab3:
    st.markdown("### ℹ️ เกี่ยวกับโมเดล")

    st.markdown("""
    #### 🧠 โมเดล SVM (Support Vector Machine)

    **SVM** คืออัลกอริทึม Machine Learning ที่ใช้สำหรับ Binary Classification
    โดยค้นหา Hyperplane ที่แบ่งกลุ่มข้อมูลได้ดีที่สุด

    #### 🎯 Target Variable
    - **0** = ไม่มีความเสี่ยงภาวะซึมเศร้า (No Depression)
    - **1** = มีความเสี่ยงภาวะซึมเศร้า (Depression)

    #### ⚙️ กระบวนการทำงาน

    | ขั้นตอน | รายละเอียด |
    |---------|------------|
    | 1. Data Loading | โหลดข้อมูล CSV |
    | 2. EDA | สำรวจข้อมูลและสร้างกราฟ |
    | 3. Preprocessing | Encode categorical, Scale numerical |
    | 4. Train/Test Split | แบ่งข้อมูล 80/20 |
    | 5. GridSearchCV | หา Hyperparameters ที่ดีที่สุด |
    | 6. Training | เทรน SVM ด้วย best params |
    | 7. Evaluation | Accuracy, F1, ROC-AUC, Confusion Matrix |
    | 8. Save Model | บันทึกด้วย joblib |
    """)

    if os.path.exists("plots/confusion_matrix.png"):
        st.markdown("#### 📊 Confusion Matrix (จาก Training)")
        st.image("plots/confusion_matrix.png")

    if os.path.exists("plots/feature_importance.png"):
        st.markdown("#### 📊 Feature Importance")
        st.image("plots/feature_importance.png")

    if os.path.exists("plots/eda_summary.png"):
        st.markdown("#### 📊 EDA Summary")
        st.image("plots/eda_summary.png")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999;'>"
    "🧠 Teen Depression Risk Predictor | Built with SVM + Streamlit | "
    "⚠️ สำหรับการศึกษาเท่านั้น"
    "</div>",
    unsafe_allow_html=True
)