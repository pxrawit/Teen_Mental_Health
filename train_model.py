"""
=============================================================
  Teen Mental Health - Depression Prediction (SVM)
  รันผ่าน CMD: python train_model.py
=============================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Tahoma"  # รองรับภาษาไทย

# ============================================================
# STEP 1: โหลดข้อมูล
# ============================================================
print("=" * 60)
print("📌 STEP 1: โหลดข้อมูล (Load Data)")
print("=" * 60)

# โหลด CSV (มี header แล้ว)
df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

print(f"  ✅ ขนาดข้อมูล: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
print(f"\n📊 คอลัมน์ทั้งหมด:")
for col in df.columns:
    print(f"    • {col}")

print(f"\n📋 ตัวอย่างข้อมูล 5 แถวแรก:")
print(df.head().to_string(index=False))

print(f"\n📈 สถิติเบื้องต้น:")
print(df.describe().to_string())

# ============================================================
# STEP 2: สำรวจข้อมูล (EDA)
# ============================================================
print("\n" + "=" * 60)
print("📌 STEP 2: สำรวจข้อมูล (EDA)")
print("=" * 60)

print(f"\n  🔍 ค่า Missing:\n{df.isnull().sum()}")
print(f"\n  🎯 การกระจายของ Target (depression_label):")
print(df["depression_label"].value_counts().to_string())
print(f"  📊 สัดส่วน: {df['depression_label'].value_counts(normalize=True).to_dict()}")

# แยกประเภทคอลัมน์
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
print(f"\n  📝 Categorical Columns: {cat_cols}")
print(f"  🔢 Numerical Columns:   {num_cols}")

# สร้างกราฟ EDA
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Distribution of Target
axes[0, 0].bar(["No Depression (0)", "Depression (1)"],
               df["depression_label"].value_counts().values,
               color=["#28a745", "#dc3545"])
axes[0, 0].set_title("Distribution of Depression Label")
axes[0, 0].set_ylabel("Count")

# 2. Stress Level vs Depression
sns.boxplot(x="depression_label", y="stress_level", data=df, ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Stress Level vs Depression")

# 3. Anxiety Level vs Depression
sns.boxplot(x="depression_label", y="anxiety_level", data=df, ax=axes[0, 2], palette="Set2")
axes[0, 2].set_title("Anxiety Level vs Depression")

# 4. Addiction Level vs Depression
sns.boxplot(x="depression_label", y="addiction_level", data=df, ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("Addiction Level vs Depression")

# 5. Sleep Hours vs Depression
sns.boxplot(x="depression_label", y="sleep_hours", data=df, ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Sleep Hours vs Depression")

# 6. Social Media Hours vs Depression
sns.boxplot(x="depression_label", y="daily_social_media_hours", data=df, ax=axes[1, 2], palette="Set2")
axes[1, 2].set_title("Social Media Hours vs Depression")

plt.tight_layout()
plt.savefig("plots/eda_summary.png", dpi=150)
print("\n  📊 บันทึกกราฟ EDA → plots/eda_summary.png")

# ============================================================
# STEP 3: Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("📌 STEP 3: Preprocessing ข้อมูล")
print("=" * 60)

target_col = "depression_label"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"  🎯 Target: {target_col} (Binary: 0 = No Depression, 1 = Depression)")

# Encode Categorical Features
encoders = {}
categorical_features = ["gender", "platform_usage", "social_interaction_level"]

for col in categorical_features:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"  🔄 Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\n  📐 Features shape: {X.shape}")
print(f"  📝 Feature names:  {X.columns.tolist()}")

# แบ่ง Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  ✂️  Train size: {X_train.shape[0]}")
print(f"  ✂️  Test size:  {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  📏 Feature Scaling (StandardScaler) เสร็จสมบูรณ์")

# ============================================================
# STEP 4: เทรนโมเดล SVM
# ============================================================
print("\n" + "=" * 60)
print("📌 STEP 4: สร้างและเทรนโมเดล SVM")
print("=" * 60)

# SVM พื้นฐาน
svm_base = SVC(kernel="rbf", random_state=42)
svm_base.fit(X_train_scaled, y_train)
y_pred_base = svm_base.predict(X_test_scaled)
print(f"  🔰 SVM เบื้องต้น Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")

# GridSearchCV
print("\n  ⏳ กำลังทำ Hyperparameter Tuning (GridSearchCV)...")

param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["rbf", "linear", "poly"],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

grid_search = GridSearchCV(
    SVC(random_state=42, class_weight="balanced", probability=True),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train_scaled, y_train)

best_svm = grid_search.best_estimator_
print(f"  🏆 Best Parameters: {grid_search.best_params_}")
print(f"  🏆 Best CV F1-Score: {grid_search.best_score_:.4f}")

# ============================================================
# STEP 5: ประเมินผล
# ============================================================
print("\n" + "=" * 60)
print("📌 STEP 5: ประเมินโมเดล (Evaluation)")
print("=" * 60)

y_pred = best_svm.predict(X_test_scaled)
y_prob = best_svm.predict_proba(X_test_scaled)[:, 1]

print(f"\n  📊 Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  📊 F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"  📊 ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n  📋 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Depression (0)", "Depression (1)"]
))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Depression", "Depression"],
    yticklabels=["No Depression", "Depression"],
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - SVM Model")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
print("  📊 บันทึก Confusion Matrix → plots/confusion_matrix.png")

# Cross-Validation
cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5, scoring="accuracy")
print(f"\n  🔄 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# STEP 6: Feature Importance
# ============================================================
print("\n" + "=" * 60)
print("📌 STEP 6: Feature Analysis")
print("=" * 60)

svm_linear = SVC(kernel="linear", random_state=42, class_weight="balanced")
svm_linear.fit(X_train_scaled, y_train)

if hasattr(svm_linear, "coef_"):
    feature_importance = np.abs(svm_linear.coef_[0])
    importance_df = pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Importance": feature_importance
    }).sort_values("Importance", ascending=False)
    print("\n  📊 Feature Importance (Linear SVM):")
    print(importance_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Feature Importance (Linear SVM)")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    print("\n  📊 บันทึก Feature Importance → plots/feature_importance.png")

# ============================================================
# STEP 7: บันทึกโมเดล
# ============================================================
print("\n" + "=" * 60)
print("📌 STEP 7: บันทึกโมเดล (Save Model)")
print("=" * 60)

os.makedirs("saved_model", exist_ok=True)

joblib.dump(best_svm, "saved_model/svm_model.pkl")
print("  💾 บันทึก svm_model.pkl")

joblib.dump(scaler, "saved_model/scaler.pkl")
print("  💾 บันทึก scaler.pkl")

artifacts = {
    "feature_encoders": encoders,
    "feature_names": X.columns.tolist(),
    "best_params": grid_search.best_params_,
    "categorical_features": categorical_features,
    "target_classes": ["No Depression (0)", "Depression (1)"],
}
joblib.dump(artifacts, "saved_model/encoders.pkl")
print("  💾 บันทึก encoders.pkl")

# ============================================================
# STEP 8: ทดสอบทำนาย
# ============================================================
print("\n" + "=" * 60)
print("📌 STEP 8: ทดสอบทำนายตัวอย่างใหม่")
print("=" * 60)

sample = pd.DataFrame([{
    "age": 16,
    "gender": "female",
    "daily_social_media_hours": 7.5,
    "platform_usage": "TikTok",
    "sleep_hours": 5.0,
    "screen_time_before_sleep": 2.5,
    "academic_performance": 2.5,
    "physical_activity": 0.5,
    "social_interaction_level": "low",
    "stress_level": 8,
    "anxiety_level": 9,
    "addiction_level": 8,
}])

sample_processed = sample.copy()
for col in categorical_features:
    if col in sample_processed.columns and col in encoders:
        sample_processed[col] = encoders[col].transform(sample_processed[col].astype(str))

sample_scaled = scaler.transform(sample_processed)
prediction = best_svm.predict(sample_scaled)[0]
probability = best_svm.predict_proba(sample_scaled)[0]

print(f"\n  🧪 ตัวอย่าง: Age=16, Female, Social=7.5h, Stress=8, Anxiety=9")
print(f"  🎯 ทำนาย: {'Depression (1)' if prediction == 1 else 'No Depression (0)'}")
print(f"  📊 ความน่าจะเป็น: No Depression={probability[0]:.2%}, Depression={probability[1]:.2%}")

print("\n" + "=" * 60)
print("✅ เสร็จสมบูรณ์! โมเดลพร้อมใช้งาน")
print("   📁 โมเดล: saved_model/svm_model.pkl")
print("   🚀 รัน Streamlit: streamlit run app.py")
print("=" * 60)