import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ▶ 추가: TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np

# ---------------------------
# 1. 데이터 로드
# ---------------------------
df = pd.read_json("final_stage2_with_injury.json", lines=True)

# ---------------------------
# 2. 기본 컬럼 확인 및 전처리
# ---------------------------
print("컬럼 확인:", df.columns.tolist())

# injury_yn을 숫자로 변환 (y/n → 1/0)
df["injury_yn"] = df["injury_yn"].apply(lambda x: 1 if str(x).lower() == "y" else 0)

# time_slot(Q1~Q8)을 원-핫 인코딩(dummy)
df = pd.get_dummies(df, columns=["time_slot"], prefix="", prefix_sep="")

# ---------------------------
# 3. Feature set 정의
# ---------------------------
base_features = ["sent_score_model", "sent_score_lexicon", "slot_delta_sent_final", "injury_yn"]

# Extended: 시간구간 dummy(Q1~Q8) 포함
time_features = [c for c in df.columns if c.startswith("Q")]
target = "result"  # ✅ 경기 결과 필드명 변경 (기존 y → result)

X_base = df[base_features]
X_extended = df[base_features + time_features]
y = df[target]

# ---------------------------
# 4. Train/Test split
# ---------------------------
X_train_b, X_test_b, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
X_train_e, X_test_e, _, _ = train_test_split(X_extended, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------
# 5. 공통 평가 함수 (sklearn 모델용)
# ---------------------------
def evaluate_sklearn(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # 확률/결정함수 → AUC
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        proba = model.decision_function(X_test)
        proba = MinMaxScaler().fit_transform(proba.reshape(-1, 1)).ravel()
    else:
        proba = pred  # 최악의 경우 대체
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }

# ---------------------------
# 6. LSTM 전용 빌드/평가
# ---------------------------
def build_lstm(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),              # (1, n_features)
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

def evaluate_lstm(X_train, X_test, y_train, y_test, epochs=30, batch_size=32):
    # (n, n_features) → (n, 1, n_features)
    X_train_r = np.asarray(X_train, dtype="float32").reshape(-1, 1, X_train.shape[1])
    X_test_r  = np.asarray(X_test,  dtype="float32").reshape(-1, 1, X_test.shape[1])
    y_train_r = np.asarray(y_train, dtype="float32")
    y_test_r  = np.asarray(y_test,  dtype="float32")

    m = build_lstm((1, X_train.shape[1]))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    m.fit(X_train_r, y_train_r, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    proba = m.predict(X_test_r, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y_test_r, pred),
        "Precision": precision_score(y_test_r, pred),
        "Recall": recall_score(y_test_r, pred),
        "F1": f1_score(y_test_r, pred),
        "AUC": roc_auc_score(y_test_r, proba),
    }

# ---------------------------
# 7. 모델 정의 및 실행
# ---------------------------
models_cfg = {
    "XGBoost": ("sk", XGBClassifier(eval_metric='logloss', random_state=42)),
    "RandomForest": ("sk", RandomForestClassifier(random_state=42)),
    "LSTM": ("lstm", None),  # ✅ SVM → LSTM
}

results = []
for name, (kind, model) in models_cfg.items():
    if kind == "sk":
        r_base = evaluate_sklearn(model, X_train_b, X_test_b, y_train, y_test)
        r_ext  = evaluate_sklearn(model, X_train_e, X_test_e, y_train, y_test)
    else:  # LSTM
        r_base = evaluate_lstm(X_train_b, X_test_b, y_train, y_test)
        r_ext  = evaluate_lstm(X_train_e, X_test_e, y_train, y_test)
    results.append((name, "Base", r_base))
    results.append((name, "Extended", r_ext))

# ---------------------------
# 8. 결과 정리
# ---------------------------
df_results = pd.DataFrame([{"Model": m, "Type": t, **r} for m, t, r in results])
print("\n=== 모델별 성능 비교 결과 ===")
print(df_results.round(3))

# ---------------------------
# 9. 시각화 (기존 로직 유지, 모델 순서만 변경)
# ---------------------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

models_order = ["XGBoost", "RandomForest", "LSTM"]  # ✅ LSTM 반영

f1_base = []
f1_ext = []
auc_base = []
auc_ext = []
for m in models_order:
    f1_base.append(df_results[(df_results["Model"] == m) & (df_results["Type"] == "Base")]["F1"].values[0])
    f1_ext.append(df_results[(df_results["Model"] == m) & (df_results["Type"] == "Extended")]["F1"].values[0])
    auc_base.append(df_results[(df_results["Model"] == m) & (df_results["Type"] == "Base")]["AUC"].values[0])
    auc_ext.append(df_results[(df_results["Model"] == m) & (df_results["Type"] == "Extended")]["AUC"].values[0])

x = np.arange(len(models_order))
bar_width = 0.35

preferred_fonts = ["Malgun Gothic", "NanumGothic", "AppleGothic"]
for f in preferred_fonts:
    if f in [font.name for font in fm.fontManager.ttflist]:
        mpl.rcParams["font.family"] = f
        break
mpl.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(7,5))
plt.bar(x - bar_width/2, f1_base, width=bar_width, label='Base')
plt.bar(x + bar_width/2, f1_ext, width=bar_width, label='Extended')
plt.xticks(x, models_order, fontsize=11); plt.ylabel('F1-score', fontsize=12); plt.xlabel('모델', fontsize=12)
plt.title('모델별 F1-score 비교 (시간구간 변수 포함 여부)', fontsize=14)
plt.legend(); plt.tight_layout()
plt.savefig("model_f1_comparison.png", dpi=300); plt.close()

plt.figure(figsize=(7,5))
plt.bar(x - bar_width/2, auc_base, width=bar_width, label='Base')
plt.bar(x + bar_width/2, auc_ext, width=bar_width, label='Extended')
plt.xticks(x, models_order, fontsize=11); plt.ylabel('AUC', fontsize=12); plt.xlabel('모델', fontsize=12)
plt.title('모델별 AUC 비교 (시간구간 변수 포함 여부)', fontsize=14)
plt.legend(); plt.tight_layout()
plt.savefig("model_auc_comparison.png", dpi=300); plt.close()
