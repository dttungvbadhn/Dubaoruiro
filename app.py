import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# ─── Cấu hình trang ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phát hiện Giao dịch Gian lận",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Phát hiện Giao dịch Gian lận")
st.markdown("Huấn luyện & so sánh mô hình · Dự báo đơn lẻ · Dự báo hàng loạt")

# ─── Sidebar: tải dữ liệu ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Dữ liệu huấn luyện")
    uploaded = st.file_uploader("Tải lên file CSV (có cột `default`)", type=["csv"])
    test_size = st.slider("Tỉ lệ tập Test (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random State", value=32, step=1)
    train_btn = st.button("🚀 Huấn luyện mô hình", type="primary")

# ─── Cache: tải & huấn luyện ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_resource(show_spinner=False)
def train_models(file_bytes, test_size, random_state):
    df = pd.read_csv(io.BytesIO(file_bytes))
    X = df.drop("default", axis=1)
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        fpr, tpr, _ = roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)
        results[name] = dict(
            model=m, report=report, cm=cm, auc=auc,
            fpr=fpr, tpr=tpr, feature_names=list(X.columns)
        )
    return results, X_test, y_test, list(X.columns), df

# ─── Session state ────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False

if train_btn and uploaded:
    file_bytes = uploaded.read()
    with st.spinner("Đang huấn luyện..."):
        results, X_test, y_test, feat_names, df = train_models(
            file_bytes, test_size, random_state
        )
    st.session_state.trained = True
    st.session_state.results = results
    st.session_state.feat_names = feat_names
    st.session_state.df = df
    st.sidebar.success("✅ Huấn luyện xong!")

elif train_btn and not uploaded:
    st.sidebar.warning("Vui lòng tải file CSV trước.")

# ─── Nội dung chính ───────────────────────────────────────────────────────────
if not st.session_state.trained:
    # Trang chào / hướng dẫn
    st.info("👈 Tải file CSV lên sidebar rồi nhấn **Huấn luyện mô hình** để bắt đầu.")
    with st.expander("📋 Định dạng CSV yêu cầu"):
        st.markdown(
            "File cần có **14 cột đặc trưng số** (`X_1` … `X_14`) "
            "và **1 cột nhãn** `default` (0 = bình thường, 1 = gian lận)."
        )
        sample = pd.DataFrame({
            f"X_{i}": [round(np.random.uniform(0, 1), 3) for _ in range(3)]
            for i in range(1, 15)
        })
        sample["default"] = [0, 1, 0]
        st.dataframe(sample, use_container_width=True)
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
results = st.session_state.results
feat_names = st.session_state.feat_names
df = st.session_state.df

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 So sánh mô hình", "📈 Chi tiết & ROC", "🔎 Dự báo đơn lẻ", "📦 Dự báo hàng loạt"]
)

# ─── Tab 1: So sánh ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Tổng quan hiệu suất các mô hình")

    metrics_rows = []
    for name, r in results.items():
        rep = r["report"]
        metrics_rows.append({
            "Mô hình": name,
            "Accuracy": round(rep["accuracy"], 4),
            "Precision (fraud)": round(rep["1"]["precision"], 4),
            "Recall (fraud)": round(rep["1"]["recall"], 4),
            "F1 (fraud)": round(rep["1"]["f1-score"], 4),
            "AUC-ROC": round(r["auc"], 4) if r["auc"] else "N/A",
        })
    comp_df = pd.DataFrame(metrics_rows).set_index("Mô hình")

    # Tô màu best per column
    def highlight_max(s):
        try:
            is_max = s == s.max()
            return ["background-color: #d4edda; font-weight:bold" if v else "" for v in is_max]
        except Exception:
            return ["" for _ in s]

    st.dataframe(
        comp_df.style.apply(highlight_max).format("{:.4f}"),
        use_container_width=True,
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(comp_df))
    w = 0.18
    cols = ["Accuracy", "Precision (fraud)", "Recall (fraud)", "F1 (fraud)", "AUC-ROC"]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    for i, (col, color) in enumerate(zip(cols, colors)):
        vals = comp_df[col].astype(float)
        ax.bar(x + i * w, vals, w, label=col, color=color)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(comp_df.index, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("So sánh hiệu suất mô hình")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Thống kê dữ liệu
    st.markdown("---")
    st.subheader("📊 Phân bố dữ liệu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng giao dịch", len(df))
    col2.metric("Bình thường (0)", int((df["default"] == 0).sum()))
    col3.metric("Gian lận (1)", int((df["default"] == 1).sum()))

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    counts = df["default"].value_counts().sort_index()
    ax2.bar(["Bình thường (0)", "Gian lận (1)"], counts.values,
            color=["#4e79a7", "#e15759"])
    ax2.set_title("Phân bố nhãn")
    ax2.grid(axis="y", alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

# ─── Tab 2: Chi tiết & ROC ─────────────────────────────────────────────────────
with tab2:
    selected_model = st.selectbox("Chọn mô hình", list(results.keys()))
    r = results[selected_model]

    col_a, col_b = st.columns(2)

    # Confusion matrix
    with col_a:
        st.markdown("**Ma trận nhầm lẫn**")
        cm = r["cm"]
        fig3, ax3 = plt.subplots(figsize=(4, 3.5))
        im = ax3.imshow(cm, cmap="Blues")
        ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
        ax3.set_xticklabels(["Dự báo 0", "Dự báo 1"])
        ax3.set_yticklabels(["Thực tế 0", "Thực tế 1"])
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax3)
        ax3.set_title(selected_model)
        st.pyplot(fig3)
        plt.close(fig3)

    # ROC curve
    with col_b:
        st.markdown("**Đường cong ROC**")
        fig4, ax4 = plt.subplots(figsize=(4, 3.5))
        for name, res in results.items():
            if res["fpr"] is not None:
                ax4.plot(res["fpr"], res["tpr"],
                         label=f"{name} (AUC={res['auc']:.3f})")
        ax4.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
        ax4.set_title("ROC – tất cả mô hình")
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

    # Classification report
    st.markdown("**Báo cáo phân loại chi tiết**")
    rep_df = pd.DataFrame(r["report"]).T.drop(columns=["support"], errors="ignore")
    st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

    # Feature importance (RF / DT)
    model_obj = r["model"]
    if hasattr(model_obj, "feature_importances_"):
        st.markdown("**Mức độ quan trọng đặc trưng**")
        fi = pd.Series(model_obj.feature_importances_, index=feat_names).sort_values(ascending=True)
        fig5, ax5 = plt.subplots(figsize=(6, max(3, len(fi) * 0.3)))
        fi.plot.barh(ax=ax5, color="#4e79a7")
        ax5.set_title(f"Feature Importance – {selected_model}")
        ax5.grid(axis="x", alpha=0.3)
        st.pyplot(fig5)
        plt.close(fig5)

# ─── Tab 3: Dự báo đơn lẻ ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Nhập thông tin một giao dịch")
    pred_model_name = st.selectbox("Mô hình dùng để dự báo", list(results.keys()), key="pred_sel")
    pred_model = results[pred_model_name]["model"]

    # Lấy min/max từ df để làm slider bounds
    cols_half = len(feat_names) // 2
    input_vals = {}
    col_l, col_r = st.columns(2)
    for i, feat in enumerate(feat_names):
        fmin = float(df[feat].min())
        fmax = float(df[feat].max())
        fmean = float(df[feat].mean())
        target_col = col_l if i < cols_half else col_r
        input_vals[feat] = target_col.number_input(
            feat, value=round(fmean, 6), format="%.6f", key=f"inp_{feat}"
        )

    if st.button("🔍 Dự báo giao dịch này", type="primary"):
        X_input = pd.DataFrame([input_vals])
        pred = pred_model.predict(X_input)[0]
        prob = pred_model.predict_proba(X_input)[0] if hasattr(pred_model, "predict_proba") else None

        if pred == 1:
            st.error(f"⚠️ **GIAN LẶN** – Xác suất gian lận: {prob[1]*100:.1f}%")
        else:
            st.success(f"✅ **BÌNH THƯỜNG** – Xác suất gian lận: {prob[1]*100:.1f}%" if prob is not None else "✅ Bình thường")

        if prob is not None:
            fig6, ax6 = plt.subplots(figsize=(5, 1.5))
            ax6.barh(["Bình thường", "Gian lận"], [prob[0], prob[1]],
                     color=["#4e79a7", "#e15759"])
            ax6.set_xlim(0, 1)
            ax6.set_title("Xác suất dự báo")
            ax6.grid(axis="x", alpha=0.3)
            st.pyplot(fig6)
            plt.close(fig6)

# ─── Tab 4: Dự báo hàng loạt ──────────────────────────────────────────────────
with tab4:
    st.subheader("Tải lên file CSV để dự báo hàng loạt")
    batch_model_name = st.selectbox("Mô hình", list(results.keys()), key="batch_sel")
    batch_model = results[batch_model_name]["model"]

    batch_file = st.file_uploader("File CSV (không cần cột `default`)", type=["csv"], key="batch_up")

    if batch_file:
        X_batch = pd.read_csv(batch_file)
        # Giữ đúng các cột đặc trưng nếu có thừa
        available = [f for f in feat_names if f in X_batch.columns]
        if len(available) < len(feat_names):
            st.warning(f"Thiếu cột: {set(feat_names) - set(available)}")
        else:
            X_pred_batch = X_batch[feat_names]
            preds = batch_model.predict(X_pred_batch)
            probs = batch_model.predict_proba(X_pred_batch)[:, 1] if hasattr(batch_model, "predict_proba") else None

            result_df = X_batch.copy()
            result_df["Dự báo (default)"] = preds
            if probs is not None:
                result_df["Xác suất gian lận"] = probs.round(4)

            fraud_count = int(preds.sum())
            normal_count = len(preds) - fraud_count
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng giao dịch", len(preds))
            c2.metric("Bình thường", normal_count)
            c3.metric("Gian lận phát hiện", fraud_count)

            st.dataframe(
                result_df.style.apply(
                    lambda row: ["background-color:#fde8e8" if row["Dự báo (default)"] == 1 else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
            )

            csv_out = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Tải kết quả CSV",
                data=csv_out,
                file_name="ket_qua_du_bao.csv",
                mime="text/csv",
            )
