import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
import plotly.express as px

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

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Dữ liệu huấn luyện")
    uploaded = st.file_uploader("Tải lên file CSV (có cột `default`)", type=["csv"])
    test_size = st.slider("Tỉ lệ tập Test (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random State", value=32, step=1)
    train_btn = st.button("🚀 Huấn luyện mô hình", type="primary")

# ─── Cache: huấn luyện ────────────────────────────────────────────────────────
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
    return results, list(X.columns), df

# ─── Session state ────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False

if train_btn and uploaded:
    file_bytes = uploaded.read()
    with st.spinner("Đang huấn luyện..."):
        results, feat_names, df = train_models(file_bytes, test_size, random_state)
    st.session_state.trained = True
    st.session_state.results = results
    st.session_state.feat_names = feat_names
    st.session_state.df = df
    st.sidebar.success("✅ Huấn luyện xong!")
elif train_btn and not uploaded:
    st.sidebar.warning("Vui lòng tải file CSV trước.")

# ─── Trang chờ ────────────────────────────────────────────────────────────────
if not st.session_state.trained:
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

# ─── Lấy state ────────────────────────────────────────────────────────────────
results   = st.session_state.results
feat_names = st.session_state.feat_names
df        = st.session_state.df

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 So sánh mô hình", "📈 Chi tiết & ROC", "🔎 Dự báo đơn lẻ", "📦 Dự báo hàng loạt"]
)

# ─── Tab 1: So sánh ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Tổng quan hiệu suất các mô hình")

    rows = []
    for name, r in results.items():
        rep = r["report"]
        rows.append({
            "Mô hình": name,
            "Accuracy":          round(rep["accuracy"], 4),
            "Precision (fraud)": round(rep["1"]["precision"], 4),
            "Recall (fraud)":    round(rep["1"]["recall"], 4),
            "F1 (fraud)":        round(rep["1"]["f1-score"], 4),
            "AUC-ROC":           round(r["auc"], 4) if r["auc"] else None,
        })
    comp_df = pd.DataFrame(rows).set_index("Mô hình")

    def highlight_max(s):
        try:
            is_max = s == s.max()
            return ["background-color:#d4edda;font-weight:bold" if v else "" for v in is_max]
        except Exception:
            return ["" for _ in s]

    st.dataframe(comp_df.style.apply(highlight_max).format("{:.4f}"), use_container_width=True)

    # Bar chart – Plotly
    metric_cols = ["Accuracy", "Precision (fraud)", "Recall (fraud)", "F1 (fraud)", "AUC-ROC"]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    fig_bar = go.Figure()
    for col, color in zip(metric_cols, colors):
        fig_bar.add_trace(go.Bar(
            name=col,
            x=comp_df.index.tolist(),
            y=comp_df[col].astype(float).tolist(),
            marker_color=color,
        ))
    fig_bar.update_layout(
        barmode="group", title="So sánh hiệu suất mô hình",
        yaxis=dict(range=[0, 1.1]), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Phân bố nhãn
    st.markdown("---")
    st.subheader("📊 Phân bố dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng giao dịch", len(df))
    c2.metric("Bình thường (0)", int((df["default"] == 0).sum()))
    c3.metric("Gian lận (1)",    int((df["default"] == 1).sum()))

    counts = df["default"].value_counts().sort_index()
    fig_pie = px.pie(
        values=counts.values,
        names=["Bình thường (0)", "Gian lận (1)"],
        color_discrete_sequence=["#4e79a7", "#e15759"],
        title="Tỉ lệ nhãn",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ─── Tab 2: Chi tiết & ROC ────────────────────────────────────────────────────
with tab2:
    selected = st.selectbox("Chọn mô hình", list(results.keys()))
    r = results[selected]

    col_a, col_b = st.columns(2)

    # Confusion matrix – Plotly heatmap
    with col_a:
        st.markdown("**Ma trận nhầm lẫn**")
        cm = r["cm"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Dự báo 0", "Dự báo 1"],
            y=["Thực tế 0", "Thực tế 1"],
            title=selected,
        )
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC – Plotly
    with col_b:
        st.markdown("**Đường cong ROC**")
        fig_roc = go.Figure()
        for name, res in results.items():
            if res["fpr"] is not None:
                fig_roc.add_trace(go.Scatter(
                    x=res["fpr"].tolist(), y=res["tpr"].tolist(),
                    mode="lines",
                    name=f"{name} (AUC={res['auc']:.3f})",
                ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"), name="Random",
        ))
        fig_roc.update_layout(
            title="ROC – tất cả mô hình",
            xaxis_title="FPR", yaxis_title="TPR", height=320,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Classification report
    st.markdown("**Báo cáo phân loại chi tiết**")
    rep_df = pd.DataFrame(r["report"]).T.drop(columns=["support"], errors="ignore")
    st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

    # Feature importance
    model_obj = r["model"]
    if hasattr(model_obj, "feature_importances_"):
        st.markdown("**Mức độ quan trọng đặc trưng**")
        fi = pd.Series(model_obj.feature_importances_, index=feat_names).sort_values()
        fig_fi = px.bar(
            x=fi.values, y=fi.index, orientation="h",
            title=f"Feature Importance – {selected}",
            color_discrete_sequence=["#4e79a7"],
        )
        fig_fi.update_layout(height=max(300, len(fi) * 28))
        st.plotly_chart(fig_fi, use_container_width=True)

# ─── Tab 3: Dự báo đơn lẻ ────────────────────────────────────────────────────
with tab3:
    st.subheader("Nhập thông tin một giao dịch")
    pred_model_name = st.selectbox("Mô hình dùng để dự báo", list(results.keys()), key="pred_sel")
    pred_model = results[pred_model_name]["model"]

    input_vals = {}
    col_l, col_r = st.columns(2)
    half = len(feat_names) // 2
    for i, feat in enumerate(feat_names):
        fmean = float(df[feat].mean())
        target_col = col_l if i < half else col_r
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
            p_str = f" – Xác suất gian lận: {prob[1]*100:.1f}%" if prob is not None else ""
            st.success(f"✅ **BÌNH THƯỜNG**{p_str}")

        if prob is not None:
            fig_prob = go.Figure(go.Bar(
                x=[prob[0], prob[1]],
                y=["Bình thường", "Gian lận"],
                orientation="h",
                marker_color=["#4e79a7", "#e15759"],
            ))
            fig_prob.update_layout(
                title="Xác suất dự báo", xaxis=dict(range=[0, 1]), height=200,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

# ─── Tab 4: Dự báo hàng loạt ─────────────────────────────────────────────────
with tab4:
    st.subheader("Tải lên file CSV để dự báo hàng loạt")
    batch_model_name = st.selectbox("Mô hình", list(results.keys()), key="batch_sel")
    batch_model = results[batch_model_name]["model"]

    batch_file = st.file_uploader("File CSV (không cần cột `default`)", type=["csv"], key="batch_up")

    if batch_file:
        X_batch = pd.read_csv(batch_file)
        missing = set(feat_names) - set(X_batch.columns)
        if missing:
            st.warning(f"Thiếu cột: {missing}")
        else:
            X_pred = X_batch[feat_names]
            preds = batch_model.predict(X_pred)
            probs = batch_model.predict_proba(X_pred)[:, 1] if hasattr(batch_model, "predict_proba") else None

            result_df = X_batch.copy()
            result_df["Dự báo (default)"] = preds
            if probs is not None:
                result_df["Xác suất gian lận"] = probs.round(4)

            fraud_n  = int(preds.sum())
            normal_n = len(preds) - fraud_n
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng giao dịch", len(preds))
            c2.metric("Bình thường",    normal_n)
            c3.metric("Gian lận phát hiện", fraud_n)

            st.dataframe(
                result_df.style.apply(
                    lambda row: [
                        "background-color:#fde8e8" if row["Dự báo (default)"] == 1 else ""
                        for _ in row
                    ],
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
