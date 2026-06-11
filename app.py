import subprocess, sys

# Tự động cài scikit-learn nếu chưa có (fix Streamlit Cloud)
try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "--quiet"])

import streamlit as st
import pandas as pd
import numpy as np
import io

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

# ─── Logo Agribank + Tiêu đề ──────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:18px;margin-bottom:6px;">
  <!-- Logo SVG Agribank -->
  <svg xmlns="http://www.w3.org/2000/svg" width="80" height="95" viewBox="0 0 160 190">
    <defs>
      <clipPath id="agri_green">
        <rect x="20" y="10" width="120" height="120" rx="14"/>
      </clipPath>
    </defs>
    <!-- Nền đỏ -->
    <rect x="0" y="0" width="160" height="160" rx="16" fill="#C0202A"/>
    <!-- Hình vuông xanh lá -->
    <rect x="20" y="10" width="120" height="120" rx="14" fill="#2E6B3E"/>
    <!-- Dải vàng chéo -->
    <polygon points="20,78 78,10 120,10 20,110"  fill="#C8972A" clip-path="url(#agri_green)"/>
    <polygon points="140,72 82,130 40,130 140,98" fill="#C8972A" clip-path="url(#agri_green)"/>
    <polygon points="20,78 140,10 140,72 20,110"  fill="#C8972A" clip-path="url(#agri_green)"/>
    <!-- Bông lúa trắng -->
    <g clip-path="url(#agri_green)" transform="translate(80,70) rotate(-42)">
      <line x1="0" y1="-42" x2="0" y2="32" stroke="white" stroke-width="3"/>
      <ellipse cx="-8"  cy="-32" rx="7" ry="4" fill="white" transform="rotate(-30,-8,-32)"/>
      <ellipse cx="8"   cy="-32" rx="7" ry="4" fill="white" transform="rotate(30,8,-32)"/>
      <ellipse cx="-9"  cy="-20" rx="7" ry="4" fill="white" transform="rotate(-25,-9,-20)"/>
      <ellipse cx="9"   cy="-20" rx="7" ry="4" fill="white" transform="rotate(25,9,-20)"/>
      <ellipse cx="-10" cy="-8"  rx="7" ry="4" fill="white" transform="rotate(-20,-10,-8)"/>
      <ellipse cx="10"  cy="-8"  rx="7" ry="4" fill="white" transform="rotate(20,10,-8)"/>
      <ellipse cx="-9"  cy="4"   rx="6" ry="3.5" fill="white" transform="rotate(-14,-9,4)"/>
      <ellipse cx="9"   cy="4"   rx="6" ry="3.5" fill="white" transform="rotate(14,9,4)"/>
      <ellipse cx="-7"  cy="16"  rx="5" ry="3"   fill="white" transform="rotate(-8,-7,16)"/>
      <ellipse cx="7"   cy="16"  rx="5" ry="3"   fill="white" transform="rotate(8,7,16)"/>
      <ellipse cx="0"   cy="-42" rx="5" ry="3.5" fill="white"/>
    </g>
    <!-- Chữ AGRIBANK trắng -->
    <text x="80" y="182" font-family="Arial Black,Arial,sans-serif"
          font-size="16" font-weight="900" fill="white"
          text-anchor="middle" letter-spacing="2">AGRIBANK</text>
  </svg>
  <!-- Tiêu đề app -->
  <div>
    <div style="font-size:24px;font-weight:900;color:#0D47A1;line-height:1.2;">
      🔍 Phát hiện Giao dịch Gian lận
    </div>
    <div style="font-size:13px;color:#1565C0;font-weight:700;margin-top:2px;">
      Huấn luyện &amp; so sánh mô hình · Dự báo đơn lẻ · Dự báo hàng loạt
    </div>
  </div>
</div>
<hr style="border:2px solid #1565C0;margin-top:4px;margin-bottom:16px;">
""", unsafe_allow_html=True)

# ─── CSS toàn cục: in đậm + màu xanh dương ───────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"], .stApp, .stMarkdown, p, span, div,
label, .stTextInput, .stNumberInput, .stSelectbox,
.stSlider, .stFileUploader, .stTabs,
.stExpander, .stSidebar, [data-testid="stSidebar"] * {
    color: #1565C0 !important;
    font-weight: 700 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #0D47A1 !important;
    font-weight: 900 !important;
}
button[data-baseweb="tab"] > div {
    color: #1565C0 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricValue"] > div {
    color: #1565C0 !important;
    font-weight: 700 !important;
}
.stButton > button {
    color: #ffffff !important;
    font-weight: 700 !important;
}
input, textarea, select {
    color: #1565C0 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

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
        "Decision Tree":       DecisionTreeClassifier(random_state=random_state),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=random_state),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred  = m.predict(X_test)
        y_prob  = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None
        report  = classification_report(y_test, y_pred, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        fpr, tpr, _ = roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)
        results[name] = dict(
            model=m, report=report, cm=cm, auc=auc,
            fpr=fpr, tpr=tpr,
        )
    return results, list(X.columns), df

# ─── Session state ────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False

if train_btn and uploaded:
    file_bytes = uploaded.read()
    with st.spinner("Đang huấn luyện..."):
        results, feat_names, df = train_models(file_bytes, test_size, random_state)
    st.session_state.update(trained=True, results=results, feat_names=feat_names, df=df)
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
        sample = pd.DataFrame(
            {f"X_{i}": [round(np.random.uniform(0, 1), 3) for _ in range(3)] for i in range(1, 15)}
        )
        sample["default"] = [0, 1, 0]
        st.dataframe(sample, use_container_width=True)
    st.stop()

# ─── Helpers vẽ bằng HTML/SVG thuần ──────────────────────────────────────────
def render_confusion_matrix(cm, title=""):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = cm.sum()
    def cell(val, bg):
        return (f'<td style="background:{bg};color:#fff;font-size:22px;font-weight:bold;'
                f'text-align:center;padding:18px 30px;border-radius:6px;">{val}</td>')
    html = f"""
    <p style="font-weight:600;margin-bottom:6px">{title}</p>
    <table style="border-collapse:separate;border-spacing:6px;">
      <tr><th></th><th style="padding:6px 30px;color:#555">Dự báo 0</th>
                   <th style="padding:6px 30px;color:#555">Dự báo 1</th></tr>
      <tr><th style="color:#555;padding-right:10px">Thực tế 0</th>
          {cell(tn,'#4e79a7')}{cell(fp,'#e15759')}</tr>
      <tr><th style="color:#555;padding-right:10px">Thực tế 1</th>
          {cell(fn,'#e15759')}{cell(tp,'#59a14f')}</tr>
    </table>
    <p style="font-size:12px;color:#888;margin-top:6px">
      TN={tn} &nbsp; FP={fp} &nbsp; FN={fn} &nbsp; TP={tp}
    </p>"""
    st.markdown(html, unsafe_allow_html=True)

def render_roc_svg(results_dict):
    """Vẽ ROC curve bằng SVG thuần, không cần thư viện ngoài."""
    W, H, PAD = 420, 300, 50
    inner_w = W - 2*PAD
    inner_h = H - 2*PAD
    colors = ["#4e79a7","#e15759","#59a14f"]
    lines = []
    legend_items = []
    for (name, res), color in zip(results_dict.items(), colors):
        if res["fpr"] is None:
            continue
        fpr_arr = np.array(res["fpr"])
        tpr_arr = np.array(res["tpr"])
        pts = " ".join(
            f"{PAD + fpr*inner_w:.1f},{PAD + inner_h - tpr*inner_h:.1f}"
            for fpr, tpr in zip(fpr_arr, tpr_arr)
        )
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_items.append((color, f"{name} AUC={res['auc']:.3f}"))

    # diagonal
    d_pts = f"{PAD},{PAD+inner_h} {PAD+inner_w},{PAD}"
    lines.append(f'<polyline points="{d_pts}" fill="none" stroke="#aaa" stroke-width="1" stroke-dasharray="5,4"/>')

    # axes
    axes = (f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{PAD+inner_h}" stroke="#555" stroke-width="1.5"/>'
            f'<line x1="{PAD}" y1="{PAD+inner_h}" x2="{PAD+inner_w}" y2="{PAD+inner_h}" stroke="#555" stroke-width="1.5"/>')
    axis_labels = (
        f'<text x="{W//2}" y="{H-6}" text-anchor="middle" font-size="12" fill="#555">FPR</text>'
        f'<text x="12" y="{H//2}" text-anchor="middle" font-size="12" fill="#555" '
        f'transform="rotate(-90,12,{H//2})">TPR</text>'
    )
    # tick labels
    ticks = ""
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        x = PAD + v*inner_w
        y = PAD + inner_h - v*inner_h
        ticks += (f'<text x="{PAD-4}" y="{y+4}" text-anchor="end" font-size="10" fill="#777">{v:.2f}</text>'
                  f'<text x="{x}" y="{PAD+inner_h+14}" text-anchor="middle" font-size="10" fill="#777">{v:.2f}</text>')

    # legend
    leg = ""
    for i, (color, label) in enumerate(legend_items):
        ly = PAD + 16 + i*20
        leg += (f'<rect x="{PAD+inner_w-160}" y="{ly-10}" width="12" height="12" fill="{color}"/>'
                f'<text x="{PAD+inner_w-144}" y="{ly}" font-size="11" fill="#333">{label}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
      <rect width="{W}" height="{H}" fill="#fafafa" rx="8"/>
      <text x="{W//2}" y="20" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Đường cong ROC</text>
      {axes}{axis_labels}{ticks}
      {''.join(lines)}
      {leg}
    </svg>"""
    st.markdown(svg, unsafe_allow_html=True)

def render_bar_svg(labels, values, colors_list, title="", width=500, height=260):
    """Bar chart ngang bằng SVG thuần."""
    PAD_L, PAD_R, PAD_T, PAD_B = 160, 30, 40, 30
    inner_w = width - PAD_L - PAD_R
    bar_h   = max(14, (height - PAD_T - PAD_B) // len(labels) - 6)
    max_val = max(values) if values else 1
    svg_bars = ""
    for i, (label, val, color) in enumerate(zip(labels, values, colors_list)):
        y = PAD_T + i * (bar_h + 6)
        bw = int(val / max_val * inner_w)
        svg_bars += (
            f'<text x="{PAD_L-6}" y="{y+bar_h//2+4}" text-anchor="end" font-size="11" fill="#333">{label}</text>'
            f'<rect x="{PAD_L}" y="{y}" width="{bw}" height="{bar_h}" fill="{color}" rx="3"/>'
            f'<text x="{PAD_L+bw+4}" y="{y+bar_h//2+4}" font-size="11" fill="#555">{val:.4f}</text>'
        )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<rect width="{width}" height="{height}" fill="#fafafa" rx="8"/>'
           f'<text x="{width//2}" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">{title}</text>'
           f'{svg_bars}</svg>')
    st.markdown(svg, unsafe_allow_html=True)

# ─── Lấy state ────────────────────────────────────────────────────────────────
results    = st.session_state.results
feat_names = st.session_state.feat_names
df         = st.session_state.df

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
            "Mô hình":           name,
            "Accuracy":          round(rep["accuracy"], 4),
            "Precision (fraud)": round(rep["1"]["precision"], 4),
            "Recall (fraud)":    round(rep["1"]["recall"], 4),
            "F1 (fraud)":        round(rep["1"]["f1-score"], 4),
            "AUC-ROC":           round(r["auc"], 4) if r["auc"] else 0.0,
        })
    comp_df = pd.DataFrame(rows).set_index("Mô hình")

    def highlight_max(s):
        try:
            is_max = s == s.max()
            return ["background-color:#d4edda;font-weight:bold" if v else "" for v in is_max]
        except Exception:
            return ["" for _ in s]

    st.dataframe(comp_df.style.apply(highlight_max).format("{:.4f}"), use_container_width=True)

    # Bar chart bằng SVG cho từng metric
    metric_colors = {
        "Accuracy":          "#4e79a7",
        "Precision (fraud)": "#f28e2b",
        "Recall (fraud)":    "#e15759",
        "F1 (fraud)":        "#76b7b2",
        "AUC-ROC":           "#59a14f",
    }
    for metric, color in metric_colors.items():
        render_bar_svg(
            labels=comp_df.index.tolist(),
            values=comp_df[metric].astype(float).tolist(),
            colors_list=[color] * len(comp_df),
            title=metric,
            width=460,
            height=110,
        )

    # Phân bố nhãn
    st.markdown("---")
    st.subheader("📊 Phân bố dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng giao dịch",  len(df))
    c2.metric("Bình thường (0)", int((df["default"] == 0).sum()))
    c3.metric("Gian lận (1)",    int((df["default"] == 1).sum()))

    counts = df["default"].value_counts().sort_index()
    render_bar_svg(
        labels=["Bình thường (0)", "Gian lận (1)"],
        values=counts.values.tolist(),
        colors_list=["#4e79a7", "#e15759"],
        title="Số lượng giao dịch theo nhãn",
        width=400, height=120,
    )

# ─── Tab 2: Chi tiết & ROC ────────────────────────────────────────────────────
with tab2:
    selected = st.selectbox("Chọn mô hình", list(results.keys()))
    r = results[selected]

    col_a, col_b = st.columns(2)
    with col_a:
        render_confusion_matrix(r["cm"], title=f"Ma trận nhầm lẫn – {selected}")
    with col_b:
        render_roc_svg(results)

    st.markdown("---")
    st.markdown("**Báo cáo phân loại chi tiết**")
    rep_df = pd.DataFrame(r["report"]).T.drop(columns=["support"], errors="ignore")
    st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

    model_obj = r["model"]
    if hasattr(model_obj, "feature_importances_"):
        st.markdown("**Mức độ quan trọng đặc trưng**")
        fi = pd.Series(model_obj.feature_importances_, index=feat_names).sort_values()
        render_bar_svg(
            labels=fi.index.tolist(),
            values=fi.values.tolist(),
            colors_list=["#4e79a7"] * len(fi),
            title=f"Feature Importance – {selected}",
            width=520,
            height=60 + len(fi) * 26,
        )

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
        target = col_l if i < half else col_r
        input_vals[feat] = target.number_input(
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
            render_bar_svg(
                labels=["Bình thường", "Gian lận"],
                values=[round(prob[0], 4), round(prob[1], 4)],
                colors_list=["#4e79a7", "#e15759"],
                title="Xác suất dự báo",
                width=420, height=110,
            )

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
            preds  = batch_model.predict(X_pred)
            probs  = (batch_model.predict_proba(X_pred)[:, 1]
                      if hasattr(batch_model, "predict_proba") else None)

            result_df = X_batch.copy()
            result_df["Dự báo (default)"] = preds
            if probs is not None:
                result_df["Xác suất gian lận"] = probs.round(4)

            fraud_n  = int(preds.sum())
            normal_n = len(preds) - fraud_n
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng giao dịch",     len(preds))
            c2.metric("Bình thường",         normal_n)
            c3.metric("Gian lận phát hiện",  fraud_n)

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
