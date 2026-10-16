"""
Bina.az Price Predictor — Enhanced Streamlit UI
================================================
Design: dark-navy nav + hero  /  light-gray body  /  white cards  /  teal accents.
Navigation uses URL query-params  (?page=data | predict | models | error).

DEMO MODE: If the dataset / trained models are unavailable (e.g. Streamlit Cloud),
           the app shows all static results, tables, and figures that were committed
           to the repo, with a friendly notice on the Predict tab.

Run locally:
    streamlit run streamlit_app.py
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ── src imports (graceful on any error) ───────────────────────────────────────
_SRC_OK  = False
_SRC_ERR = ""
try:
    from src import data_prep as dp, serve   # noqa: F401
    _SRC_OK = True
except Exception as _e:
    _SRC_ERR = str(_e)

TABLES = Path("report/tables")
FIGS   = Path("report/figures")

# ── Page config (MUST be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Bina.az Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
#MainMenu, footer, header { visibility: hidden; height: 0; overflow: hidden; }
[data-testid="stSidebar"]  { display: none !important; }
[data-testid="stAppViewContainer"] { background: #f1f5f9; }
.block-container { padding: 0 2rem 4rem !important; max-width: 100% !important; }

/* nav */
.nav-wrap {
    display: flex; align-items: center; gap: 1.5rem;
    background: #0f172a; padding: 0 4.5rem; margin: 0 -2rem;
    min-height: 64px; box-shadow: 0 2px 16px rgba(0,0,0,.45);
    position: sticky; top: 0; z-index: 1000;
}
.nav-logo { display: flex; align-items: center; gap: 10px; font-weight: 700;
            color: white; font-size: 1rem; flex-shrink: 0; }
.nav-icon { width: 34px; height: 34px; background: #0d9488; border-radius: 8px;
            display: grid; place-items: center; font-size: 1.1rem; flex-shrink: 0; }
.nav-links { display: flex; gap: 4px; margin-left: auto; flex-wrap: wrap; padding: 12px 0; }
.nav-lnk   { padding: 6px 18px; border-radius: 20px; color: rgba(255,255,255,.6);
             font-size: .875rem; font-weight: 500; text-decoration: none;
             border: 1px solid transparent; white-space: nowrap; transition: .15s; }
.nav-lnk:hover  { color: white; }
.nav-lnk.active { color: white; border-color: rgba(255,255,255,.75); background: rgba(255,255,255,.08); }

/* hero */
.hero { background: #0f172a; padding: 3rem 4.5rem 2.5rem;
        margin: 0 -2rem; border-bottom: 3px solid #0d9488; }
.h-label { color: #0d9488; font-size: .7rem; font-weight: 700;
           letter-spacing: .15em; text-transform: uppercase; margin-bottom: .6rem; }
.h-title { color: white; font-size: 2.4rem; font-weight: 800; line-height: 1.15; margin: 0 0 .5rem; }
.h-sub   { color: rgba(255,255,255,.55); font-size: .93rem; margin-bottom: 1.75rem; }
.stat-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.s-card { background: #1e293b; border-radius: 12px; padding: 1.15rem 1.5rem;
          flex: 1; min-width: 130px; border: 1px solid rgba(255,255,255,.06); }
.s-val  { color: white; font-size: 2rem; font-weight: 700; line-height: 1; }
.s-lbl  { color: rgba(255,255,255,.45); font-size: .78rem; margin-top: .3rem; }

/* content */
.cont { padding: 2rem 2.5rem 0; margin: 0 -2rem; }

/* cards */
.card { background: white; border-radius: 14px; padding: 1.75rem 2rem;
        margin-bottom: 1.25rem; box-shadow: 0 1px 5px rgba(0,0,0,.07);
        border: 1px solid rgba(0,0,0,.05); animation: fadeUp .38s ease both; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.c-title { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: .2rem; }
.c-sub   { font-size: .82rem; color: #64748b; margin-bottom: 1.2rem; }

/* form as card */
[data-testid="stForm"] {
    background: white !important; border-radius: 14px !important;
    padding: 1.75rem 2rem !important; border: 1px solid rgba(0,0,0,.05) !important;
    box-shadow: 0 1px 5px rgba(0,0,0,.07) !important; animation: fadeUp .38s ease both;
}

/* prediction */
.price-big  { font-size: 3rem; font-weight: 800; letter-spacing: -.03em; color: #0f172a; line-height: 1; }
.price-unit { font-size: 1.15rem; color: #94a3b8; font-weight: 400; }
.price-sqm  { font-size: .85rem; color: #64748b; margin-top: .25rem; }
.badge      { display: inline-block; border-radius: 20px; padding: 5px 18px;
              font-weight: 600; font-size: .88rem; margin: .75rem 0 .5rem; }
.badge-pre  { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.badge-std  { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; }
.prob-track { height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; margin: .5rem 0; }
.prob-fill  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #0d9488, #d97706); }
.prob-lbl   { font-size: .8rem; color: #64748b; }

/* tables */
.rtable { width: 100%; border-collapse: collapse; font-size: .875rem; }
.rtable th { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
             color: #94a3b8; padding: .55rem .8rem; border-bottom: 1px solid #e2e8f0; text-align: right; }
.rtable th:first-child { text-align: left; }
.rtable td { padding: .6rem .8rem; border-bottom: 1px solid #f8fafc; color: #334155; text-align: right; }
.rtable td:first-child { text-align: left; font-weight: 500; color: #0f172a; }
.rtable tr:last-child td { border-bottom: none; }
.rtable tr:hover td { background: #f8fafc; }

/* confusion matrix */
.cm-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }
.cm-cell { border-radius: 10px; text-align: center; padding: 1.5rem .5rem; font-size: 1.75rem; font-weight: 700; }
.cm-tp   { background: #0d9488; color: white; }
.cm-tn   { background: #0d9488cc; color: white; }
.cm-fp   { background: #f1f5f9; color: #94a3b8; }
.cm-fn   { background: #f1f5f9; color: #94a3b8; }
.cm-sub  { display: block; font-size: .7rem; font-weight: 500; opacity: .82; margin-top: .2rem; }

/* demo banner */
.demo-banner { background: #1e293b; border-left: 4px solid #0d9488; border-radius: 0 10px 10px 0;
               padding: .9rem 1.25rem; color: rgba(255,255,255,.85); font-size: .88rem; margin-bottom: 1.25rem; }

/* notice boxes */
.box-warn { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px;
            padding: .75rem 1rem; color: #92400e; font-size: .84rem; margin: .5rem 0; }
.box-info { background: #f0f9ff; border: 1px solid #7dd3fc; border-radius: 8px;
            padding: .75rem 1rem; color: #075985; font-size: .84rem; margin: .5rem 0; }
.box-err  { background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px;
            padding: .75rem 1rem; color: #991b1b; font-size: .84rem; margin: .5rem 0; }

/* primary button */
.stButton > button[kind="primary"] {
    background: #0d9488 !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 1rem !important; width: 100% !important;
    transition: background .2s, box-shadow .2s, transform .15s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #0b7a83 !important;
    box-shadow: 0 4px 16px rgba(13,148,136,.4) !important;
    transform: translateY(-1px) !important;
}
[data-baseweb="select"] { border-radius: 8px !important; }
[data-baseweb="input"]  { border-radius: 8px !important; }
[data-testid="stImage"] img { border-radius: 8px; }
[data-testid="stHorizontalBlock"] { padding: 0 2.5rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTING
# ══════════════════════════════════════════════════════════════════════════════
PAGES = {"data": "Data", "predict": "Predict", "models": "Models", "error": "Error analysis"}

try:
    _raw  = st.query_params.get("page", "predict")
    _page = _raw[0] if isinstance(_raw, (list, tuple)) else str(_raw)
    page  = _page if _page in PAGES else "predict"
except Exception:
    page = "predict"

# ══════════════════════════════════════════════════════════════════════════════
#  NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
_nav_links = "".join(
    '<a href="?page={k}" class="nav-lnk{a}">{v}</a>'.format(
        k=k, a=" active" if k == page else "", v=v
    )
    for k, v in PAGES.items()
)
st.markdown(
    '<div class="nav-wrap">'
    '<div class="nav-logo"><div class="nav-icon">🏠</div>Bina.az Price Predictor</div>'
    '<div class="nav-links">' + _nav_links + '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  BUNDLE  (returns None gracefully if dataset missing — demo mode)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models…")
def get_bundle():
    if not _SRC_OK:
        return None
    try:
        if serve.BUNDLE_PATH.exists():
            try:
                return serve.load_bundle()
            except Exception:
                pass
        b = serve.build_bundle(dp.clean(dp.load_raw(), verbose=False))
        serve.save_bundle(b)
        return b
    except Exception:
        return None   # dataset missing → demo mode, no crash


bundle = get_bundle() if _SRC_OK else None
_DEMO  = (bundle is None)   # True on Streamlit Cloud without dataset

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hero(label: str, title: str, sub: str, stats: list) -> None:
    cards = "".join(
        '<div class="s-card"><div class="s-val">{v}</div><div class="s-lbl">{l}</div></div>'.format(v=v, l=l)
        for v, l in stats
    )
    st.markdown(
        '<div class="hero">'
        '<div class="h-label">' + label + '</div>'
        '<div class="h-title">' + title + '</div>'
        '<div class="h-sub">'   + sub   + '</div>'
        '<div class="stat-row">' + cards + '</div></div>',
        unsafe_allow_html=True,
    )


def demo_banner() -> None:
    st.markdown(
        '<div class="demo-banner">'
        '📊 <strong>Demo mode</strong> — running without the live dataset. '
        'All benchmark results, figures, and tables come from the committed <code>report/</code> files. '
        'Clone the repo and add <code>data/bina_az_sale.csv</code> to enable live predictions.'
        '</div>',
        unsafe_allow_html=True,
    )


def show_fig(name: str, caption: str = "") -> None:
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def md_to_table(md: str) -> str:
    rows = [ln.strip() for ln in md.strip().splitlines()
            if ln.strip() and ln.strip().startswith("|")]
    if len(rows) < 3:
        return "<p style='font-size:.85rem;color:#64748b'>" + md + "</p>"
    headers = [h.strip() for h in rows[0].split("|")[1:-1]]
    th    = "".join("<th>" + h + "</th>" for h in headers)
    tbody = ""
    for row in rows[2:]:
        cells = [c.strip() for c in row.split("|")[1:-1]]
        if cells and any(cells):
            tbody += "<tr>" + "".join("<td>" + c + "</td>" for c in cells) + "</tr>"
    return '<table class="rtable"><thead><tr>' + th + '</tr></thead><tbody>' + tbody + '</tbody></table>'


def read_table(name: str) -> str:
    p = TABLES / name
    return md_to_table(p.read_text(encoding="utf-8")) if p.exists() else ""


def _no_results() -> str:
    return '<div class="box-info">Run <code>python -m src.run_all</code> locally and commit the results.</div>'


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA
# ══════════════════════════════════════════════════════════════════════════════
if page == "data":
    hero(
        "DATASET",
        "Bina.az real estate listings",
        "63,908 cleaned rows · 164 encoded features · Oct 2024 Kaggle snapshot.",
        [("63,908", "cleaned listings"), ("164", "encoded features"), ("210k AZN", "tier threshold")],
    )
    st.markdown('<div class="cont">', unsafe_allow_html=True)
    if _DEMO:
        demo_banner()

    st.markdown("""
<div class="card">
  <div class="c-title">Data pipeline</div>
  <div class="c-sub">Cleaning → leakage removal → stratified split → per-fold standardisation</div>
  <table class="rtable">
    <thead><tr><th>Step</th><th>Detail</th></tr></thead>
    <tbody>
      <tr><td>Source</td><td>Kaggle <em>binaaz-sale-project</em> by sehriyarmemmedli</td></tr>
      <tr><td>Raw rows</td><td>~150,000</td></tr>
      <tr><td>After cleaning</td><td>63,908</td></tr>
      <tr><td>Leakage dropped</td><td>unit_price, total_price, temporal indicators</td></tr>
      <tr><td>Split (74 / 16 / 10 %)</td><td>44,732 train · 9,588 val · 9,588 test</td></tr>
      <tr><td>Standardisation</td><td>μ / σ from train only — applied to val &amp; test</td></tr>
      <tr><td>SHA-256</td><td>b6a4c67e…aec2 (141,882,994 bytes)</td></tr>
    </tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    eda_figs = [
        ("01_price_dist.png",     "Price distribution (log scale)"),
        ("02_area_vs_price.png",  "Area vs Price"),
        ("03_room_dist.png",      "Rooms distribution"),
        ("04_district_price.png", "Median price by district"),
        ("05_category_dist.png",  "Category breakdown"),
        ("06_floor_dist.png",     "Floor distribution"),
        ("07_correlation.png",    "Feature correlation heatmap"),
    ]
    avail = [(n, c) for n, c in eda_figs if (FIGS / n).exists()]
    if avail:
        st.markdown(
            '<div class="card"><div class="c-title">Exploratory Data Analysis</div>'
            '<div class="c-sub">Generated by <code>src/eda.py</code></div>',
            unsafe_allow_html=True,
        )
        for i in range(0, len(avail), 2):
            g1, g2 = st.columns(2, gap="medium")
            with g1:
                show_fig(avail[i][0], avail[i][1])
            if i + 1 < len(avail):
                with g2:
                    show_fig(avail[i + 1][0], avail[i + 1][1])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card"><div class="c-title">EDA Figures</div>'
            '<div class="box-info">Figures not found in <code>report/figures/</code>.</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "predict":
    hero(
        "INTERACTIVE DEMO",
        "Price a listing in seconds",
        "Enter the same fields the models were trained on and compare Task A and Task B outputs.",
        [("Task A", "log(price) regression"), ("Task B", "premium vs standard"), ("4", "models available")],
    )
    st.markdown('<div class="cont">', unsafe_allow_html=True)

    if _DEMO:
        demo_banner()
        # Show a static example result so the page is still useful
        st.markdown("""
<div class="card">
  <div class="c-title">Example prediction  <span style="font-size:.78rem;font-weight:400;color:#94a3b8">(hardcoded — add dataset for live inference)</span></div>
  <div class="c-sub">Yeni tikili · Nəsimi r. · 85 m² · 3 otaq · 7/16 mərtəbə · Təmirli · Kupçalı</div>
  <div class="price-big">235,000 <span class="price-unit">AZN</span></div>
  <div class="price-sqm">2,765 AZN / m²</div>
  <div class="badge badge-pre">💎 Premium tier</div>
  <div class="prob-track"><div class="prob-fill" style="width:87%"></div></div>
  <div class="prob-lbl">P(premium) = 0.87 · threshold 210,000 AZN (train median)</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        col_form, col_res = st.columns([11, 9], gap="large")

        with col_form:
            with st.form("listing_form"):
                st.markdown(
                    '<div class="c-title">Listing details</div>'
                    '<div class="c-sub">Features used by the models</div>',
                    unsafe_allow_html=True,
                )
                r1, r2       = st.columns(2)
                category     = r1.selectbox("Category",              bundle["categories"])
                district     = r2.selectbox("District",              bundle["district_order"])
                area         = r1.number_input("Area, m²",           10.0, 20_000.0, 85.0, 5.0)
                rooms        = r2.number_input("Rooms",              0, 30, 3)
                floor        = r1.number_input("Floor",              0, 60, 7)
                floors_total = r2.number_input("Floors in building", 0, 60, 16)
                land         = r1.number_input("Land area, sot",     0.0, 1_000.0, 0.0, 0.5)
                b1, b2       = st.columns(2)
                repair_opt   = b1.selectbox("Repair",     ["Yes", "No"])
                bill_opt     = b2.selectbox("Bill of sale", ["Yes", "No"])
                mortgage_opt = b1.selectbox("Mortgage",   ["No", "Yes"])
                agent_opt    = b2.selectbox("Seller",     ["Agent", "Owner"])
                st.markdown("---")
                m1, m2       = st.columns(2)
                reg_name     = m1.selectbox("Price model (Task A)", list(bundle["models"]["regression"]))
                tier_name    = m2.selectbox("Tier model  (Task B)", list(bundle["models"]["tier"]))
                submitted    = st.form_submit_button(
                    "Predict price", type="primary", use_container_width=True
                )

        with col_res:
            st.markdown(
                '<div class="c-title" style="margin-bottom:.2rem">Prediction</div>'
                '<div class="c-sub">Model output for the described listing</div>',
                unsafe_allow_html=True,
            )
            if not submitted:
                st.markdown(
                    '<div class="box-info">Fill in the listing details on the left and click <strong>Predict price</strong>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                if floor > floors_total > 0:
                    st.markdown(
                        '<div class="box-warn">⚠️ Floor exceeds total floors in the building.</div>',
                        unsafe_allow_html=True,
                    )
                try:
                    out = serve.predict(
                        bundle,
                        {
                            "category":         category,
                            "district":         district,
                            "area_m2":          float(area),
                            "rooms":            int(rooms),
                            "floor":            int(floor),
                            "floors_total":     int(floors_total),
                            "land_sot":         float(land),
                            "has_repair":       (repair_opt   == "Yes"),
                            "has_bill_of_sale": (bill_opt     == "Yes"),
                            "has_mortgage":     (mortgage_opt == "Yes"),
                            "is_agent":         (agent_opt    == "Agent"),
                        },
                        reg_name,
                        tier_name,
                    )
                except Exception as exc:
                    st.markdown(
                        '<div class="box-err">Prediction failed: ' + str(exc) + '</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

                target = max(1, int(out["price_azn"]))
                step   = max(1, target // 40)
                ph     = st.empty()
                for v in range(0, target, step):
                    ph.markdown(
                        '<div class="price-big">' + "{:,}".format(v) +
                        ' <span class="price-unit">AZN</span></div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.016)
                ph.markdown(
                    '<div class="price-big">' + "{:,}".format(target) +
                    ' <span class="price-unit">AZN</span></div>'
                    '<div class="price-sqm">' + "{:,.0f}".format(target / max(float(area), 1.0)) +
                    " AZN / m²</div>",
                    unsafe_allow_html=True,
                )

                is_pre = out["is_premium"]
                st.markdown(
                    '<div class="badge badge-pre">💎 Premium tier</div>' if is_pre
                    else '<div class="badge badge-std">🏡 Standard tier</div>',
                    unsafe_allow_html=True,
                )

                thr_fmt = "{:,.0f}".format(out["threshold"])
                if out["score_kind"] == "probability":
                    prob = float(np.clip(out["score"], 0.0, 1.0))
                    pct  = int(prob * 100)
                    st.markdown(
                        '<div class="prob-track"><div class="prob-fill" style="width:' + str(pct) + '%"></div></div>'
                        '<div class="prob-lbl">P(premium) = {:.2f} · threshold {} AZN</div>'.format(prob, thr_fmt),
                        unsafe_allow_html=True,
                    )
                else:
                    mg = out["score"]
                    st.markdown(
                        '<div class="prob-lbl">SVM margin = {:+.2f} · threshold {} AZN</div>'.format(mg, thr_fmt),
                        unsafe_allow_html=True,
                    )

                if not out["known_inputs"]:
                    st.markdown(
                        '<div class="box-warn">⚠️ Unseen district / category — result is extrapolated.</div>',
                        unsafe_allow_html=True,
                    )
                if is_pre:
                    st.balloons()

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "models":
    hero(
        "BENCHMARK",
        "Our models versus scikit-learn",
        "Same split, same features, matched settings. Our own tree and SVM join the table when merged.",
        [("0.921", "F1, our tree (test)"), ("0.930", "ROC-AUC, LinearSVC (test)"), ("0.761", "R² on log(price)")],
    )
    st.markdown('<div class="cont">', unsafe_allow_html=True)
    if _DEMO:
        demo_banner()

    _b_test = read_table("task_b_test.md") or _no_results()
    st.markdown(
        '<div class="card"><div class="c-title">Task B — price tier classification</div>'
        '<div class="c-sub">Test set · premium is the positive class</div>'
        + _b_test + '</div>',
        unsafe_allow_html=True,
    )

    col_a, col_cm = st.columns([3, 2], gap="large")
    with col_a:
        _a_test   = read_table("task_a_test.md")
        _b_cv     = read_table("task_b_cv.md")
        _cv_block = (
            '<div class="c-sub" style="margin-top:1.2rem;padding-top:1.2rem;border-top:1px solid #f1f5f9">'
            '5-fold cross-validation (Task B)</div>' + _b_cv
            if _b_cv else ""
        )
        st.markdown(
            '<div class="card"><div class="c-title">Task A — log(price) regression</div>'
            '<div class="c-sub">Test set</div>'
            + (_a_test or _no_results()) + _cv_block + '</div>',
            unsafe_allow_html=True,
        )

    with col_cm:
        _cm = FIGS / "08_confusion_matrices.png"
        if _cm.exists():
            st.markdown(
                '<div class="card"><div class="c-title">Confusion matrix</div>'
                '<div class="c-sub">sklearn tree · test set</div>',
                unsafe_allow_html=True,
            )
            st.image(str(_cm), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="card">
  <div class="c-title">Confusion matrix</div>
  <div class="c-sub">sklearn tree · test set (from report)</div>
  <div style="display:flex;justify-content:space-evenly;font-size:.72rem;color:#94a3b8;margin-bottom:.4rem">
    <span>pred. standard</span><span>pred. premium</span>
  </div>
  <div class="cm-grid">
    <div class="cm-cell cm-tn">4,438<span class="cm-sub">true standard ✓</span></div>
    <div class="cm-cell cm-fp">416<span class="cm-sub">false premium</span></div>
    <div class="cm-cell cm-fn">388<span class="cm-sub">false standard</span></div>
    <div class="cm-cell cm-tp">4,346<span class="cm-sub">true premium ✓</span></div>
  </div>
</div>""", unsafe_allow_html=True)

    _roc = FIGS / "09_roc_curves.png"
    if _roc.exists():
        st.markdown(
            '<div class="card"><div class="c-title">ROC curves</div>'
            '<div class="c-sub">Test set · all classifiers</div>',
            unsafe_allow_html=True,
        )
        st.image(str(_roc), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "error":
    hero(
        "DIAGNOSIS",
        "Where and why the models fail",
        "Errors by price range, category and district · computed on the test set.",
        [("14.4 %", "tree tier error, 200–350k"), ("34.8 %", "median price error, >1M"), ("294", "luxury listings in test")],
    )
    st.markdown('<div class="cont">', unsafe_allow_html=True)
    if _DEMO:
        demo_banner()

    col_chart, col_text = st.columns([3, 2], gap="large")

    with col_chart:
        _err_pngs = sorted(FIGS.glob("err_*.png")) if FIGS.exists() else []
        if _err_pngs:
            st.markdown(
                '<div class="card"><div class="c-title">Error diagnostics</div>'
                '<div class="c-sub">Generated by <code>src/error_analysis.py</code></div>',
                unsafe_allow_html=True,
            )
            for _p in _err_pngs:
                st.image(str(_p), caption=_p.stem.replace("_", " "), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                matplotlib.rcParams.update({"font.family": "sans-serif"})

                bands    = ["<100k", "100–200k", "200–350k", "350–600k", "600k–1M", ">1M"]
                tree_err = [3.9,  7.9,  14.4, 3.5, 2.8, 2.0]
                lsvc_err = [6.0, 18.6,  23.9, 4.6, 3.6, 4.8]
                x, w     = np.arange(len(bands)), 0.37

                fig, ax = plt.subplots(figsize=(8, 4.2))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                for sp in ["top", "right", "left"]:
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color("#e2e8f0")
                ax.yaxis.set_visible(False)
                ax.set_xticks(x)
                ax.set_xticklabels(bands, fontsize=9.5, color="#64748b")
                ax.tick_params(colors="#64748b")
                b1 = ax.bar(x - w / 2, tree_err, w, color="#0d9488", zorder=3)
                b2 = ax.bar(x + w / 2, lsvc_err, w, color="#d97706", zorder=3)
                ax.set_ylim(0, max(lsvc_err) * 1.32)
                for bar in list(b1) + list(b2):
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + .25,
                            str(h), ha="center", va="bottom",
                            fontsize=8.5, fontweight="600", color="#334155")
                ax.legend(
                    handles=[mpatches.Patch(color="#0d9488", label="sklearn tree"),
                             mpatches.Patch(color="#d97706", label="sklearn LinearSVC")],
                    loc="upper right", fontsize=9, frameon=False,
                )
                plt.tight_layout(pad=1.2)
                st.markdown(
                    '<div class="card"><div class="c-title">Tier error rate by price range, %</div>'
                    '<div class="c-sub">Test set</div>',
                    unsafe_allow_html=True,
                )
                st.pyplot(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                plt.close(fig)
            except Exception as ex:
                st.markdown(
                    '<div class="box-info">Chart unavailable: ' + str(ex) + '</div>',
                    unsafe_allow_html=True,
                )

    with col_text:
        st.markdown("""
<div class="card">
  <div class="c-title">What the errors say</div>
  <p style="font-size:.9rem;color:#334155;line-height:1.65;margin-bottom:.9rem">
    Tier mistakes pile up in the <strong>200–350k band</strong>,
    right around the 210,000 AZN decision threshold.
  </p>
  <p style="font-size:.9rem;color:#334155;line-height:1.65;margin-bottom:.9rem">
    Luxury listings (&gt;1M AZN) are easy to classify as premium,
    but the tree <strong>underprices</strong> them badly:
    median regression error 34.8 %, bias &minus;540,000 AZN.
  </p>
  <p style="font-size:.82rem;color:#94a3b8">
    Source: <code>report/tables/</code> — written by <code>python -m src.run_all</code>.
  </p>
</div>
""", unsafe_allow_html=True)

    if TABLES.exists():
        for _tbl in sorted(TABLES.glob("*_price_range.md")) + sorted(TABLES.glob("*_category.md")):
            try:
                _cnt   = _tbl.read_text(encoding="utf-8")
                _title = _tbl.stem.replace("_", " ").title()
                st.markdown(
                    '<div class="card"><div class="c-title">' + _title + '</div>'
                    + md_to_table(_cnt) + '</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

    _pa = FIGS / "10_pred_vs_actual.png"
    if _pa.exists():
        st.markdown(
            '<div class="card"><div class="c-title">Predicted vs actual log(price)</div>'
            '<div class="c-sub">Task A regression · test set</div>',
            unsafe_allow_html=True,
        )
        st.image(str(_pa), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
