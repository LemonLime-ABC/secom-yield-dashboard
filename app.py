# ============================================
# app.py — 반도체 수율 분석 대시보드 (통합 교정판)
# ============================================
# 실행:  python -m streamlit run app.py
#
# [이 판의 핵심 원칙]
# 1. 웨이퍼 단위 확률은 항상 OOF(교차검증 예측)를 쓴다.
#    최종 모델이 자기 학습 데이터를 예측한 값(PR-AUC 0.999로 관측)은
#    과대평가이므로 화면에 쓰지 않는다. (비교 설명용으로만 표시)
# 2. 성능 숫자는 하드코딩하지 않고 전부 artifacts에서 읽는다.
#    → main2.py를 다시 돌리면 화면 숫자가 자동으로 갱신된다.
# 3. 합성 주입 시나리오와 별도로, 데이터에 실제로 존재하는
#    수율 이탈(2008-08-03 주간)을 분석하는 화면을 둔다.

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import xgboost as xgb
import plotly.graph_objects as go

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, 'artifacts')

st.set_page_config(page_title="반도체 수율 분석 대시보드", layout="wide")

# --------------------------------------------
# 데이터·모델 로드 (전부 캐시)
# --------------------------------------------
@st.cache_resource
def load_all():
    bundle = joblib.load(os.path.join(ART, 'secom_model.joblib'))
    pre = joblib.load(os.path.join(ART, 'precomputed.joblib'))
    exc = joblib.load(os.path.join(ART, 'excursion.joblib'))
    scen = joblib.load(os.path.join(ART, 'scenarios.joblib'))
    spc = joblib.load(os.path.join(ART, 'spc_baseline.joblib'))
    return bundle, pre, exc, scen, spc

@st.cache_data
def load_raw():
    data = pd.read_csv(os.path.join(BASE, 'secom.data'), sep=r'\s+', header=None)
    return data

bundle, pre, exc, scen, spc = load_all()
raw = load_raw()

model = bundle['model']
THRESHOLD = bundle['threshold']
sensor_cols = bundle['sensor_columns']
miss_cols = bundle['miss_cols']
all_cols = bundle['columns']
normal_median = bundle['normal_median']
normal_sigma = bundle['normal_sigma']
oof_prob = bundle['oof_prob']
y = bundle['y_true']
timestamps = pd.to_datetime(bundle['timestamps'])
fold_scores = bundle['fold_scores']
thr_table = pd.DataFrame(bundle['threshold_table'])

shap_all = pre['shap_all']
prob_insample = pre['prob_insample']

X = raw[sensor_cols].copy()

def rebuild(Xs):
    ind = Xs[miss_cols].isna().astype(int).add_prefix('m_')
    return pd.concat([Xs, ind], axis=1).reindex(columns=all_cols)

# 그래프 공통 색상
C = {'model': '#d93025', 'actual': '#1a73e8', 'gray': '#9aa0a6', 'line': '#5f6368'}

@st.cache_data(show_spinner="반응 곡선 계산 중")
def response_curve(sensor, n_points=40, sample_n=150):
    """센서 값을 전 구간에 훑으며 모델 예측 평균을 계산하고,
    실제 데이터의 구간별 불량률을 함께 반환한다.

    트리 모델은 분기점에서만 값이 바뀌므로 곡선이 계단 모양이 된다.
    이 계단의 폭이 곧 모델의 분해능이며, 마지막 계단 밖으로는
    아무리 값을 밀어도 예측이 변하지 않는다(포화).
    """
    s = X[sensor].dropna()
    lo, hi = s.quantile(0.02), s.quantile(0.98)
    if lo >= hi:
        return None
    grid = np.linspace(lo, hi, n_points)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(sample_n, len(X)), replace=False)
    base = X.iloc[idx].copy()
    col = base.columns.get_loc(sensor)
    preds = []
    for v in grid:
        t = base.copy()
        t.iloc[:, col] = v
        preds.append(float(model.predict_proba(rebuild(t).values)[:, 1].mean()))
    # 실제 데이터의 구간별(10분위) 불량률
    try:
        bins = pd.qcut(s, 10, duplicates='drop')
        actual = pd.DataFrame({'v': s, 'y': y[s.index]}).groupby(
            bins, observed=True).agg(mid=('v', 'median'), rate=('y', 'mean'),
                                     n=('y', 'size')).reset_index(drop=True)
    except Exception:
        actual = None
    return grid, np.array(preds), actual

@st.cache_data
def weekly_model_trend():
    """주간 실제 불량률과 모델 평균 확률(OOF)을 함께 집계."""
    d = pd.DataFrame({'ts': timestamps, 'y': y, 'oof': oof_prob}).sort_values('ts')
    g = d.set_index('ts').resample('W').agg(
        n=('y', 'size'), fail=('y', 'sum'), model=('oof', 'mean'))
    g = g[g.n >= 10].copy()
    g['rate'] = g.fail / g.n
    return g

# 파생 지표 (전부 저장값에서 계산 — 하드코딩 없음)
pr_mean = float(np.mean(fold_scores))
pr_std = float(np.std(fold_scores, ddof=1))
row20 = thr_table[thr_table.threshold == THRESHOLD].iloc[0]
REC, PREC = float(row20.recall), float(row20.precision)
TP, FP = int(row20.tp), int(row20.fp)
N_BAD = int(y.sum())

# --------------------------------------------
# 화면 선택 (사이드바)
# --------------------------------------------
PAGES = ["개별 웨이퍼 진단", "What-if 시뮬레이터", "실제 수율 이탈 분석",
         "합성 시나리오 검증", "모델 성능·방법론"]
page = st.sidebar.radio("화면 선택", PAGES)
st.sidebar.divider()
st.sidebar.caption(
    f"SECOM 1,567장 / 센서 {len(sensor_cols)}개\n\n"
    f"운영 임계값 {THRESHOLD} / OOF 기반 표시"
)

st.title("반도체 수율 분석 대시보드")

tab1 = st.container() if page == PAGES[0] else st.empty()
tab2 = st.container() if page == PAGES[1] else st.empty()
tab3 = st.container() if page == PAGES[2] else st.empty()
tab4 = st.container() if page == PAGES[3] else st.empty()
tab5 = st.container() if page == PAGES[4] else st.empty()

# ============================================
# 화면 1: 개별 웨이퍼 진단 (OOF 기반)
# ============================================
with tab1:
    st.subheader("웨이퍼 단위 불량 예측 및 원인 분해")
    st.caption(
        "표시되는 확률은 교차검증(out-of-fold) 예측입니다. "
        "각 웨이퍼는 자신을 학습에 쓰지 않은 모델에게 평가받은 값이므로, "
        "실제 운영에서 처음 보는 웨이퍼에 대한 성능과 같은 조건입니다."
    )

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        view = st.radio("웨이퍼 목록", ["전체", "실제 불량만", "위험 판정만(OOF)"])
        if view == "실제 불량만":
            candidates = np.where(y == 1)[0]
        elif view == "위험 판정만(OOF)":
            candidates = np.where(oof_prob >= THRESHOLD)[0]
        else:
            candidates = np.arange(len(y))
        wafer_id = st.selectbox("웨이퍼 번호", candidates, index=0)

    p_oof = oof_prob[wafer_id]
    p_in = prob_insample[wafer_id]
    actual = "불량" if y[wafer_id] == 1 else "정상"
    verdict = "위험 (정밀검사 대상)" if p_oof >= THRESHOLD else "통과"

    with col_info:
        c1, c2, c3 = st.columns(3)
        c1.metric("불량 확률 (OOF)", f"{p_oof:.3f}")
        c2.metric(f"판정 (임계값 {THRESHOLD})", verdict)
        c3.metric("실제 라벨", actual)
        st.caption(
            f"참고 — 최종 모델이 이 웨이퍼(자기 학습 데이터)를 예측하면 {p_in:.3f}이 나옵니다. "
            f"학습 데이터 예측은 과대평가되므로 판정에 쓰지 않습니다."
        )

    st.divider()
    st.markdown("#### 판정 근거 (SHAP 기여도 상위 10)")
    contrib = shap_all[wafer_id]
    top_idx = np.argsort(np.abs(contrib))[::-1][:10]
    rows = []
    for i in top_idx:
        cname = all_cols[i]
        is_ind = isinstance(cname, str) and str(cname).startswith('m_')
        if is_ind:
            label, val_txt, sig_txt = f"{cname} (결측 여부)", "-", "-"
        else:
            label = f"센서 {cname}"
            v = X.loc[wafer_id, cname]
            med, sig = normal_median[cname], normal_sigma[cname]
            if pd.isna(v):
                val_txt, sig_txt = "결측", "-"
            elif sig and sig > 0:
                val_txt, sig_txt = f"{v:.3f}", f"{(v - med) / sig:+.2f}σ"
            else:
                val_txt, sig_txt = f"{v:.3f}", "-"
        rows.append({"센서": label, "SHAP": contrib[i],
                     "방향": "불량 쪽" if contrib[i] > 0 else "정상 쪽",
                     "측정값": val_txt, "정상 대비": sig_txt})
    df_top = pd.DataFrame(rows)
    a, b = st.columns(2)
    with a:
        st.bar_chart(df_top.set_index("센서")[["SHAP"]], horizontal=True)
    with b:
        st.dataframe(df_top, hide_index=True, use_container_width=True)

    # --------------------------------------------
    # 반응 곡선 — 이 센서 값이 변하면 위험도가 어떻게 움직이나
    # --------------------------------------------
    st.divider()
    st.markdown("#### 반응 곡선 — 이 센서의 값이 변하면 위험도가 어떻게 움직이나")
    st.caption(
        "센서 값을 전 구간에 걸쳐 훑으며 모델 예측을 그립니다. "
        "실제 데이터의 구간별 불량률을 함께 겹쳐, "
        "**모델이 학습한 관계가 데이터와 맞는지** 확인할 수 있습니다."
    )

    # 이 웨이퍼의 SHAP 상위 센서 중에서 고르게 한다
    curve_cands = [c for c in df_top["센서"]
                   if str(c).startswith("센서 ")]
    curve_cands = [int(str(c).replace("센서 ", "")) for c in curve_cands]
    if curve_cands:
        pick_sensor = st.selectbox("반응 곡선을 볼 센서", curve_cands, index=0,
                                   key="curve_sensor")
        rc = response_curve(pick_sensor)
        if rc is None:
            st.info("이 센서는 값의 산포가 없어 반응 곡선을 그릴 수 없습니다.")
        else:
            grid, preds, actual = rc
            this_val = X.loc[wafer_id, pick_sensor]

            fc = go.Figure()
            fc.add_trace(go.Scatter(
                x=grid, y=preds, mode="lines", name="모델 예측 (평균)",
                line=dict(color=C['actual'], width=2.5, shape='hv')))
            if actual is not None:
                fc.add_trace(go.Scatter(
                    x=actual['mid'], y=actual['rate'], mode="lines+markers",
                    name="실제 불량률 (구간별)", yaxis="y2",
                    line=dict(color=C['model'], width=1.5, dash="dot"),
                    marker=dict(size=8, symbol="diamond")))
            if not pd.isna(this_val):
                fc.add_vline(x=float(this_val), line_dash="dash",
                             line_color=C['line'],
                             annotation_text=f"이 웨이퍼 {this_val:.2f}")
            fc.update_layout(
                height=340, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title=f"센서 {pick_sensor} 값",
                yaxis=dict(title="모델 평균 확률"),
                yaxis2=dict(title="실제 불량률", overlaying="y", side="right",
                            tickformat=".0%"),
                legend=dict(orientation="h", y=1.18))
            st.plotly_chart(fc, use_container_width=True)

            n_steps = len(np.unique(np.round(preds, 6)))
            span = preds.max() - preds.min()
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("예측 변동폭", f"{span:.4f}",
                       help="이 센서만 전 구간 움직였을 때 확률이 변하는 폭")
            rc2.metric("계단 수", f"{n_steps}개",
                       help="트리 분기점 개수. 모델의 분해능에 해당")
            rc3.metric("최고 위험 구간", f"{grid[int(np.argmax(preds))]:.2f}")

            st.info(
                "**곡선이 계단 모양인 이유** — 트리 모델은 학습에서 찾은 분기점에서만 "
                "판단이 바뀝니다. 계단의 폭이 곧 이 모델의 분해능이며, "
                "마지막 계단 바깥으로는 값을 아무리 밀어도 예측이 변하지 않습니다"
                "(검출 포화). 계측기의 검출 한계와 같은 개념입니다.\n\n"
                "**두 선을 비교하는 법** — 모델 예측(실선)과 실제 불량률(점선)이 "
                "같은 방향으로 움직이면, 모델이 학습한 관계가 실제 데이터와 일치한다는 "
                "뜻입니다. 어긋나는 구간이 있다면 그 영역은 표본이 적거나 "
                "다른 센서와의 조합으로 설명되는 부분입니다."
            )

# ============================================
# 화면 2: What-if 시뮬레이터
# ============================================
with tab2:
    st.subheader("센서 조작에 따른 예측 변화 시뮬레이션")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        base_wafer = st.selectbox("기준 웨이퍼", np.arange(len(y)), index=0, key="sim_w")
        n_sliders = st.slider("조작할 센서 개수", 1, 5, 3)

    global_imp = np.abs(shap_all).mean(axis=0)
    ranked = np.argsort(global_imp)[::-1]
    cand = []
    for i in ranked:
        c = all_cols[i]
        if not (isinstance(c, str) and str(c).startswith('m_')):
            if normal_sigma[c] > 0:
                cand.append(c)
        if len(cand) >= 20:
            break

    with col_b:
        st.markdown("**조작할 센서와 이탈 정도 (σ)**")
        adjust = {}
        for k in range(n_sliders):
            c1, c2 = st.columns([1, 2])
            with c1:
                s = st.selectbox(f"센서 {k+1}", cand, index=k, key=f"s_{k}")
            with c2:
                sg = st.slider(f"센서 {s} 이탈량", -3.0, 5.0, 0.0, 0.1, key=f"g_{k}")
            adjust[s] = sg

    st.divider()
    modified = X.iloc[[base_wafer]].copy()
    for s, sg in adjust.items():
        med, sig = normal_median[s], normal_sigma[s]
        modified.iloc[0, modified.columns.get_loc(s)] = med + sg * sig

    p_before = oof_prob[base_wafer]
    Xm = rebuild(modified)
    p_after = model.predict_proba(Xm.values)[0, 1]
    shap_before = shap_all[base_wafer]
    shap_after = model.get_booster().predict(
        xgb.DMatrix(Xm.values), pred_contribs=True)[0, :-1]

    m1, m2, m3 = st.columns(3)
    m1.metric("조작 전 (OOF)", f"{p_before:.3f}")
    m2.metric("조작 후 (모델 예측)", f"{p_after:.3f}", delta=f"{p_after - p_before:+.3f}")
    m3.metric("조작 후 판정", "위험" if p_after >= THRESHOLD else "통과")
    st.caption(
        "조작 전은 OOF 확률, 조작 후는 조작된 입력에 대한 모델 예측입니다. "
        "조작된 값은 모델이 학습한 적 없는 조합이므로 암기 효과가 크게 줄지만, "
        "미조작 부분에는 잔여 과대평가가 있을 수 있어 변화량 위주로 해석합니다."
    )

    st.divider()
    st.markdown("#### 조작으로 인한 판단 근거 변화 (SHAP 변화량 상위 10)")
    delta = shap_after - shap_before
    tops = np.argsort(np.abs(delta))[::-1][:10]
    rows = []
    for i in tops:
        c = all_cols[i]
        is_ind = isinstance(c, str) and str(c).startswith('m_')
        label = f"{c} (결측)" if is_ind else f"센서 {c}"
        rows.append({"센서": label, "변화량": delta[i],
                     "직접 조작": "O" if (not is_ind and c in adjust and adjust[c] != 0) else ""})
    dfd = pd.DataFrame(rows)
    a, b = st.columns(2)
    with a:
        st.bar_chart(dfd.set_index("센서")[["변화량"]], horizontal=True)
    with b:
        st.dataframe(dfd, hide_index=True, use_container_width=True)
    st.info(
        "이 시뮬레이션은 모델이 예측하는 확률의 변화를 보여줍니다. "
        "모델은 상관관계를 학습한 것이므로, 센서 조정이 실제 수율을 바꾼다는 "
        "인과관계를 의미하지 않습니다."
    )

# ============================================
# 화면 3: 실제 수율 이탈 분석 (신설)
# ============================================
with tab3:
    st.subheader("실제 수율 이탈(Excursion) 분석 — 2008-08-03 주간")
    st.caption(
        "합성 주입이 아니라 데이터에 실제로 존재하는 이탈입니다. "
        "타임스탬프(측정 시각)를 이용해 주간 불량률 관리도를 그리고, "
        "관리한계를 초과한 주간을 원인 분석했습니다."
    )

    weekly = exc['weekly']
    p_bar = exc['p_bar']
    exc_week = exc['excursion_week']
    exc_mask = exc['excursion_mask']
    calm_mask = exc['calm_mask']

    exc_n = int(exc_mask.sum())
    exc_fail = int(y[exc_mask].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("이탈 주간", f"{pd.Timestamp(exc_week).date()}")
    k2.metric("해당 주 불량률", f"{exc_fail}/{exc_n} = {exc_fail/exc_n:.1%}",
              delta=f"전체 평균 {p_bar:.1%}", delta_color="off")
    wk_row = weekly.loc[exc_week]
    k3.metric("관리 상한(UCL)", f"{wk_row.ucl:.1%}")
    k4.metric("이항검정 p값", f"{wk_row.pval:.4f}")

    st.markdown("#### 주간 불량률 p-관리도")
    chart = pd.DataFrame({
        "불량률": weekly.rate.values,
        "관리 상한(UCL)": weekly.ucl.values,
        "중심선(평균)": p_bar,
    }, index=[str(d.date()) for d in weekly.index])
    st.line_chart(chart)
    st.caption(
        "각 주의 UCL이 다른 이유: 표본 수가 적은 주는 우연 변동이 크므로 "
        "관리 상한이 자동으로 넓어집니다 (p ± 3√(p(1-p)/n))."
    )

    st.divider()
    st.markdown("#### 모델이 이 이탈을 감지했는가")
    exc_flag = (oof_prob[exc_mask] >= THRESHOLD).mean()
    calm_flag = (oof_prob[calm_mask] >= THRESHOLD).mean()
    exc_30 = (oof_prob[exc_mask] >= THRESHOLD).sum() / exc_n * spc['batch_size']
    z_exc = (exc_30 - spc['mean']) / spc['std']

    g1, g2, g3 = st.columns(3)
    g1.metric("이탈 주간 위험판정 비율(OOF)", f"{exc_flag:.1%}")
    g2.metric("안정 구간 위험판정 비율(OOF)", f"{calm_flag:.1%}",
              delta=f"{exc_flag/calm_flag:.1f}배 차이", delta_color="off")
    g3.metric(f"{spc['batch_size']}장 환산 Z값", f"{z_exc:+.2f}",
              delta="관리한계 초과" if exc_30 > spc['ucl'] else "한계 내",
              delta_color="inverse" if exc_30 > spc['ucl'] else "normal")

    if exc_30 > spc['ucl']:
        st.error(
            f"모델(OOF 확률)은 이탈 주간에 웨이퍼의 {exc_flag:.0%}를 위험 판정했습니다. "
            f"안정 구간({calm_flag:.0%})의 {exc_flag/calm_flag:.1f}배이며, "
            f"{spc['batch_size']}장 환산 {exc_30:.1f}장으로 관리 상한 {spc['ucl']:.1f}장을 초과합니다. "
            f"합성 주입 없이, 실제 발생한 이탈을 배치 신호로 감지했음을 의미합니다."
        )

    st.divider()
    st.markdown("#### 모델이 실제 수율 추이를 따라가는가")
    st.caption(
        "주간 실제 불량률과 모델 평균 확률(OOF)을 겹쳐 그립니다. "
        "두 선이 같은 방향으로 움직이면, 모델이 개별 판정은 부정확해도 "
        "집계 수준의 수율 변화는 따라간다는 뜻입니다."
    )
    gw = weekly_model_trend()
    corr = float(np.corrcoef(gw.rate, gw.model)[0, 1])
    labels = [str(d.date()) for d in gw.index]

    ftrend = go.Figure()
    ftrend.add_trace(go.Scatter(
        x=labels, y=gw.rate, mode="lines+markers", name="실제 불량률",
        line=dict(color=C['actual'], width=2)))
    ftrend.add_trace(go.Scatter(
        x=labels, y=gw.model, mode="lines+markers", name="모델 평균 확률(OOF)",
        yaxis="y2", line=dict(color=C['model'], width=2, dash="dot")))
    ftrend.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="주",
        yaxis=dict(title="실제 불량률", tickformat=".0%"),
        yaxis2=dict(title="모델 평균 확률", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(ftrend, use_container_width=True)

    tr1, tr2 = st.columns(2)
    tr1.metric("실제 불량률 – 모델 확률 상관계수", f"r = {corr:+.3f}")
    ratio = float((gw.model / gw.rate).replace([np.inf, -np.inf], np.nan).mean())
    tr2.metric("모델 확률 / 실제 불량률 평균 배율", f"{ratio:.2f}배")

    st.warning(
        f"**두 축의 눈금이 다릅니다.** 모델 확률은 실제 불량률의 약 {ratio:.1f}배로 "
        f"편향되어 있습니다. 불균형 보정(scale_pos_weight≈14)의 결과이며, "
        f"순위와 추이는 정확하나 절대값은 보정되지 않았습니다. "
        f"상관계수 {corr:+.3f}는 **집계 단위의 추세 추종**을 뜻하며, "
        f"웨이퍼 단위 정확도(Recall {REC:.0%})와는 다른 이야기입니다."
    )

    st.divider()
    st.markdown("#### 원인 후보 센서 (이탈 주간 vs 안정 구간 대조)")
    st.caption(
        "효과크기 d = (이탈 주간 평균 − 안정 구간 평균) / 안정 구간 표준편차. "
        "모델과 무관하게 데이터만으로 계산한 값입니다."
    )
    top_s = exc['top_sensors']
    df_exc = pd.DataFrame({
        "센서": [f"센서 {c}" for c in top_s.index],
        "효과크기 d": top_s.values,
    })
    a, b = st.columns(2)
    with a:
        st.bar_chart(df_exc.set_index("센서"), horizontal=True)
    with b:
        st.dataframe(df_exc, hide_index=True, use_container_width=True)
    st.info(
        "d가 3을 넘는 센서들(300, 299, 165, 164 등)은 이탈 주간에 안정 구간 대비 "
        "3 표준편차 이상 이동해 있었습니다. 센서가 익명화되어 있어 어느 공정 모듈인지 "
        "특정할 수는 없으나, 인접 번호 센서들이 함께 이동한 패턴은 "
        "특정 설비군의 계측값이 동반 이동했을 가능성과 부합합니다."
    )

# ============================================
# 화면 4: 합성 시나리오 검증
# ============================================
with tab4:
    st.subheader("합성 시나리오 검증 — 주입·대조군·관리도 판정")
    st.caption(
        "정상 웨이퍼 100장 중 앞 30장에만 이상을 주입한 통제 실험입니다. "
        "미조작 70장의 확률은 OOF를 사용하므로 오탐 기준이 정직합니다."
    )

    scenarios = scen['scenarios']
    n_inj = scen['n_inject']
    base_oof = scen['base_oof']

    names = {k: v['name'] for k, v in scenarios.items()}
    picked = st.selectbox("시나리오 선택", list(names.keys()),
                          format_func=lambda k: names[k])
    sc = scenarios[picked]
    prob = sc['prob']
    injected = sc['injected']

    over = int((prob >= THRESHOLD).sum())
    base_over = int((base_oof >= THRESHOLD).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("위험 판정", f"{over}/100장", delta=f"{over - base_over:+d} (조작 전 대비)")
    k2.metric("조작 구간 검출", f"{int((prob[:n_inj] >= THRESHOLD).sum())}/{n_inj}장")
    k3.metric("미조작 구간 오탐(OOF)", f"{int((prob[n_inj:] >= THRESHOLD).sum())}/70장")

    st.markdown("#### 웨이퍼별 확률과 관리 기준")
    chart_df = pd.DataFrame({
        "불량 확률": prob,
        "운영 임계값": THRESHOLD,
    })
    st.line_chart(chart_df)

    # SPC 판정
    if picked != 'S5':
        ev = spc['scenario_eval'].get(picked)
        det = ev['detected']
        if ev['over_ucl']:
            st.error(
                f"**관리한계 초과 — 이상 신호.** 조작 구간 위험 판정 {det}장은 "
                f"OOF 기반 기준선({spc['mean']:.2f}±{spc['std']:.2f}장)의 관리 상한 "
                f"{spc['ucl']:.1f}장을 초과합니다 (Z={ev['z']:+.2f}, p={ev['p']:.4f})."
            )
        elif ev['z'] > 2:
            st.warning(
                f"**경계 상태.** {det}장은 관리 상한({spc['ucl']:.1f}장) 이내이나 "
                f"기준선보다 {ev['z']:.1f}σ 높습니다 (p={ev['p']:.4f}). "
                f"3σ 한계는 오경보 억제를 위해 보수적이므로 연속 관측이 필요한 영역입니다."
            )
        else:
            st.info(
                f"**관리한계 이내.** {det}장은 정상 변동 범위"
                f"({spc['p95_low']:.0f}~{spc['p95_high']:.0f}장) 안입니다 (p={ev['p']:.4f}). "
                f"이 수준의 이탈은 배치 신호로 구별되지 않습니다."
            )
    else:
        seg = [(s, int((prob[s:s+20] >= THRESHOLD).sum()), prob[s:s+20].mean())
               for s in range(0, 100, 20)]
        st.dataframe(pd.DataFrame(
            {"구간": [f"{s}–{s+19}" for s, _, _ in seg],
             "위험 판정/20": [d for _, d, _ in seg],
             "평균 확률": [f"{m:.3f}" for _, _, m in seg]}),
            hide_index=True, use_container_width=True)
        st.info("구간이 진행될수록 판정 수와 평균 확률이 증가합니다. "
                "개별로는 정상 판정되는 초기에도 배치 추세로는 악화가 드러납니다.")

    # 원인 진단 (조작 센서 역추적)
    if picked != 'S5':
        st.divider()
        st.markdown("#### 원인 진단 (정상 대비 이탈 상위)")
        batch = sc['batch']
        target = batch.iloc[:n_inj]
        devs = []
        for c in sensor_cols:
            med, sig = normal_median[c], normal_sigma[c]
            if not sig or sig <= 0 or pd.isna(sig):
                continue
            v = target[c].median()
            if pd.isna(v):
                continue
            devs.append({"센서": c, "이탈량(σ)": (v - med) / sig})
        dfv = pd.DataFrame(devs)
        dfv["abs"] = dfv["이탈량(σ)"].abs()
        dfv = dfv.sort_values("abs", ascending=False).head(10)
        dfv["실제 조작"] = dfv["센서"].apply(lambda c: "O" if c in injected else "")
        dfv["센서"] = dfv["센서"].apply(lambda c: f"센서 {int(c)}")
        a, b = st.columns(2)
        with a:
            st.bar_chart(dfv.set_index("센서")[["이탈량(σ)"]], horizontal=True)
        with b:
            st.dataframe(dfv[["센서", "이탈량(σ)", "실제 조작"]],
                         hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### 대조 실험 — 의미 있는 센서 vs 무작위 센서")
    st.caption(
        "불량 관련 상위 센서 K개를 흔든 경우와, 무작위 센서 K개를 같은 크기로 흔든 경우를 "
        "비교합니다. 센서 선정은 모델과 무관한 단변량 효과크기 d 기준이며 방향을 보정했습니다. "
        "무작위 대조군이 함께 오르지 않아야 '많이 흔들면 그냥 오른다'는 반론이 배제됩니다."
    )
    sweep = pd.DataFrame(scen['sweep'])
    sweep_view = pd.DataFrame({
        "조작 센서 수 K": sweep.K,
        "상위 d 센서: 검출/30": sweep.top_det,
        "무작위 센서: 검출/30": sweep.rand_det.round(1),
        "상위 d: 평균확률": sweep.top_prob.round(3),
        "무작위: 평균확률": sweep.rand_prob.round(3),
    })
    a, b = st.columns(2)
    with a:
        st.dataframe(sweep_view, hide_index=True, use_container_width=True)
    with b:
        st.line_chart(sweep_view.set_index("조작 센서 수 K")[
            ["상위 d 센서: 검출/30", "무작위 센서: 검출/30"]])
    st.success(
        "상위 d 센서 주입은 검출을 크게 올리지만, 같은 개수의 무작위 센서 주입은 "
        "기준선 부근에 머뭅니다. 모델이 '흔들림의 양'이 아니라 "
        "'불량과 연관된 방향의 이동'에 반응함을 보여줍니다."
    )


# ============================================
# 화면 5: 모델 성능·방법론
# ============================================
with tab5:
    st.subheader("모델 성능 및 검증 방법론")

    st.markdown("#### 시스템의 역할 정의")
    st.info(
        "본 시스템은 개별 웨이퍼 판정기가 아니라, 배치 단위로 공정 이상을 감지하고 "
        "원인 센서를 규명하는 진단 도구입니다. 웨이퍼 1장의 예측만으로 설비를 조정하는 것은 "
        "과잉조정(over-adjustment)에 해당하므로 배치 통계를 근거로 판단합니다."
    )
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            f"""
            **강점 (배치 단위)**
            - 실제 수율 이탈(2008-08-03 주간)을 배치 신호로 감지
            - 무작위 대조군 대비 명확한 분리 (의미 있는 이동에만 반응)
            - 단일 관리도가 놓치는 '관리한계 내 이탈의 누적' 포착
            """
        )
    with r2:
        st.markdown(
            f"""
            **한계 (개별 웨이퍼 단위, OOF 기준)**
            - Recall {REC:.1%} — 불량 {N_BAD}장 중 {N_BAD - TP}장 미검출
            - Precision {PREC:.1%} — 위험 판정 {TP + FP}장 중 실제 불량 {TP}장
            - 개별 판정에 근거한 의사결정은 부적절
            """
        )

    st.divider()
    st.markdown("#### 검증 결과 (전부 OOF 기준)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("5-Fold PR-AUC", f"{pr_mean:.3f} ± {pr_std:.3f}")
    p2.metric("무작위 기준선 대비", f"{pr_mean / y.mean():.1f}배",
              help=f"PR-AUC의 무작위 기준선은 양성 비율 {y.mean():.3f}")
    p3.metric(f"Recall @ {THRESHOLD}", f"{REC:.1%}")
    p4.metric(f"Precision @ {THRESHOLD}", f"{PREC:.1%}")
    st.caption("± 값은 fold 5개의 표본표준편차(ddof=1)입니다.")

    st.divider()
    st.markdown("#### 왜 OOF인가 — 학습 데이터 예측과의 비교")
    from sklearn.metrics import average_precision_score
    ap_in = average_precision_score(y, prob_insample)
    ap_oof = average_precision_score(y, oof_prob)
    cmp = pd.DataFrame({
        "구분": ["최종 모델이 학습 데이터를 예측", "교차검증 OOF 예측 (본 대시보드)"],
        "PR-AUC": [f"{ap_in:.3f}", f"{ap_oof:.3f}"],
        f"Recall @ {THRESHOLD}": [
            f"{((prob_insample >= THRESHOLD) & (y == 1)).sum() / N_BAD:.1%}",
            f"{REC:.1%}"],
        "성격": ["과대평가 (암기한 답 재확인)", "정직 (처음 보는 웨이퍼 조건)"],
    })
    st.dataframe(cmp, hide_index=True, use_container_width=True)
    st.warning(
        "최종 모델은 1,567장 전부를 학습했으므로 같은 데이터를 예측하면 "
        f"PR-AUC {ap_in:.3f}이라는 비현실적 수치가 나옵니다. "
        "본 대시보드의 모든 웨이퍼 단위 확률은 OOF를 사용합니다."
    )

    st.divider()
    st.markdown("#### 임계값을 직접 움직여 보기")
    st.caption(
        "임계값을 바꾸면 검사 우선순위에 오르는 웨이퍼 수와 성능이 어떻게 "
        "맞바뀌는지 즉시 확인할 수 있습니다. 모든 수치는 OOF 기준입니다."
    )
    thr_live = st.slider("판정 임계값", 0.05, 0.95, float(THRESHOLD), 0.05)
    pred_live = (oof_prob >= thr_live)
    tp_l = int((pred_live & (y == 1)).sum())
    fp_l = int((pred_live & (y == 0)).sum())
    fn_l = int((~pred_live & (y == 1)).sum())
    rec_l = tp_l / max(tp_l + fn_l, 1)
    prec_l = tp_l / max(tp_l + fp_l, 1)
    p_base = float(y.mean())

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("우선순위 대상", f"{tp_l + fp_l}장",
              delta=f"전체의 {(tp_l + fp_l) / len(y):.1%}", delta_color="off")
    s2.metric("포착한 불량", f"{tp_l} / {N_BAD}")
    s3.metric("Recall", f"{rec_l:.1%}")
    s4.metric("Precision", f"{prec_l:.1%}")
    s5.metric("무작위 대비 농축률", f"{prec_l / p_base:.2f}배" if prec_l > 0 else "—",
              help=f"무작위로 같은 수를 골랐을 때의 불량 비율 {p_base:.3f} 대비")

    st.info(
        f"임계값 {thr_live:.2f}에서 상위 {(tp_l + fp_l) / len(y):.0%}를 우선순위군으로 "
        f"분류하면, 그 안의 불량 밀도가 무작위 대비 **{prec_l / p_base:.2f}배**가 됩니다. "
        f"다만 이는 검사 물량을 줄인다는 뜻이 아닙니다. 반도체 양산은 전수검사가 "
        f"전제이므로, 이 수치는 **어느 웨이퍼를 먼저 들여다볼지 정하는 우선순위**와 "
        f"**배치 단위 이상 신호의 강도**로 해석해야 합니다."
    )

    dl, dr = st.columns(2)
    with dl:
        st.markdown("**OOF 확률 분포 — 정상과 불량이 얼마나 겹치는가**")
        fh = go.Figure()
        fh.add_trace(go.Histogram(x=oof_prob[y == 0], name="정상", nbinsx=50,
                                  marker_color=C['gray'], opacity=0.75))
        fh.add_trace(go.Histogram(x=oof_prob[y == 1], name="불량", nbinsx=50,
                                  marker_color=C['model'], opacity=0.85))
        fh.add_vline(x=thr_live, line_dash="dash", line_color=C['line'])
        fh.update_layout(barmode='overlay', height=320, yaxis_type="log",
                         margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="불량 확률(OOF)",
                         yaxis_title="웨이퍼 수 (로그 눈금)",
                         legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fh, use_container_width=True)
        st.caption(
            "세로축이 로그 눈금인 이유는 정상이 불량보다 14배 많아 "
            "일반 눈금에서는 불량 막대가 보이지 않기 때문입니다. "
            "두 분포가 넓게 겹쳐 있는 것이 Precision이 낮은 직접적 원인입니다."
        )
    with dr:
        st.markdown("**Precision–Recall 곡선**")
        order = np.argsort(-oof_prob)
        ys = y[order]
        tps = np.cumsum(ys)
        rec_curve = tps / max(ys.sum(), 1)
        prec_curve = tps / np.arange(1, len(ys) + 1)
        fp_ = go.Figure()
        fp_.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines",
                                 line=dict(color=C['actual'], width=2), name="모델"))
        fp_.add_hline(y=p_base, line_dash="dot", line_color=C['line'],
                      annotation_text=f"무작위 {p_base:.3f}")
        fp_.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Recall", yaxis_title="Precision",
                          yaxis_range=[0, 0.6])
        st.plotly_chart(fp_, use_container_width=True)
        st.caption(
            f"곡선이 점선(무작위 {p_base:.3f})보다 위에 있으면 모델이 의미 있는 "
            f"순위를 매기고 있다는 뜻입니다. 이 곡선 아래 면적이 PR-AUC {ap_oof:.3f}입니다."
        )

    st.divider()
    st.markdown("#### 임계값 선정 근거")
    tv = thr_table.rename(columns={
        'threshold': '임계값', 'tp': f'검출/{N_BAD}', 'fp': '오탐',
        'recall': 'Recall', 'precision': 'Precision'})
    a, b = st.columns(2)
    with a:
        st.dataframe(tv, hide_index=True, use_container_width=True)
    with b:
        st.line_chart(tv.set_index("임계값")[["Recall", "Precision"]])
    st.caption(
        "임계값은 성능을 바꾸는 것이 아니라 Recall과 Precision의 배분을 정하는 문제입니다. "
        "배치 단위 이상 감지가 목적이므로 검출 쪽에 무게를 둔 값을 채택했습니다."
    )

    st.divider()
    st.markdown("#### 평가지표·검증 방법 선택")
    a1, a2 = st.columns(2)
    with a1:
        st.error(
            f"**정확도 미사용** — 불량 비율이 {y.mean():.1%}이므로 전부 정상으로 예측해도 "
            f"정확도 {1 - y.mean():.1%}가 나옵니다. 초기 모델이 실제로 정확도 92.7%를 "
            f"기록하며 불량 0장을 검출했습니다."
        )
        st.error(
            "**ROC-AUC 미사용** — FPR 분모가 정상 전체(1,463장)라 오탐이 늘어도 "
            "지표가 둔감합니다. 불균형 데이터에는 PR-AUC가 적합합니다."
        )
    with a2:
        st.markdown(
            f"""
            **5-Fold 교차검증** — 단일 분할은 검증 불량이 약 21장뿐이라 평가가 분할 운에
            좌우됩니다 (fold별 PR-AUC {min(fold_scores):.3f}~{max(fold_scores):.3f}).
            5-Fold는 불량 {N_BAD}장 전체가 한 번씩 검증됩니다.

            **누수 차단** — 결측 지시자 기준을 fold 내 학습 데이터에서만 산출,
            임계값을 OOF로 결정.

            **확률 보정 주의** — scale_pos_weight≈14를 사용하므로 모델 확률은
            실제 불량률보다 부풀려져 있습니다. 확률을 절대값이 아니라
            순위·비교 용도로 사용합니다.
            """
        )

    st.divider()
    st.markdown("#### 검증의 두 축")
    st.markdown(
        """
        | 축 | 내용 | 결과 |
        |---|---|---|
        | 실제 이탈 | 2008-08-03 주간 (불량률 20.8%, p=0.0010) | OOF 위험판정이 안정 구간의 수 배, 관리한계 초과 |
        | 합성 주입 | 방향 보정 주입 + 무작위 대조군 | 의미 센서만 반응, 대조군은 기준선 유지 |
        """
    )

    st.divider()
    st.markdown("#### 분석의 한계")
    st.warning(
        f"""
        1. **개별 판정 신뢰도** — Recall {REC:.0%} / Precision {PREC:.0%} (OOF). 개별 웨이퍼
        판정에 근거한 의사결정은 부적절하며 배치 통계를 사용해야 합니다.

        2. **센서 익명화** — 원인 센서를 특정 공정 단계와 연결할 수 없습니다.
        실제 업무에서는 분석 결과를 물리적 메커니즘과 연결하는 과정이 추가로 필요합니다.

        3. **상관 ≠ 인과** — What-if 결과는 예측 확률의 변화이며 실제 수율 변화가 아닙니다.

        4. **검출 포화** — 트리 모델은 학습 데이터의 분기점 범위를 넘는 이탈에 추가 반응하지
        않습니다 (민감 구간 약 +1~2σ).

        5. **단일 excursion** — 시계열 검증은 관리한계를 초과한 1개 주간에 기반합니다.
        더 많은 이탈 사례로 일반화하려면 추가 데이터가 필요합니다.
        """
    )

# ============================================
# 숨긴 화면 정리
# ============================================
# st.empty()는 요소를 하나만 담고 마지막 요소를 남긴다.
# 그래서 선택하지 않은 화면의 마지막 문단이 현재 화면 아래에 보이던 문제가 있었다.
# 모든 화면을 그린 뒤, 선택하지 않은 화면의 자리를 비운다.
for _name, _tab in zip(PAGES, [tab1, tab2, tab3, tab4, tab5]):
    if _name != page:
        _tab.empty()
