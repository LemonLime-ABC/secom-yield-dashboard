# ============================================
# app.py — 반도체 수율 최적화 대시보드
# ============================================
# 실행 방법 (터미널):  python -m streamlit run app.py
# 종료 방법: 터미널에서 Ctrl+C
#
# 이 파일은 main2.py와 성격이 다르다.
#  - main2.py : 한 번 실행하고 끝 (분석·학습)
#  - app.py   : 계속 켜져 있으면서 사용자가 조작할 때마다 화면을 다시 그림
# 그래서 매번 모델을 학습하지 않고, main2.py가 저장해둔 파일을 불러 쓴다.

import streamlit as st          # 웹 대시보드 도구
import pandas as pd
import numpy as np
import joblib                   # 저장된 모델 불러오기
import xgboost as xgb

# --------------------------------------------
# 페이지 기본 설정 (가장 먼저 한 번만 호출해야 함)
# --------------------------------------------
st.set_page_config(
    page_title="반도체 수율 최적화 대시보드",
    layout="wide",              # 화면을 넓게 사용
)

# --------------------------------------------
# 데이터·모델 불러오기
# --------------------------------------------
# @st.cache_resource : 한 번 불러온 것을 기억해두는 표시.
#   이게 없으면 사용자가 버튼 하나 누를 때마다 파일을 다시 읽어 느려진다.
@st.cache_resource
def load_artifacts():
    bundle = joblib.load('artifacts/secom_model.joblib')
    return bundle

@st.cache_resource
def load_scenarios():
    return joblib.load('artifacts/scenarios.joblib')

scen_bundle = load_scenarios()

@st.cache_resource
def load_doe():
    return joblib.load('artifacts/doe.joblib')

doe = load_doe()

@st.cache_resource
def load_spc():
    return joblib.load('artifacts/spc_baseline.joblib')

spc = load_spc()

@st.cache_data
def load_raw_data():
    data = pd.read_csv('secom.data', sep=r'\s+', header=None)
    labels = pd.read_csv('secom_labels.data', sep=r'\s+', header=None)
    y = (labels[0] == 1).astype(int).values
    nu = data.nunique(dropna=True)
    X = data.drop(columns=nu[nu <= 1].index)
    return X, y

bundle = load_artifacts()
model = bundle['model']
THRESHOLD = bundle['threshold']
sensor_cols = bundle['sensor_columns']
miss_cols = bundle['miss_cols']

X, y = load_raw_data()

# 모델 입력 형태 만들기 (센서 + 결측 지시자) — main2.py와 동일한 방식
X_final = pd.concat(
    [X, X[miss_cols].isna().astype(int).add_prefix('m_')], axis=1
)

# 전체 웨이퍼의 불량 확률과 SHAP을 미리 계산해둔다 (한 번만)
@st.cache_data
def compute_all():
    prob = model.predict_proba(X_final.values)[:, 1]
    shap_v = model.get_booster().predict(
        xgb.DMatrix(X_final.values), pred_contribs=True)[:, :-1]
    return prob, shap_v

prob_all, shap_all = compute_all()

# 정상 웨이퍼 기준 통계 (몇 σ 벗어났는지 계산용)
normal_median = X[y == 0].median()
normal_sigma = (X[y == 0].quantile(0.75) - X[y == 0].quantile(0.25)) / 1.349

# --------------------------------------------
# 화면 제목
# --------------------------------------------
st.title("반도체 수율 최적화 대시보드")
st.caption("SECOM 공정 센서 데이터 기반 불량 예측 및 원인 분석 시스템")

# --------------------------------------------
# 화면 선택 (사이드바)
# --------------------------------------------
# st.tabs()는 화면을 다시 그릴 때마다 첫 탭으로 초기화되어,
# 탭 내부에서 값을 조작하면 표시가 어긋나는 문제가 있다.
# 사이드바 라디오 방식은 선택 상태가 유지되므로 이 문제가 발생하지 않는다.
PAGES = ["개별 웨이퍼 진단", "What-if 시뮬레이터", "배치 시나리오", "요인설계(DOE)", "모델 성능"]
page = st.sidebar.radio("화면 선택", PAGES)

st.sidebar.divider()
st.sidebar.caption(
    "SECOM 공정 센서 데이터 기반\n\n"
    "불량 예측 및 원인 분석 시스템"
)

# 각 화면을 조건부로 표시하기 위한 컨테이너
# (기존 with tab1: 구조를 그대로 쓰기 위해 이름을 맞춰둔다)
tab1 = st.container() if page == PAGES[0] else st.empty()
tab2 = st.container() if page == PAGES[1] else st.empty()
tab3 = st.container() if page == PAGES[2] else st.empty()
tab4 = st.container() if page == PAGES[3] else st.empty()
tab5 = st.container() if page == PAGES[4] else st.empty()
# ============================================
# 탭 1: 개별 웨이퍼 진단
# ============================================
with tab1:
    st.subheader("웨이퍼 단위 불량 예측 및 원인 분해")

    # --- 웨이퍼 선택 ---
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        # 보기 편하도록 필터 제공
        view = st.radio("웨이퍼 목록", ["전체", "실제 불량만", "모델이 위험 판정한 것만"])
        if view == "실제 불량만":
            candidates = np.where(y == 1)[0]
        elif view == "모델이 위험 판정한 것만":
            candidates = np.where(prob_all >= THRESHOLD)[0]
        else:
            candidates = np.arange(len(y))

        wafer_id = st.selectbox("웨이퍼 번호 선택", candidates, index=0)

    # --- 선택된 웨이퍼의 판정 ---
    p = prob_all[wafer_id]
    actual = "불량" if y[wafer_id] == 1 else "정상"
    verdict = "위험 (정밀검사 대상)" if p >= THRESHOLD else "통과"

    with col_info:
        c1, c2, c3 = st.columns(3)
        # st.metric : 숫자를 크게 강조해 보여주는 요소
        c1.metric("불량 확률", f"{p:.3f}")
        c2.metric(f"판정 (임계값 {THRESHOLD})", verdict)
        c3.metric("실제 라벨", actual)

        if p >= THRESHOLD:
            st.warning(f"이 웨이퍼는 불량 확률이 임계값 {THRESHOLD} 이상이므로 정밀검사 대상으로 분류됩니다.")
        else:
            st.success(f"불량 확률이 임계값 {THRESHOLD} 미만입니다.")

    st.divider()

    # --- SHAP 원인 분해 ---
    st.markdown("#### 판정 근거 (SHAP 기여도 상위 10개 센서)")

    contrib = shap_all[wafer_id]
    top_idx = np.argsort(np.abs(contrib))[::-1][:10]

    rows = []
    for i in top_idx:
        col_name = X_final.columns[i]
        # 결측 지시자 열(m_로 시작)과 실제 센서를 구분
        is_indicator = isinstance(col_name, str) and col_name.startswith('m_')
        if is_indicator:
            label = f"{col_name} (결측 여부)"
            sigma_txt = "-"
            value_txt = "-"
        else:
            label = f"센서 {col_name}"
            v = X.loc[wafer_id, col_name]
            med, sig = normal_median[col_name], normal_sigma[col_name]
            if pd.isna(v):
                value_txt, sigma_txt = "결측", "-"
            elif sig and sig > 0:
                value_txt = f"{v:.3f}"
                sigma_txt = f"{(v - med) / sig:+.2f}σ"
            else:
                value_txt, sigma_txt = f"{v:.3f}", "-"
        rows.append({
            "센서": label,
            "SHAP 기여도": contrib[i],
            "방향": "불량 쪽" if contrib[i] > 0 else "정상 쪽",
            "측정값": value_txt,
            "정상 대비": sigma_txt,
        })

    df_top = pd.DataFrame(rows)

    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        # 막대그래프: 센서별 기여도
        chart_df = df_top.set_index("센서")[["SHAP 기여도"]]
        st.bar_chart(chart_df, horizontal=True)
    with col_table:
        st.dataframe(df_top, hide_index=True, use_container_width=True)

    st.caption(
        "SHAP 기여도가 양수이면 불량 판정 방향으로, 음수이면 정상 판정 방향으로 작용한 센서입니다. "
        "'정상 대비'는 정상 웨이퍼 중앙값 기준으로 몇 σ 벗어났는지를 나타냅니다."
    )

    # ============================================
# 탭 2: What-if 시뮬레이터
# ============================================
# 센서 값을 슬라이더로 조작하면 불량 확률과 SHAP이 즉시 다시 계산된다.
# 목적: "이 센서가 이만큼 틀어지면 모델 판정이 어떻게 변하는가"를 직접 확인.
#
# [주의 - 표현]
#   슬라이더로 바뀌는 것은 '모델이 예측하는 불량 확률'이지 실제 수율이 아니다.
#   모델은 상관관계를 학습했을 뿐 인과를 보장하지 않는다.

with tab2:
    st.subheader("센서 조작에 따른 예측 변화 시뮬레이션")

    # --- 기준 웨이퍼 선택 ---
    col_a, col_b = st.columns([1, 2])
    with col_a:
        base_wafer = st.selectbox(
            "기준 웨이퍼", np.arange(len(y)), index=0, key="sim_wafer"
        )
        n_sliders = st.slider("조작할 센서 개수", 1, 5, 3)

    # 조작 후보: 전역 기여도 상위 센서들 (실제 센서만, 결측 지시자 제외)
    global_imp = np.abs(shap_all).mean(axis=0)
    ranked = np.argsort(global_imp)[::-1]
    candidate_sensors = []
    for i in ranked:
        cname = X_final.columns[i]
        if not (isinstance(cname, str) and cname.startswith('m_')):
            if normal_sigma[cname] > 0:      # 산포가 0인 센서는 조작 불가
                candidate_sensors.append(cname)
        if len(candidate_sensors) >= 20:
            break

    # --- 슬라이더로 센서 값 조작 ---
    with col_b:
        st.markdown("**조작할 센서와 이탈 정도 (σ 단위)**")
        adjustments = {}                      # {센서번호: 몇 σ}
        for k in range(n_sliders):
            c1, c2 = st.columns([1, 2])
            with c1:
                sensor = st.selectbox(
                    f"센서 {k+1}", candidate_sensors, index=k, key=f"sensor_{k}"
                )
            with c2:
                sigma = st.slider(
                    f"센서 {sensor} 이탈량", -3.0, 5.0, 0.0, 0.1, key=f"sigma_{k}"
                )
            adjustments[sensor] = sigma

    st.divider()

    # --- 조작된 웨이퍼 만들기 ---
    # 원본 한 장을 복사한 뒤, 지정한 센서 값을 (중앙값 + kσ)로 교체
    modified = X_final.iloc[[base_wafer]].copy()
    for sensor, sigma in adjustments.items():
        med, sig = normal_median[sensor], normal_sigma[sensor]
        modified.iloc[0, modified.columns.get_loc(sensor)] = med + sigma * sig

    # --- 조작 전/후 예측 ---
    prob_before = prob_all[base_wafer]
    prob_after = model.predict_proba(modified.values)[0, 1]

    shap_before = shap_all[base_wafer]
    shap_after = model.get_booster().predict(
        xgb.DMatrix(modified.values), pred_contribs=True)[0, :-1]

    # --- 결과 표시 ---
    m1, m2, m3 = st.columns(3)
    m1.metric("조작 전 불량 확률", f"{prob_before:.3f}")
    # delta : 변화량을 화살표와 함께 표시
    m2.metric("조작 후 불량 확률", f"{prob_after:.3f}",
              delta=f"{prob_after - prob_before:+.3f}")
    verdict_after = "위험 (정밀검사)" if prob_after >= THRESHOLD else "통과"
    m3.metric(f"조작 후 판정 (임계값 {THRESHOLD})", verdict_after)

    # 판정이 뒤바뀌었는지 알림
    flipped_to_risk = prob_before < THRESHOLD <= prob_after
    flipped_to_pass = prob_after < THRESHOLD <= prob_before
    if flipped_to_risk:
        st.error("조작으로 인해 판정이 '통과'에서 '위험'으로 바뀌었습니다.")
    elif flipped_to_pass:
        st.success("조작으로 인해 판정이 '위험'에서 '통과'로 바뀌었습니다.")

    st.divider()

    # --- SHAP 변화량 분석 ---
    st.markdown("#### 조작으로 인한 판단 근거 변화 (SHAP 증가량 상위 10개)")

    delta_shap = shap_after - shap_before
    top_delta = np.argsort(np.abs(delta_shap))[::-1][:10]

    rows = []
    for i in top_delta:
        cname = X_final.columns[i]
        is_ind = isinstance(cname, str) and cname.startswith('m_')
        label = f"{cname} (결측)" if is_ind else f"센서 {cname}"
        was_adjusted = (not is_ind) and (cname in adjustments) and (adjustments[cname] != 0)
        rows.append({
            "센서": label,
            "조작 전": shap_before[i],
            "조작 후": shap_after[i],
            "변화량": delta_shap[i],
            "직접 조작": "O" if was_adjusted else "",
        })
    df_delta = pd.DataFrame(rows)

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.bar_chart(df_delta.set_index("센서")[["변화량"]], horizontal=True)
    with cc2:
        st.dataframe(df_delta, hide_index=True, use_container_width=True)

    st.caption(
        "변화량이 큰 센서일수록 이번 조작으로 판단 근거가 크게 바뀐 항목입니다. "
        "'직접 조작' 표시가 있는 센서가 실제로 슬라이더로 움직인 센서이며, "
        "이 항목이 변화량 상위에 나타나면 모델이 조작을 올바르게 인식했다는 의미입니다."
    )

    st.info(
        "이 시뮬레이션은 모델이 예측하는 불량 확률의 변화를 보여줍니다. "
        "모델은 센서 값과 불량 사이의 통계적 상관관계를 학습한 것이며, "
        "특정 센서를 조정하면 실제 수율이 개선된다는 인과관계를 의미하지 않습니다."
    )

    # ============================================
# 탭 3: 배치 시나리오 진단 + SPC 관리도
# ============================================
# 미리 만들어둔 5가지 이상 상황(각 100장 배치)을 골라 진단한다.
# - 관리도(SPC)로 배치 전체의 불량 확률 추이를 보고 관리한계 이탈을 표시
# - 어느 센서가 정상 대비 얼마나 벗어났는지 순위로 원인 진단
#
# 웨이퍼 1장의 결과로 설비를 조정하는 것은 과잉조정(over-adjustment) 오류이므로,
# 이 화면은 '배치 단위'로 판단한다.

with tab3:
    st.subheader("배치 단위 이상 감지 및 원인 진단")

    scenarios = scen_bundle['scenarios']
    base_batch = scen_bundle['base_batch']
    base_prob = scen_bundle['base_prob']
    n_inject = scen_bundle['n_inject']

    # --- 시나리오 선택 ---
    scen_names = {k: v['name'] for k, v in scenarios.items()}
    picked = st.selectbox(
        "진단할 배치 선택",
        list(scen_names.keys()),
        format_func=lambda k: scen_names[k],   # 화면에는 이름을 보여줌
    )
    sc = scenarios[picked]
    batch = sc['batch']
    prob = sc['prob']
    injected = sc['injected']

    # --- 배치 요약 지표 ---
    over = (prob >= THRESHOLD).sum()
    base_over = (base_prob >= THRESHOLD).sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("배치 크기", f"{len(prob)}장")
    k2.metric("위험 판정 웨이퍼", f"{over}장",
              delta=f"{over - base_over:+d}장 (정상 배치 대비)")
    k3.metric("배치 평균 불량확률", f"{prob.mean():.3f}",
              delta=f"{prob.mean() - base_prob.mean():+.3f}")
    k4.metric("위험 판정 비율", f"{over / len(prob) * 100:.1f}%")

    st.divider()

    # --- SPC 관리도 (기준선 기반) ---
    st.markdown("#### 관리도 (Control Chart)")

    bl_mean = spc['mean']
    bl_std = spc['std']
    ucl_count = spc['ucl']
    p95_lo, p95_hi = spc['p95_low'], spc['p95_high']
    n_trials = spc['n_trials']
    batch_unit = spc['batch_size']

    st.caption(
        f"정상 웨이퍼 배치({batch_unit}장)를 {n_trials}회 반복 추출하여 "
        f"'위험 판정 개수'의 자연 변동을 측정하고, 이로부터 관리한계를 산출했습니다."
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("정상 배치 기준값", f"{bl_mean:.2f}장", help=f"{batch_unit}장당 위험 판정 개수의 평균")
    b2.metric("표준편차", f"{bl_std:.2f}장")
    b3.metric("정상 변동 범위 (95%)", f"{p95_lo:.0f}~{p95_hi:.0f}장")
    b4.metric("관리 상한 (UCL)", f"{ucl_count:.1f}장", help="평균 + 3σ")

    # --- 확률 관리도 (웨이퍼별 추이) ---
    ucl_prob = base_prob.mean() + 3 * base_prob.std()
    chart_df = pd.DataFrame({
        "불량 확률": prob,
        "관리 상한(UCL)": ucl_prob,
        "중심선": base_prob.mean(),
        "운영 임계값": THRESHOLD,
    })
    st.line_chart(chart_df)
    st.caption("웨이퍼별 불량 확률 추이. 시나리오 S1~S4는 앞 30장이 조작 구간입니다.")

    # --- 통계적 판정 ---
    st.markdown("##### 통계적 판정")

    if picked != 'S5':
        detected_n = int((prob[:n_inject] >= THRESHOLD).sum())
        z_score = (detected_n - bl_mean) / bl_std
        over_ucl = detected_n > ucl_count

        j1, j2, j3 = st.columns(3)
        j1.metric("조작 구간 위험 판정", f"{detected_n}장 / {n_inject}장",
                  delta=f"{detected_n - bl_mean:+.1f}장 (기준값 대비)")
        j2.metric("Z값", f"{z_score:+.2f}", help="기준값에서 표준편차 몇 개만큼 벗어났는가")

        ev = spc['scenario_eval'].get(picked)
        if ev:
            j3.metric("p값", f"{ev['p']:.4f}",
                      help="정상 공정에서 이 결과가 우연히 나올 확률")

        if over_ucl:
            st.error(
                f"**관리한계 초과 — 이상 신호로 판정합니다.** "
                f"위험 판정 {detected_n}장은 관리 상한 {ucl_count:.1f}장을 초과하며, "
                f"정상 공정에서 이 수준이 우연히 발생할 확률은 "
                f"{ev['p']*100:.2f}%입니다. 공정 이상 원인 조사가 필요합니다."
            )
        elif z_score > 2:
            st.warning(
                f"**경계 상태 — 관리한계 이내이나 주의가 필요합니다.** "
                f"위험 판정 {detected_n}장은 관리 상한 {ucl_count:.1f}장을 넘지 않으나, "
                f"기준값보다 {z_score:.1f}σ 높습니다(p={ev['p']:.4f}). "
                f"3σ 관리한계는 오경보를 억제하기 위해 보수적으로 설정되므로, "
                f"이 구간의 이상은 단일 판정으로 놓칠 수 있습니다. "
                f"연속 관측이나 보조 판정 규칙이 필요한 영역입니다."
            )
        else:
            p_txt = f" (p={ev['p']:.4f})" if ev else ""
            st.info(
                f"**관리한계 이내 — 통계적으로 정상 공정과 구별되지 않습니다.** "
                f"위험 판정 {detected_n}장은 정상 변동 범위({p95_lo:.0f}~{p95_hi:.0f}장) "
                f"안에 있습니다{p_txt}. "
                f"주입한 이상이 모델의 검출 범위를 벗어났음을 의미합니다."
            )
    else:
        # 드리프트는 배치 전체가 변하므로 구간별로 평가
        st.caption("드리프트 시나리오는 배치 전체가 점진적으로 변하므로 구간별로 평가합니다.")
        seg_rows = []
        for s in range(0, 100, 20):
            e = s + 20
            cnt = int((prob[s:e] >= THRESHOLD).sum())
            # 20장 기준으로 환산한 기대값
            expected = bl_mean * (20 / batch_unit)
            seg_rows.append({
                "구간": f"{s}–{e-1}번",
                "위험 판정": f"{cnt} / 20장",
                "기대값": f"{expected:.1f}장",
                "평균 확률": f"{prob[s:e].mean():.3f}",
            })
        st.dataframe(pd.DataFrame(seg_rows), hide_index=True, use_container_width=True)
        st.info(
            "구간이 진행될수록 위험 판정 수와 평균 확률이 증가하는 경향을 확인할 수 있습니다. "
            "개별 웨이퍼로는 정상 판정되는 초기 단계에서도 배치 추세로는 악화가 드러납니다."
        )

    # --- 기준선 분포 시각화 ---
    with st.expander("정상 배치의 위험 판정 개수 분포 (기준선 산출 근거)"):
        counts = spc['baseline_counts']
        hist = pd.Series(counts).value_counts().sort_index()
        hist_df = pd.DataFrame({"발생 횟수": hist.values}, index=hist.index)
        hist_df.index.name = "위험 판정 개수"
        st.bar_chart(hist_df)
        st.markdown(
            f"""
            정상 웨이퍼만으로 구성된 {batch_unit}장 배치를 {n_trials}회 반복 추출한 결과입니다.

            - 평균 **{bl_mean:.2f}장**, 표준편차 **{bl_std:.2f}장**
            - 95% 구간 **{p95_lo:.0f}~{p95_hi:.0f}장**
            - 관리 상한(평균+3σ) **{ucl_count:.1f}장**

            모델은 정상 웨이퍼도 일정 비율로 위험 판정합니다(오탐).
            중요한 것은 **그 오탐 비율이 안정적**이라는 점입니다.
            개별 판정은 부정확해도 개수의 분포는 예측 가능하므로,
            이 분포를 기준선으로 삼아 배치 단위 이상을 판정할 수 있습니다.

            **한계** — 표준편차가 평균 대비 약 {bl_std/bl_mean*100:.0f}%로 변동이 작지 않습니다.
            {batch_unit}장은 통계적으로 작은 표본이며, 배치 크기를 늘리면
            표준오차가 표본 수의 제곱근에 반비례하여 감소하므로 검출력이 향상됩니다.
            """
        )

    st.divider()

    # --- 원인 진단: 어느 센서가 정상 대비 벗어났는가 ---
    st.markdown("#### 원인 진단 (정상 대비 이탈 상위 센서)")

    # 배치 앞부분(조작 구간)과 정상 기준을 비교해 이탈량(σ) 계산
    target_rows = batch.iloc[:n_inject] if picked != 'S5' else batch
    deviations = []
    for c in sensor_cols:
        med, sig = normal_median[c], normal_sigma[c]
        if sig is None or sig <= 0 or pd.isna(sig):
            continue
        v = target_rows[c].median()
        if pd.isna(v):
            continue
        deviations.append({
            "센서": c,
            "이탈량(σ)": (v - med) / sig,
            "배치 중앙값": v,
            "정상 중앙값": med,
        })

    df_dev = pd.DataFrame(deviations)
    # 이탈 크기순 정렬 (절댓값 기준)
    df_dev["절대이탈"] = df_dev["이탈량(σ)"].abs()
    df_dev = df_dev.sort_values("절대이탈", ascending=False).head(10)

    # 실제로 조작한 센서를 표시해 진단이 맞았는지 확인 가능하게
    df_dev["실제 조작 센서"] = df_dev["센서"].apply(
        lambda c: "O" if c in injected else ""
    )
    df_dev["센서"] = df_dev["센서"].apply(lambda c: f"센서 {c}")

    d1, d2 = st.columns([1, 1])
    with d1:
        st.bar_chart(df_dev.set_index("센서")[["이탈량(σ)"]], horizontal=True)
    with d2:
        st.dataframe(
            df_dev[["센서", "이탈량(σ)", "배치 중앙값", "정상 중앙값", "실제 조작 센서"]],
            hide_index=True, use_container_width=True
        )

    # 진단 정확도 평가
    detected_set = set(
        int(s.replace("센서 ", "")) for s in df_dev["센서"]
    )
    hit = len(detected_set & set(injected))
    st.info(
        f"이 배치에서 실제로 이상이 주입된 센서는 {injected} 입니다. "
        f"이탈 상위 10개 센서 중 {hit}개가 일치했습니다."
    )

    st.caption(
        "이 진단은 배치 단위 통계에 기반합니다. 웨이퍼 1장의 예측 결과만으로 "
        "설비를 조정하는 것은 통계적 공정 관리 관점에서 과잉조정에 해당하므로, "
        "배치 수준의 이탈 경향을 근거로 판단합니다."
    )

# ============================================
# 탭 4: 모델 성능 및 검증 근거
# ============================================
# 이 화면의 목적은 "성능이 좋다"가 아니라
# "이 숫자가 정직하게 측정되었고, 한계를 정확히 파악했다"를 보이는 것.

with tab5:
    st.subheader("모델 성능 및 검증 방법론")

    # --------------------------------------------
    # 시스템 역할 정의 (프레임 명시)
    # --------------------------------------------
    st.markdown("#### 시스템의 역할 정의")
    st.info(
        "본 시스템은 개별 웨이퍼의 합격/불합격을 판정하는 도구가 아니라, "
        "**100장 단위 배치에서 공정 이상을 감지하고 원인 센서를 규명하는 진단 도구**입니다. "
        "웨이퍼 1장의 예측만으로 설비를 조정하는 것은 통계적 공정 관리 관점에서 "
        "과잉조정(over-adjustment)에 해당하므로, 배치 단위 통계를 근거로 판단합니다."
    )

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            """
            **강점 (배치 단위)**

            - 원인 지목 정확도 **87~90%**
            - 통제 조건 하 배치 이상 감지 (오탐 불변)
            - 단일 관리도가 놓치는 조합 이상 포착
            - 점진적 드리프트 조기 감지
            """
        )
    with r2:
        st.markdown(
            """
            **한계 (개별 웨이퍼 단위)**

            - Recall **52.9%** — 불량 104장 중 49장 미검출
            - Precision **15.6%** — 위험 판정 6~7장 중 1장만 실제 불량
            - 개별 판정에 근거한 의사결정은 부적절
            """
        )

    st.divider()

    # --------------------------------------------
    # 핵심 지표
    # --------------------------------------------
    st.markdown("#### 검증 결과")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("5-Fold PR-AUC", "0.199 ± 0.044")
    p2.metric("무작위 기준선 대비", "약 3.0배", help="PR-AUC의 무작위 기준선은 0.5가 아니라 양성 비율 0.066")
    p3.metric("원인 지목 정확도", "87~90%")
    p4.metric("운영 임계값", f"{THRESHOLD}")

    st.warning(
        "**PR-AUC 해석 시 주의** — PR-AUC의 무작위 기준선은 0.5가 아니라 양성 클래스 비율(0.066)입니다. "
        "0.199는 절대적으로 낮은 값이지만 무작위 대비로는 약 3배 수준입니다."
    )

    st.divider()

    # --------------------------------------------
    # Recall / Precision 구분
    # --------------------------------------------
    st.markdown("#### Recall과 Precision의 구분")
    st.caption("두 지표는 분모가 다릅니다. 혼동하기 쉬운 지점입니다.")

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            r"""
            $$\text{Recall} = \frac{TP}{TP + FN} = \frac{55}{104} = 52.9\%$$

            **실제 불량 104장 중** 55장 검출.
            분모가 실제 불량 전체.
            → 불량 유출 방지 관점
            """
        )
    with q2:
        st.markdown(
            r"""
            $$\text{Precision} = \frac{TP}{TP + FP} = \frac{55}{353} = 15.6\%$$

            **위험 판정 353장 중** 55장이 실제 불량.
            분모가 판정 건수.
            → 불필요한 검사 비용 관점
            """
        )

    st.markdown(
        """
        **임계값 0.20 운영 시 실제 분포**

        | 구분 | 수량 |
        |---|---|
        | 전체 웨이퍼 | 1,567장 |
        | 위험 판정 | 353장 (22.5%) |
        | └ 실제 불량 | 55장 |
        | └ 실제 정상 (오탐) | 298장 |
        | 미검출 불량 | **49장** |

        불량 1장 검출을 위해 평균 6.4장을 지목합니다.
        """
    )

    st.divider()

    # --------------------------------------------
    # 임계값 선정 근거
    # --------------------------------------------
    st.markdown("#### 임계값 선정 근거")
    df_thr = pd.DataFrame({
        "임계값": [0.50, 0.40, 0.30, 0.20],
        "검출(/104)": [22, 26, 37, 55],
        "오탐(/1463)": [56, 97, 166, 298],
        "Recall": [0.212, 0.250, 0.356, 0.529],
        "Precision": [0.282, 0.211, 0.182, 0.156],
        "F1": [0.242, 0.229, 0.241, 0.241],
    })
    t1, t2 = st.columns([1, 1])
    with t1:
        st.dataframe(df_thr, hide_index=True, use_container_width=True)
    with t2:
        st.line_chart(df_thr.set_index("임계값")[["Recall", "Precision"]])
    st.caption(
        "F1은 어느 임계값에서도 0.23~0.24로 거의 변하지 않습니다. "
        "즉 임계값 선택은 성능 향상이 아니라 Recall과 Precision의 배분을 결정하는 문제입니다. "
        "배치 단위 이상 감지를 목적으로 하므로 검출 쪽에 무게를 둔 0.20을 채택했습니다."
    )

    st.divider()

    # --------------------------------------------
    # 평가지표 선택
    # --------------------------------------------
    st.markdown("#### 정확도(Accuracy)와 ROC-AUC를 사용하지 않는 이유")
    a1, a2 = st.columns(2)
    with a1:
        st.metric("불량 비율", "6.64%", help="104 / 1,567장")
        st.error(
            "**정확도** — 모든 웨이퍼를 정상으로 예측해도 93.4%가 산출됩니다. "
            "실제로 초기 모델이 정확도 92.7%를 기록하면서 불량 21장 중 0장을 검출했습니다."
        )
    with a2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.error(
            "**ROC-AUC** — False Positive Rate의 분모가 정상 샘플 전체(1,463장)입니다. "
            "정상이 압도적으로 많으면 오탐이 상당수 발생해도 지표가 크게 나빠지지 않아 "
            "성능이 실제보다 좋아 보입니다."
        )
    st.success(
        "**대안** — Recall(유출 방지), Precision(검사 비용), PR-AUC(임계값 비의존 종합)를 사용합니다."
    )

    st.divider()

    # --------------------------------------------
    # 검증 방법론
    # --------------------------------------------
    st.markdown("#### 검증 방법론")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown(
            """
            **5-Fold 교차검증 채택 이유**

            단일 분할 시 검증용 불량이 21장에 불과해 평가가 분할 방식에 좌우됩니다.
            실제로 fold별 PR-AUC가 0.158 ~ 0.261로 편차가 컸습니다.

            5-Fold 방식에서는 불량 104장 전체가 한 번씩 검증 대상이 되며,
            각 웨이퍼는 자신을 제외한 데이터로 학습된 모델에게 평가받습니다.
            **학습 데이터를 평가한 것이 아닙니다.**
            """
        )
        st.bar_chart(
            pd.DataFrame({"PR-AUC": [0.158, 0.241, 0.160, 0.172, 0.261]},
                         index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"])
        )
    with v2:
        st.markdown(
            """
            **데이터 누수 차단**

            | 항목 | 조치 |
            |---|---|
            | 전처리 | 결측 지시자 기준을 fold 내 학습 데이터에서만 산출 |
            | 임계값 | 검증 성능이 아닌 out-of-fold 예측으로 결정 |
            | 오버샘플링 | SMOTE 미사용 |

            **SMOTE를 사용하지 않은 이유**

            590차원에서 104개 소수 샘플로 합성 데이터를 생성하면 신뢰하기 어렵습니다.
            또한 분할 이전에 SMOTE를 적용하면 테스트셋 정보가 합성 샘플에 유입되어
            성능이 인위적으로 부풀려집니다. 이는 불균형 학습에서 가장 흔한 누수 유형입니다.

            대신 `scale_pos_weight`로 손실 함수에서 소수 클래스 가중치를 조정했습니다.
            """
        )

    st.divider()

    # --------------------------------------------
    # 개선 시도 기록
    # --------------------------------------------
    st.markdown("#### 성능 개선 시도 기록")
    st.caption("서로 다른 방향의 접근이 모두 PR-AUC 0.19~0.21 구간에 수렴했습니다.")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**전처리·설정 실험**")
        st.dataframe(pd.DataFrame({
            "방식": ["중앙값 대치", "NaN 유지", "NaN + 결측지시자",
                   "얕은 트리 + 강한 규제", "결합 (채택)", "특성 선별 (상위 40개)"],
            "PR-AUC": ["0.186 ± 0.051", "0.191 ± 0.037", "0.191 ± 0.037",
                       "0.190 ± 0.018", "0.206 ± 0.019", "0.125 (악화)"],
        }), hide_index=True, use_container_width=True)
        st.caption(
            "중앙값 대치를 제거하고 결측 여부를 특성으로 추가하자 "
            "점수와 안정성이 함께 개선되었습니다(표준편차 0.051 → 0.019). "
            "특성 선별은 오히려 악화되었는데, 신호가 약하게 분산된 데이터에서 "
            "특성을 줄이면 잡음과 함께 미약한 신호도 제거되기 때문입니다."
        )
    with e2:
        st.markdown("**모델 비교 (동일 5-Fold, 동일 전처리)**")
        st.dataframe(pd.DataFrame({
            "모델": ["XGBoost (채택)", "Random Forest", "Logistic Regression"],
            "PR-AUC": ["0.193 ± 0.028", "0.201 ± 0.040", "0.149 ± 0.029"],
            "최적 F1": [0.237, 0.253, 0.191],
        }), hide_index=True, use_container_width=True)
        st.caption(
            "Random Forest가 근소하게 높으나 표준편차 범위 내로 유의미한 차이가 아닙니다. "
            "Logistic Regression은 명확히 낮습니다. "
            "**모델 교체로는 개선되지 않음을 확인했습니다.**"
        )

    st.divider()

    # --------------------------------------------
    # 외부 벤치마크
    # --------------------------------------------
    st.markdown("#### 외부 연구와의 비교")
    st.markdown(
        """
        2026년 발표된 경량 트랜스포머 벤치마크 연구(arXiv:2606.24173)는 SECOM을 포함한 3개 데이터셋에서
        Random Forest, XGBoost, SVM, Logistic Regression 및 DistilBERT, TinyBERT, MobileBERT를 비교했습니다.
        해당 연구는 SECOM에서 **모든 방법을 통틀어 최고 F1이 13.6%**였다고 보고했으며,
        전통적 방법과 트랜스포머 모두 심한 불균형 데이터에서 실패한다고 결론지었습니다.
        """
    )
    st.warning(
        "**비교 시 주의** — 본 프로젝트의 F1(약 0.24)이 위 수치보다 높으나, "
        "평가 분할 방식, 임계값 설정, 연구 목적(3개 데이터셋 across 표준 설정 비교 vs 단일 데이터 집중 튜닝)이 "
        "달라 직접적인 우열 비교는 성립하지 않습니다. "
        "**본 프로젝트의 결과가 공개 벤치마크와 동일한 수준대에 위치하며, "
        "이것이 SECOM 데이터의 구조적 한계를 시사한다**는 점까지만 확인할 수 있습니다."
    )

    st.divider()

    # --------------------------------------------
    # 시나리오 검증
    # --------------------------------------------
    st.markdown("#### 시나리오 기반 기능 검증")
    st.caption("정상 웨이퍼 100장 중 앞 30장에만 이상을 주입하고, 나머지 70장은 조작하지 않은 통제 실험입니다.")

    st.dataframe(pd.DataFrame({
        "시나리오": ["기준선 (주입 전)", "S1 단일 (센서59 +3σ)", "S2 단일 (센서130 +3σ)",
                   "S3 다중 (59·33·21 각 +2σ)", "S4 다중 (33·130·460 각 +2σ)"],
        "검출(/30)": ["2", "13", "4", "17", "11"],
        "오탐(/70)": ["11", "11", "11", "11", "11"],
        "원인 지목": ["-", "27/30", "26/30", "평균 2.1/3", "평균 1.9/3"],
    }), hide_index=True, use_container_width=True)

    st.success(
        "**통제 조건 성립** — 모든 시나리오에서 조작하지 않은 70장의 오탐이 11장으로 "
        "동일하게 유지되었습니다. 검출률 증가가 주입한 이상에서 기인했음을 확인할 수 있습니다."
    )

    st.markdown(
        """
        **1. 센서별 검출 민감도 차이** — 동일한 +3σ 이탈에서 센서 59는 13장, 센서 130은 4장으로
        약 3배 차이가 발생했습니다. 저민감 항목은 모델만으로 감지하기 어려우므로
        별도 관리도나 계측 강화가 필요하다는 근거가 됩니다.

        **2. 검출 능력과 원인 규명 능력의 분리** — 검출 민감도 차이(13장 vs 4장)와 무관하게
        원인 지목 정확도는 27/30, 26/30으로 일관되게 높았습니다.
        검출된 이상에 대해서는 원인 센서를 신뢰성 있게 특정할 수 있습니다.

        **3. 다변량 분석의 유효성** — 센서 130 단독 +3σ는 4장 검출에 그쳤으나,
        동일 센서가 포함된 조합의 **+2σ 이탈은 11장이 검출**되었습니다.
        개별 이탈 크기가 더 작음에도 검출이 증가한 것으로,
        단일 센서 관리도로는 감지되지 않는 수준의 이탈도 조합 분석으로 포착 가능함을 의미합니다.

        **4. 점진적 악화 감지** — 드리프트 시나리오(S5)에서 검출률이 구간별로 4→3→5→8→7로
        증가했습니다. 개별 웨이퍼로는 정상 판정되는 초기 단계에서도 배치 추세로는 악화가 드러납니다.
        """
    )

    st.divider()

    # --------------------------------------------
    # 한계
    # --------------------------------------------
    st.markdown("#### 분석의 한계")
    st.warning(
        """
        **1. 개별 판정 신뢰도** — Recall 52.9%, Precision 15.6%로 개별 웨이퍼 판정에 근거한
        의사결정은 부적절합니다. 배치 단위 통계를 사용해야 합니다.

        **2. 데이터셋 한계** — 센서가 익명화되어 있어 원인 센서를 특정 공정 단계와 연결하는 해석이
        불가능합니다. 도메인 지식 기반 특성 공학도 적용할 수 없습니다.

        **3. 상관관계와 인과관계** — 모델은 센서 값과 불량 사이의 통계적 상관을 학습합니다.
        특정 센서를 조정하면 수율이 개선된다는 인과관계를 보장하지 않습니다.
        What-if 시뮬레이션 결과는 예측 확률의 변화이며 실제 수율 변화가 아닙니다.

        **4. 검출 한계(포화)** — 트리 기반 모델은 학습 데이터에서 관측된 분기점 범위를 초과하면
        반응이 포화됩니다. 센서 59의 경우 분기점 범위가 0.56~10.24이며,
        본 모델의 민감 구간은 약 +1σ ~ +2σ입니다.

        **5. 시뮬레이션 기반 검증** — 시나리오 검증은 인위적으로 주입한 이상에 대한 반응이며,
        실제 공정의 이상 패턴과 다를 수 있습니다.
        """
    )

    # ============================================
# 탭 4: 요인설계(DOE) 기반 상호작용 분석
# ============================================
# 센서 3개를 각각 정상(-)/이탈(+) 2수준으로 두고 8가지 조합을 전부 검증하여
# 각 센서의 주효과와 센서 간 교호작용을 분리해 정량화한다.
#
# 목적: "개별 항목은 관리 범위 내인데 조합에서 문제가 발생"하는 구조를 수치로 확인.
#       공정 통합(PI) 관점의 모듈 간 간섭 분석에 대응하는 방법론.

with tab4:
    st.subheader("2³ 요인설계 기반 센서 상호작용 분석")

    factors = doe['factors']
    level = doe['level']
    df_design = doe['design_table']
    eff_mean = doe['effects_mean']
    eff_std = doe['effects_std']

    st.info(
        f"**설계**: 센서 {factors}를 각각 정상(−) / +{level}σ 이탈(+) 2수준으로 두고 "
        f"가능한 8가지 조합을 전부 실행했습니다. "
        f"이탈 수준 +{level}σ는 일반적인 관리한계(±3σ) 이내이므로, "
        f"**단일 센서 관리도에서는 모두 정상으로 판정되는 조건**입니다."
    )

    st.divider()

    # --------------------------------------------
    # 설계표
    # --------------------------------------------
    st.markdown("#### 설계표 및 실행 결과")
    d1, d2 = st.columns([1, 1])
    with d1:
        st.dataframe(df_design, hide_index=True, use_container_width=True)
        st.caption("조건 표기: 세 자리가 각각 센서의 상태. '−'는 정상, '+'는 이탈.")
    with d2:
        chart_design = df_design.set_index('조건')[['평균확률', '검출률']]
        st.bar_chart(chart_design)
        st.caption("이탈 센서 수가 늘수록 응답이 단조 증가합니다.")

    st.divider()

    # --------------------------------------------
    # 효과 분석
    # --------------------------------------------
    st.markdown("#### 주효과 및 교호작용")
    st.caption("응답변수: 평균 불량확률 / 5회 반복 실험으로 표준편차 산출")

    eff_rows = []
    label_map = {
        'X1': f'주효과 · 센서 {factors[0]}',
        'X2': f'주효과 · 센서 {factors[1]}',
        'X3': f'주효과 · 센서 {factors[2]}',
        'X1X2': f'교호 · {factors[0]}×{factors[1]}',
        'X1X3': f'교호 · {factors[0]}×{factors[2]}',
        'X2X3': f'교호 · {factors[1]}×{factors[2]}',
        'X1X2X3': f'교호 · 3차 ({factors[0]}×{factors[1]}×{factors[2]})',
    }
    for k, label in label_map.items():
        m_, s_ = eff_mean.get(k, 0), eff_std.get(k, 0)
        eff_rows.append({
            "항목": label,
            "효과": m_,
            "표준편차": s_,
            "유의성": "유의" if abs(m_) > 2 * s_ else "미미",
        })
    df_eff_view = pd.DataFrame(eff_rows)

    e1, e2 = st.columns([1, 1])
    with e1:
        st.dataframe(df_eff_view, hide_index=True, use_container_width=True)
    with e2:
        st.bar_chart(df_eff_view.set_index("항목")[["효과"]], horizontal=True)

    st.markdown(
        f"""
        **해석**

        - **주효과** — 센서 {factors[0]}이 {eff_mean.get('X1', 0):.4f}로 가장 크며,
          나머지 두 센서의 약 3배입니다. 단일 항목 감시 우선순위가 가장 높습니다.
        - **2차 교호작용** — 세 조합 모두 **양(+)이며 통계적으로 유의**합니다.
          두 센서가 함께 이탈하면 각각의 효과 합보다 큰 반응이 나타납니다.
        - **3차 교호작용** — 사실상 0입니다. 이는 데이터에 3차 상호작용이 없다는 뜻이 아니라,
          모델을 `max_depth=2`로 설정하여 트리 하나가 최대 2개 변수까지만 조합하기 때문입니다.
          **모델 구조가 표현할 수 있는 상호작용의 차수에 제약이 있습니다.**
        """
    )

    st.divider()

    # --------------------------------------------
    # 가법성 검정 (핵심 결과)
    # --------------------------------------------
    st.markdown("#### 가법성 검정 — 효과는 단순히 더해지는가")

    solo = doe['solo_effects']
    add_sum = doe['additive_sum']
    combined = doe['combined_effect']
    base_p = doe['base_prob']
    gap = combined - add_sum
    gap_pct = (combined / add_sum - 1) * 100

    g1, g2, g3 = st.columns(3)
    g1.metric("단순 합 (독립 가정)", f"{add_sum:+.4f}")
    g2.metric("실제 동시 투입", f"{combined:+.4f}", delta=f"{gap:+.4f}")
    g3.metric("교호작용 크기", f"{gap_pct:+.1f}%")

    df_solo = pd.DataFrame({
        "조건": [f"센서 {c} 단독" for c in factors] + ["단순 합 (가법 가정)", "실제 동시 투입"],
        "불량확률 증가분": [solo[c] for c in factors] + [add_sum, combined],
    })
    s1, s2 = st.columns([1, 1])
    with s1:
        st.dataframe(df_solo, hide_index=True, use_container_width=True)
    with s2:
        st.bar_chart(df_solo.set_index("조건"), horizontal=True)

    st.error(
        f"**세 센서를 각각 따로 이탈시킨 효과의 합은 {add_sum:.4f}이나, "
        f"동시에 이탈시키면 {combined:.4f}로 {gap_pct:.1f}% 큰 반응이 나타납니다.** "
        f"각 센서의 이탈 수준(+{level}σ)은 관리한계 이내이므로 "
        f"단일 센서 관리도에서는 세 항목 모두 정상 판정을 받습니다."
    )

    st.divider()

    # --------------------------------------------
    # 공정 통합 관점의 해석
    # --------------------------------------------
    st.markdown("#### 공정 통합(Process Integration) 관점의 해석")
    st.markdown(
        f"""
        본 분석 결과는 공정 통합 단계에서 자주 발생하는 문제 구조와 대응합니다.

        **문제 구조** — 개별 단위공정이 각각의 규격을 만족함에도 통합 결과물이 불량인 경우.
        각 모듈 담당자는 자신의 공정이 관리 범위 내라고 판단하지만, 모듈 간 상호작용에서
        문제가 발생합니다.

        **본 실험에서의 대응** — 센서 {factors}를 각각 +{level}σ만 이탈시켰습니다.
        이는 관리한계 ±3σ 이내이므로 단일 항목 관리도에서는 전부 정상으로 통과합니다.
        그러나 세 항목이 동시에 이 상태일 때, 예상보다 {gap_pct:.0f}% 큰 이탈이 발생했습니다.

        **의미** — 단일 센서 관리도(SPC)만으로는 이 구조를 감지할 수 없습니다.
        다변량 분석이 기존 통계적 공정 관리의 사각지대를 보완하는 지점이 여기입니다.

        **한계** — SECOM 데이터는 센서가 익명화되어 있어 각 센서가 어느 공정 모듈에
        해당하는지 특정할 수 없습니다. 실제 공정 통합 업무에서는 본 분석으로 도출한
        상호작용을 물리적 메커니즘과 연결하는 과정이 추가로 필요합니다.
        """
    )

    st.divider()

    # --------------------------------------------
    # 방법론 주석
    # --------------------------------------------
    with st.expander("방법론 상세 — 응답변수 선택과 효과 계산"):
        st.markdown(
            r"""
            **효과 계산 방식 (코드화 계수)**

            각 요인을 $-1$(정상) / $+1$(이탈)로 코드화한 뒤, 효과는 다음과 같이 계산합니다.

            $$\text{효과} = \bar{y}_{(+1)} - \bar{y}_{(-1)}$$

            해당 요인이 $+1$인 조건들의 응답 평균에서 $-1$인 조건들의 평균을 뺍니다.
            다른 요인들은 $+1$과 $-1$이 균형 있게 배치되어 있어 상쇄되므로,
            **오직 해당 요인의 효과만 분리됩니다.**

            교호작용은 두 요인의 부호를 곱한 값을 기준으로 동일하게 계산합니다.

            **응답변수 선택**

            응답변수로 '평균 불량확률'(연속형)과 '검출률'(임계값 초과 비율)을 모두 검토했습니다.
            검출률로 분석할 경우 일부 교호작용이 통계적으로 유의하지 않게 나타나는데,
            이는 이진 판정 과정에서 정보가 손실되기 때문입니다.
            **DOE에서는 가능한 한 연속형 응답변수를 사용하는 것이 표준이며**,
            본 분석은 평균 불량확률을 주 응답변수로 사용했습니다.

            **반복 실험**

            서로 다른 웨이퍼 표본으로 5회 반복하여 각 효과의 표준편차를 산출했습니다.
            효과의 절댓값이 표준편차의 2배를 초과하는 경우 유의한 것으로 판정했습니다.
            """
        )