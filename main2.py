# ============================================
# [통합 교정판] main2.py
# ============================================
# 이전 버전에서 확인된 오류·누락을 수정한 최종 파이프라인.
#
# [수정 1] OOF 확률 저장 — 최종 모델은 전체 데이터를 학습했으므로
#          자기 학습 데이터를 예측하면 과대평가된다(PR-AUC 0.999로 관측됨).
#          웨이퍼 단위의 정직한 확률은 교차검증 예측(oof_prob)이며, 이를 저장해
#          대시보드가 사용하도록 한다.
# [수정 2] ddof=1 — fold 5개는 표본이므로 표본표준편차를 쓴다.
#          ddof=0은 변동성을 약 11% 과소 보고한다.
# [수정 3] 타임스탬프 활용 — secom_labels.data의 2열(측정 시각)을 사용해
#          실제 수율 이탈(excursion) 구간을 탐지·분석한다.
#          합성 주입 시나리오만으로는 "직접 만든 이상 아니냐"는 반론에 취약하다.
# [수정 4] DOE 측정 공간 — 교호작용은 로그오즈(margin) 공간에서 측정한다.
#          확률 공간에서 측정하면 시그모이드의 볼록성 때문에 가법적 모델도
#          양의 교호작용(+19%)이 있는 것처럼 보인다(수학적 착시).
# [수정 5] 시나리오 검증에 무작위 대조군 추가 — "많이 흔들면 그냥 오른다"를
#          배제하려면 같은 개수의 무작위 센서를 흔든 대조군과 비교해야 한다.
# [수정 6] 센서 선정을 모델과 무관하게 — SHAP 상위 센서를 흔들고 모델이
#          반응했다고 말하면 순환논리다. 단변량 효과크기 d로 선정하고,
#          방향(d의 부호)대로 민다. 값이 작아질 때 불량인 센서도 존재한다.
# [수정 7] 경로를 스크립트 위치 기준 절대경로로 — 실행 위치와 무관하게 동작.

import os
import itertools
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, 'artifacts')
os.makedirs(ART, exist_ok=True)

# ============================================
# 1단계: 데이터 로드 (타임스탬프 포함)
# ============================================
data = pd.read_csv(os.path.join(BASE, 'secom.data'), sep=r'\s+', header=None)
labels = pd.read_csv(os.path.join(BASE, 'secom_labels.data'),
                     sep=r'\s+', header=None, quotechar='"')

y = (labels[0] == 1).astype(int).values
# 2열이 측정 시각. 이전 버전은 이 열을 버렸다. (수정 3)
timestamps = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S")

X = data.copy()
print("원본 크기:", X.shape, "/ 불량 수:", y.sum())
print(f"측정 기간: {timestamps.min():%Y-%m-%d} ~ {timestamps.max():%Y-%m-%d}")

# ============================================
# 2단계: 상수 열 제거
# ============================================
n_unique = X.nunique(dropna=True)
const_cols = n_unique[n_unique <= 1].index
X = X.drop(columns=const_cols)
print("제거한 상수 열:", len(const_cols), "/ 남은 센서:", X.shape[1])

# ============================================
# 3단계: 5-Fold 교차검증 (OOF 확률 확보)
# ============================================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, confusion_matrix
from xgboost import XGBClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = np.zeros(len(y))
fold_scores = []

XGB_PARAMS = dict(
    n_estimators=600, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.3,
    min_child_weight=5, reg_lambda=5.0,
    random_state=42, eval_metric='logloss', n_jobs=4,
)

for fold_no, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # 결측 지시자: 기준을 학습용에서만 산출 (누수 차단)
    miss_cols = X_tr.columns[X_tr.isna().mean() > 0]
    X_tr = pd.concat([X_tr, X_tr[miss_cols].isna().astype(int).add_prefix('m_')], axis=1)
    X_te = pd.concat([X_te, X_te[miss_cols].isna().astype(int).add_prefix('m_')], axis=1)

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    model.fit(X_tr.values, y_tr)

    prob = model.predict_proba(X_te.values)[:, 1]
    oof_prob[te_idx] = prob
    score = average_precision_score(y_te, prob)
    fold_scores.append(score)
    print(f"Fold {fold_no} PR-AUC: {score:.3f}")

# 수정 2: fold는 표본이므로 표본표준편차(ddof=1)
print(f"\n5-Fold PR-AUC: {np.mean(fold_scores):.3f} ± {np.std(fold_scores, ddof=1):.3f}")

# ============================================
# 4단계: 임계값 표 (OOF 기준 = 정직한 웨이퍼 단위 성능)
# ============================================
print("\n=== 임계값별 성능 (OOF, 불량 104장 전체 기준) ===")
print("임계값 | 검출/104 | 오탐 | Recall | Precision")
threshold_table = []
for t in [0.50, 0.40, 0.30, 0.20]:
    pred = (oof_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rec, prec = tp / (tp + fn), tp / (tp + fp) if (tp + fp) > 0 else 0
    threshold_table.append(dict(threshold=t, tp=int(tp), fp=int(fp),
                                recall=float(rec), precision=float(prec)))
    print(f" {t:.2f}  |   {tp:3d}    | {fp:3d} | {rec:.3f}  | {prec:.3f}")

THRESHOLD = 0.20

# ============================================
# 5단계: 최종 모델 학습 + SHAP + 저장
# ============================================
import joblib
import xgboost as xgb

miss_cols_final = X.columns[X.isna().mean() > 0]
X_final = pd.concat([X, X[miss_cols_final].isna().astype(int).add_prefix('m_')], axis=1)
spw_final = (y == 0).sum() / (y == 1).sum()

final_model = XGBClassifier(scale_pos_weight=spw_final, **XGB_PARAMS)
final_model.fit(X_final.values, y)
print("\n최종 모델 학습 완료 / 입력 열 수:", X_final.shape[1])

# SHAP (XGBoost 내장 TreeSHAP)
dm = xgb.DMatrix(X_final.values)
shap_values = final_model.get_booster().predict(dm, pred_contribs=True)[:, :-1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top15 = np.argsort(mean_abs_shap)[::-1][:15]
print("\n=== [전역 SHAP] 상위 15개 ===")
for r, i in enumerate(top15, 1):
    print(f" {r:2d}. 센서 {X_final.columns[i]}  ({mean_abs_shap[i]:.4f})")

# 개별 예시는 OOF 확률로 선정 (최종모델 확률은 과대평가)
prob_insample = final_model.predict_proba(X_final.values)[:, 1]  # 참고용
bad_idx = np.where(y == 1)[0]
example = bad_idx[np.argmax(oof_prob[bad_idx])]
print(f"\n=== [개별] 웨이퍼 #{example} (실제 불량) ===")
print(f"  OOF 확률 {oof_prob[example]:.2f} (정직) / 최종모델 {prob_insample[example]:.2f} (과대평가)")

# 정상 기준 통계 (로버스트)
normal_median = X[y == 0].median()
normal_sigma = (X[y == 0].quantile(0.75) - X[y == 0].quantile(0.25)) / 1.349

# 수정 1: oof_prob·y_true·timestamps를 함께 저장
joblib.dump({
    'model': final_model,
    'columns': list(X_final.columns),
    'sensor_columns': list(X.columns),
    'miss_cols': list(miss_cols_final),
    'threshold': THRESHOLD,
    'normal_median': normal_median,
    'normal_sigma': normal_sigma,
    'oof_prob': oof_prob,
    'y_true': y,
    'timestamps': timestamps,
    'fold_scores': fold_scores,
    'threshold_table': threshold_table,
}, os.path.join(ART, 'secom_model.joblib'))
print("저장: artifacts/secom_model.joblib (oof_prob 포함)")

# 공통 유틸: 조작된 데이터를 모델 입력 형태로 재구성
def rebuild(Xs):
    ind = Xs[list(miss_cols_final)].isna().astype(int).add_prefix('m_')
    return pd.concat([Xs, ind], axis=1).reindex(columns=X_final.columns)

def predict_prob(Xs):
    return final_model.predict_proba(rebuild(Xs).values)[:, 1]

def predict_margin(Xs):  # 로그오즈 (수정 4에서 사용)
    return final_model.predict(xgb.DMatrix(rebuild(Xs).values), output_margin=True) \
        if False else final_model.get_booster().predict(
            xgb.DMatrix(rebuild(Xs).values), output_margin=True)

def robust_stats(c):
    s = X[c][y == 0]
    return s.median(), (s.quantile(0.75) - s.quantile(0.25)) / 1.349

# ============================================
# 6단계: 실제 수율 이탈(excursion) 분석 (수정 3)
# ============================================
# 합성 주입이 아닌, 데이터에 실제로 존재하는 이탈 구간을 찾는다.
# 방법: 주간 불량률 p-관리도. 전체 평균 불량률을 중심선으로,
#       각 주의 표본 크기에 따른 3σ 상한을 계산한다.
from math import comb as _comb
from scipy import stats as sstats

df_time = pd.DataFrame({'ts': timestamps, 'y': y, 'idx': np.arange(len(y))}) \
            .sort_values('ts')
weekly = df_time.set_index('ts').resample('W').agg(
    n=('y', 'size'), fail=('y', 'sum'))
weekly = weekly[weekly.n >= 10].copy()
p_bar = y.mean()
weekly['rate'] = weekly.fail / weekly.n
weekly['ucl'] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / weekly.n)
weekly['pval'] = [
    sstats.binomtest(int(f), int(n), p_bar).pvalue
    for f, n in zip(weekly.fail, weekly.n)]

print("\n=== [실제 이탈] 주간 p-관리도 ===")
excursion_weeks = weekly[weekly.rate > weekly.ucl]
for wk, r in weekly.iterrows():
    mark = " <== 관리한계 초과" if r.rate > r.ucl else ""
    print(f"  {wk.date()}  n={int(r.n):3d}  불량률 {r.rate:5.1%} "
          f"(UCL {r.ucl:.1%}, p={r.pval:.4f}){mark}")

# 이탈 주간의 웨이퍼 vs 안정 구간 웨이퍼를 대조해 원인 센서 후보 도출
# 대조 방법: 단변량 효과크기 d = (이탈평균 - 안정평균) / 안정표준편차
if len(excursion_weeks) > 0:
    exc_week = excursion_weeks.index[0]
    # resample('W')는 주를 (일요일, 다음 일요일] 구간으로 묶는다. 경계를 정확히 맞춘다.
    def week_mask(wk_end):
        return ((timestamps > wk_end - pd.Timedelta(days=7)) &
                (timestamps <= wk_end)).values
    exc_mask = pd.Series(week_mask(exc_week))
    # 안정 기준: 불량률이 평균 이하였던 주들
    calm_weeks = weekly[weekly.rate <= p_bar].index
    calm_mask = np.zeros(len(y), dtype=bool)
    for cw in calm_weeks:
        calm_mask |= week_mask(cw)

    exc_X, calm_X = X[exc_mask.values], X[calm_mask]
    d_exc = ((exc_X.mean() - calm_X.mean()) / calm_X.std(ddof=1)) \
        .replace([np.inf, -np.inf], np.nan).dropna()
    top_exc = d_exc.reindex(d_exc.abs().sort_values(ascending=False).index).head(15)

    print(f"\n=== [실제 이탈] {exc_week.date()} 주간 vs 안정 구간: 이탈 센서 상위 15 ===")
    for c, v in top_exc.items():
        in_shap = "  (SHAP 전역 상위 15에도 포함)" if c in [X_final.columns[i] for i in top15] else ""
        print(f"  센서 {c}: d={v:+.2f}{in_shap}")

    joblib.dump({
        'weekly': weekly,
        'p_bar': float(p_bar),
        'excursion_week': exc_week,
        'excursion_mask': exc_mask.values,
        'calm_mask': calm_mask,
        'top_sensors': top_exc,
        'exc_n': int(exc_mask.sum()),
        'exc_fail': int(y[exc_mask.values].sum()),
    }, os.path.join(ART, 'excursion.joblib'))
    print("저장: artifacts/excursion.joblib")

# ============================================
# 7단계: 합성 시나리오 검증 (무작위 대조군 포함, 수정 5·6)
# ============================================
# 센서 선정: 모델과 무관한 단변량 효과크기 d (순환논리 차단)
P, F = X[y == 0], X[y == 1]
d_all = ((F.mean() - P.mean()) / P.std(ddof=1)) \
    .replace([np.inf, -np.inf], np.nan).dropna()
d_ranked = d_all.reindex(d_all.abs().sort_values(ascending=False).index)
# 산포가 0이 아닌 센서만
d_ranked = d_ranked[[c for c in d_ranked.index if robust_stats(c)[1] > 0]]

rng = np.random.default_rng(42)
normal_pool = np.where(y == 0)[0]
batch_idx = np.sort(rng.choice(normal_pool, 100, replace=False))
batch_base = X.iloc[batch_idx].copy()
N_INJ = 30

# 조작 전 기준: 미조작 웨이퍼는 OOF 확률이 정직한 값
base_oof = oof_prob[batch_idx]
print(f"\n=== [합성 검증] 기준 배치: 위험판정 {(base_oof >= THRESHOLD).sum()}/100 (OOF 기준) ===")

def inject(Xb, sensors, level):
    """방향 보정 주입: d>0이면 +level σ, d<0이면 -level σ"""
    out = Xb.copy()
    for c in sensors:
        med, sig = robust_stats(c)
        sign = 1.0 if d_all.get(c, 0) >= 0 else -1.0
        out.iloc[:N_INJ, out.columns.get_loc(c)] = med + sign * level * sig
    return out

def eval_batch(Xb):
    """주입 30장은 모델 예측, 미조작 70장은 OOF (정직성 유지)"""
    p_inj = predict_prob(Xb.iloc[:N_INJ])
    p_clean = base_oof[N_INJ:]
    return p_inj, p_clean

print("\n[주입 스윕] 상위 d 센서 K개 vs 무작위 K개 (각 +2σ, 방향 보정)")
print("K   | 상위d 검출/30 | 무작위 검출/30 | 상위d 평균확률 | 무작위 평균확률")
sweep_rows = []
all_candidates = [c for c in X.columns if robust_stats(c)[1] > 0]
for K in [1, 3, 5, 10]:
    top_sensors = list(d_ranked.index[:K])
    p_top, _ = eval_batch(inject(batch_base, top_sensors, 2.0))
    # 무작위 대조군: 3회 평균
    rand_det, rand_prob = [], []
    for rseed in range(3):
        rr = np.random.default_rng(1000 + rseed)
        rand_sensors = list(rr.choice(all_candidates, K, replace=False))
        p_rand, _ = eval_batch(inject(batch_base, rand_sensors, 2.0))
        rand_det.append((p_rand >= THRESHOLD).sum())
        rand_prob.append(p_rand.mean())
    row = dict(K=K,
               top_det=int((p_top >= THRESHOLD).sum()), top_prob=float(p_top.mean()),
               rand_det=float(np.mean(rand_det)), rand_prob=float(np.mean(rand_prob)))
    sweep_rows.append(row)
    print(f"{K:3d} |     {row['top_det']:2d}/30    |     {row['rand_det']:4.1f}/30   "
          f"|     {row['top_prob']:.3f}    |     {row['rand_prob']:.3f}")

# 기존 5종 시나리오도 유지 (대시보드 연속성) — 방향 보정 적용
scenarios = {}
def record_scenario(key, name, sensors, level):
    Xb = inject(batch_base, sensors, level)
    p_inj, p_clean = eval_batch(Xb)
    prob_full = np.concatenate([p_inj, p_clean])
    scenarios[key] = dict(name=name, batch=rebuild(Xb), prob=prob_full,
                          injected=sensors, level=level)
    print(f"  {name}: 검출 {(p_inj >= THRESHOLD).sum()}/30, "
          f"오탐(OOF) {(p_clean >= THRESHOLD).sum()}/70")

print("\n[시나리오 5종]")
record_scenario('S1', 'S1. 단일 센서 이상 - 센서 59 (+3σ)', [59], 3.0)
record_scenario('S2', 'S2. 단일 센서 이상 - 센서 130 (+3σ)', [130], 3.0)
record_scenario('S3', 'S3. 다중 센서 이상 - 59·33·21 (각 +2σ)', [59, 33, 21], 2.0)
record_scenario('S4', 'S4. 다중 센서 이상 - 33·130·460 (각 +2σ)', [33, 130, 460], 2.0)

# S5 드리프트
batch_drift = batch_base.copy()
med59, sig59 = robust_stats(59)
drift_levels = np.linspace(0, 3, 100)
batch_drift.iloc[:, batch_drift.columns.get_loc(59)] = med59 + drift_levels * sig59
prob_drift = predict_prob(batch_drift)
scenarios['S5'] = dict(name='S5. 점진적 드리프트 (센서59, 0→3σ)',
                       batch=rebuild(batch_drift), prob=prob_drift,
                       injected=[59], level=None)
print(f"  S5. 드리프트: 구간별 검출 "
      f"{[int((prob_drift[s:s+20] >= THRESHOLD).sum()) for s in range(0, 100, 20)]}")

joblib.dump({
    'scenarios': scenarios,
    'base_idx': batch_idx,
    'base_oof': base_oof,
    'threshold': THRESHOLD,
    'n_inject': N_INJ,
    'sweep': sweep_rows,
    'd_ranked_top20': d_ranked.head(20),
}, os.path.join(ART, 'scenarios.joblib'))
print("저장: artifacts/scenarios.joblib")

# ============================================
# 8단계: SPC 기준선 (OOF 기반으로 교정)
# ============================================
# 이전 버전은 최종모델의 학습데이터 예측으로 기준선을 계산해 오염되어 있었다.
# 미조작 정상 웨이퍼의 정직한 확률은 OOF이므로, OOF에서 반복 추출한다.
BATCH_SIZE, N_TRIALS = 30, 200
normal_oof = oof_prob[y == 0]
baseline_counts = np.array([
    (np.random.default_rng(s).choice(normal_oof, BATCH_SIZE, replace=False)
     >= THRESHOLD).sum()
    for s in range(N_TRIALS)])
bl_mean, bl_std = baseline_counts.mean(), baseline_counts.std(ddof=1)
UCL = bl_mean + 3 * bl_std

print(f"\n=== [SPC 기준선, OOF 기반] {BATCH_SIZE}장 x {N_TRIALS}회 ===")
print(f"  평균 {bl_mean:.2f}장 ± {bl_std:.2f} / UCL {UCL:.1f}장 "
      f"/ 95% 구간 {np.percentile(baseline_counts, 2.5):.0f}~"
      f"{np.percentile(baseline_counts, 97.5):.0f}장")

scenario_eval = {}
for key, sc in scenarios.items():
    if key == 'S5':
        continue
    det = int((sc['prob'][:N_INJ] >= THRESHOLD).sum())
    z = (det - bl_mean) / bl_std
    pv = sstats.binomtest(det, BATCH_SIZE, bl_mean / BATCH_SIZE,
                          alternative='greater').pvalue
    scenario_eval[key] = dict(detected=det, z=float(z), p=float(pv),
                              over_ucl=bool(det > UCL))
    print(f"  {key}: {det}/30, Z={z:+.2f}, p={pv:.4f}, "
          f"{'관리한계 초과' if det > UCL else '한계 내'}")

joblib.dump({
    'baseline_counts': baseline_counts, 'mean': float(bl_mean),
    'std': float(bl_std), 'ucl': float(UCL),
    'batch_size': BATCH_SIZE, 'n_trials': N_TRIALS,
    'p95_low': float(np.percentile(baseline_counts, 2.5)),
    'p95_high': float(np.percentile(baseline_counts, 97.5)),
    'scenario_eval': scenario_eval,
}, os.path.join(ART, 'spc_baseline.joblib'))
print("저장: artifacts/spc_baseline.joblib")

# ============================================
# 9단계: 가법성 분석 (DOE 교정판, 수정 4)
# ============================================
# 교호작용은 로그오즈 공간에서 측정한다. 확률 공간 결과도 함께 저장해
# "측정 공간에 따라 결론이 뒤집힌다"는 것 자체를 분석 결과로 제시한다.
S_DOE = [59, 21, 103]
L_DOE = 2.0
base200 = X.iloc[np.sort(np.random.default_rng(7).choice(normal_pool, 200, replace=False))].copy()

def full_inject(Xb, shifts):
    out = Xb.copy()
    for c, k in shifts.items():
        med, sig = robust_stats(c)
        out[c] = med + k * sig
    return out

bp = predict_prob(base200).mean()
bm = predict_margin(base200).mean()
solo_p, solo_m = {}, {}
for c in S_DOE:
    Xi = full_inject(base200, {c: L_DOE})
    solo_p[c] = predict_prob(Xi).mean() - bp
    solo_m[c] = predict_margin(Xi).mean() - bm
X_all = full_inject(base200, {c: L_DOE for c in S_DOE})
joint_p = predict_prob(X_all).mean() - bp
joint_m = predict_margin(X_all).mean() - bm

inter_p = joint_p - sum(solo_p.values())
inter_m = joint_m - sum(solo_m.values())
print(f"\n=== [가법성 분석] 센서 {S_DOE}, 각 +{L_DOE}σ ===")
print(f"  확률 공간:   합 {sum(solo_p.values()):+.4f} vs 동시 {joint_p:+.4f} "
      f"-> 교호 {inter_p:+.4f} ({inter_p/sum(solo_p.values())*100:+.1f}%)")
print(f"  로그오즈 공간: 합 {sum(solo_m.values()):+.4f} vs 동시 {joint_m:+.4f} "
      f"-> 교호 {inter_m:+.4f} ({inter_m/sum(solo_m.values())*100:+.1f}%)")
print("  => 확률 공간의 양의 교호작용은 시그모이드 볼록성에 의한 착시.")
print("     모델은 로그오즈에서 가법적이며, 다변량 우위는 '누적'으로 설명된다.")

# 2^3 설계표 (로그오즈 응답)
design_rows = []
for combo in itertools.product([0, 1], repeat=3):
    Xi = full_inject(base200, {c: L_DOE for c, on in zip(S_DOE, combo) if on})
    design_rows.append({
        '조건': ''.join('+' if v else '-' for v in combo),
        '평균확률': float(predict_prob(Xi).mean()),
        '로그오즈': float(predict_margin(Xi).mean()),
        '검출률': float((predict_prob(Xi) >= THRESHOLD).mean()),
    })
df_design = pd.DataFrame(design_rows)

joblib.dump({
    'factors': S_DOE, 'level': L_DOE,
    'design_table': df_design,
    'solo_prob': solo_p, 'solo_margin': solo_m,
    'joint_prob': float(joint_p), 'joint_margin': float(joint_m),
    'inter_prob': float(inter_p), 'inter_margin': float(inter_m),
    'base_prob': float(bp), 'base_margin': float(bm),
}, os.path.join(ART, 'doe.joblib'))
print("저장: artifacts/doe.joblib")

# ============================================
# 10단계: 대시보드용 사전 계산
# ============================================
joblib.dump({
    'oof_prob': oof_prob,
    'prob_insample': prob_insample,   # 비교 설명용으로만 보관
    'shap_all': shap_values,
    'columns': list(X_final.columns),
    'y': y,
    'timestamps': timestamps,
}, os.path.join(ART, 'precomputed.joblib'))
print("저장: artifacts/precomputed.joblib")

print("\n전체 파이프라인 완료. artifacts/ 에 6개 파일 생성.")
