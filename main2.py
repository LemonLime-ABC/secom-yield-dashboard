# ============================================
# [교정판 파이프라인] main2.py
# ============================================
# 기존 main.py에서 발견된 문제를 고친 버전:
#  문제1. 한 번만 나눈 검증(불량 21개)은 운에 좌우됨 → 5-Fold 교차검증으로 교체
#  문제2. 중앙값 대치·열 버리기를 전체 데이터로 계산(누수) → fold 안에서만 계산
#  문제3. 임계값을 시험지(검증셋) 보고 결정(누수) → 교차검증 예측으로 결정
#  문제4. 상수 열 116개(정보 0) 방치 → 제거
#  문제5. 중앙값 대치가 '측정 안 됨'이라는 정보를 지움
#        → XGBoost는 빈 칸(NaN)을 스스로 처리 가능하므로 NaN 유지
#        → 대신 '빈 칸이었는지 여부'를 새 정보(결측 지시자)로 추가
# ※ 표준화는 모델 입력에서 제외한다.
#   (XGBoost는 눈금에 둔감 + NaN을 유지하려면 표준화가 걸림돌)
#   Z-score는 나중에 대시보드 '표시용'으로만 따로 계산한다.

import pandas as pd
import numpy as np

# ============================================
# 1단계: 데이터 로드 (main.py와 동일)
# ============================================
data = pd.read_csv('secom.data', sep=r'\s+', header=None)
labels = pd.read_csv('secom_labels.data', sep=r'\s+', header=None)

# 정답: 불량=1, 정상=0
y = (labels[0] == 1).astype(int).values
X = data.copy()
print("원본 크기:", X.shape, "/ 불량 수:", y.sum())

# ============================================
# 2단계: 상수 열 제거 (문제4 해결)
# ============================================
# nunique: 각 열에 서로 다른 값이 몇 종류인지. 1이면 값이 하나뿐 = 정보 없음.
n_unique = X.nunique(dropna=True)
const_cols = n_unique[n_unique <= 1].index
X = X.drop(columns=const_cols)
print("제거한 상수 열:", len(const_cols), "/ 남은 센서:", X.shape[1])

# ============================================
# 3단계: 5-Fold 교차검증으로 학습·평가 (문제1,2,5 해결)
# ============================================
# 교차검증: 데이터를 5조각으로 나눠, 5번 번갈아 시험 본다.
# → 불량 104개 전부가 한 번씩 '시험 문제'가 되어 평가가 안정된다.
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, confusion_matrix
from xgboost import XGBClassifier

# StratifiedKFold: 나눌 때 fold마다 불량 비율(6.6%)을 유지해주는 나누기 도구
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# oof(out-of-fold): 각 웨이퍼가 '시험 문제였을 때' 받은 불량 확률을 모아두는 곳.
# 5번의 시험을 합치면 1567개 전부의 정직한 예측이 모인다.
oof_prob = np.zeros(len(y))
fold_scores = []

# enumerate(...): 반복하면서 몇 번째인지(fold_no)도 같이 세어주는 문법
for fold_no, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    # 이번 fold의 학습용/시험용 분리 (.iloc: 행 번호로 잘라내기)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # --- 결측 지시자 추가 (문제5 해결) ---
    # '학습용에서' 빈 칸이 있는 열을 찾고, 그 열들에 대해
    # "빈 칸이었으면 1, 아니면 0"인 새 열들을 만들어 붙인다.
    # 기준을 학습용에서만 잡으므로 누수 없음 (문제2 해결과 같은 원리)
    miss_cols = X_tr.columns[X_tr.isna().mean() > 0]
    X_tr = pd.concat([X_tr, X_tr[miss_cols].isna().astype(int).add_prefix('m_')], axis=1)
    X_te = pd.concat([X_te, X_te[miss_cols].isna().astype(int).add_prefix('m_')], axis=1)

    # --- 불균형 보정값 (이번 fold의 학습용 기준) ---
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    # --- 모델: 고차원·약한 신호에 맞춘 설정 ---
    # max_depth=2       : 아주 얕은 나무 → 잡음에 덜 속음
    # n_estimators=600  : 대신 나무를 많이
    # colsample_bytree=0.3 : 나무마다 센서의 30%만 무작위로 봄 → 잡음 분산
    # min_child_weight, reg_lambda : 과하게 외우는 것(과적합)을 누르는 규제
    model = XGBClassifier(
        n_estimators=600, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.3,
        min_child_weight=5, reg_lambda=5.0,
        scale_pos_weight=spw, random_state=42,
        eval_metric='logloss', n_jobs=4,
    )
    model.fit(X_tr.values, y_tr)

    # 이번 fold의 시험 문제들에 대한 불량 확률을 기록
    prob = model.predict_proba(X_te.values)[:, 1]
    oof_prob[te_idx] = prob

    score = average_precision_score(y_te, prob)
    fold_scores.append(score)
    print(f"Fold {fold_no} PR-AUC: {score:.3f}")

# ============================================
# 4단계: 종합 성능 (평균 ± 편차)
# ============================================
print(f"\n5-Fold PR-AUC: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
print("(단일 분할이 아니라 5번 시험의 평균이라 훨씬 믿을 수 있는 숫자)")

# ============================================
# 5단계: 임계값 결정 (문제3 해결)
# ============================================
# oof_prob에는 1567장 전부의 '정직한'(시험일 때 받은) 확률이 있다.
# 불량 104개 전부를 기준으로 임계값별 성능을 본다.
print("\n=== 임계값별 성능 (불량 104개 전체 기준) ===")
print("임계값 | 잡은불량/104 | 오탐 | Recall | Precision")
for t in [0.50, 0.40, 0.30, 0.20]:
    pred = (oof_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rec = tp / (tp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f" {t:.2f}  |     {tp:3d}      | {fp:3d} | {rec:.3f}  | {prec:.3f}")

# --------------------------------------------
# [ 이 결과를 읽는 프레임 - 참고용 ]
# --------------------------------------------
# ● 이 모델은 '합불 판정기'가 아니라 '위험 웨이퍼 선별기(triage)'다.
#   - 예: 임계값 0.20 → 불량의 약 55%를 잡는 대신,
#     정상의 약 19%를 '정밀검사 대상'으로 올린다.
#   - "전수검사 대신, 위험한 22%만 골라 검사한다"는 현업 FDC 스크리닝 논리.
# ● SECOM은 신호가 약한 데이터라 PR-AUC 0.2 안팎이 현실적 천장임을
#   여러 방법(대치/선별/규제/지시자)으로 실험해 확인했다.
#   → 발표에서는 '높은 점수'가 아니라 '검증의 신뢰성과 운영 관점'을 근거로 말한다.

# ============================================
# 6단계: 최종 모델 확정·저장 + SHAP 해석
# ============================================
# [이 단계 결과물]
# - 검증은 끝났으니, 이제 전체 데이터로 '배포용 최종 모델' 하나를 학습한다
# - 모델과 부속 정보를 파일로 저장한다 (대시보드가 불러 쓸 재료)
# - SHAP으로 "모델이 어떤 센서를 근거로 판단하는지"를 뽑는다
#   ※ shap 패키지는 파이썬 3.14와 호환 문제가 있어,
#     XGBoost에 내장된 SHAP(TreeSHAP) 계산 기능을 사용한다. 같은 방법론이다.

import os
import joblib   # 파이썬 객체(모델 등)를 파일로 저장/불러오는 도구
import xgboost as xgb

# --------------------------------------------
# 6-1. 전체 데이터로 최종 모델 학습
# --------------------------------------------
# 교차검증 때와 같은 방식으로 결측 지시자를 만든다. (이번엔 전체 기준 — 배포용이라 OK)
miss_cols_final = X.columns[X.isna().mean() > 0]
X_final = pd.concat([X, X[miss_cols_final].isna().astype(int).add_prefix('m_')], axis=1)

spw_final = (y == 0).sum() / (y == 1).sum()

final_model = XGBClassifier(
    n_estimators=600, max_depth=2, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.3,
    min_child_weight=5, reg_lambda=5.0,
    scale_pos_weight=spw_final, random_state=42,
    eval_metric='logloss', n_jobs=4,
)
final_model.fit(X_final.values, y)
print("최종 모델 학습 완료 / 입력 열 수:", X_final.shape[1])

# --------------------------------------------
# 6-2. 모델 + 부속 정보 저장 (처음 나오는 '진짜 저장')
# --------------------------------------------
# artifacts 폴더를 만들고(이미 있으면 그냥 넘어감) 거기에 저장한다.
os.makedirs('artifacts', exist_ok=True)

# 저장 목록:
#  - 모델 자체
#  - 입력 열 이름 목록 (대시보드가 같은 순서로 데이터를 만들기 위해)
#  - 운영 임계값 (0.20으로 확정한 값)
#  - 정상 웨이퍼들의 센서별 평균/표준편차 (나중에 Z-score 표시·시나리오 생성용)
normal_mean = X[y == 0].mean()
normal_std = X[y == 0].std()

joblib.dump({
    'model': final_model,
    'columns': list(X_final.columns),
    'sensor_columns': list(X.columns),      # 원래 센서만 (지시자 제외)
    'miss_cols': list(miss_cols_final),
    'threshold': 0.20,
    'normal_mean': normal_mean,
    'normal_std': normal_std,
}, 'artifacts/secom_model.joblib')
print("저장 완료: artifacts/secom_model.joblib")

# --------------------------------------------
# 6-3. SHAP 값 계산 (XGBoost 내장 TreeSHAP)
# --------------------------------------------
# pred_contribs=True : 각 웨이퍼의 예측에 대해 "센서별 기여도(SHAP 값)"를 계산.
# 결과: (웨이퍼 수) x (열 수 + 1) 표. 마지막 열은 기준값이라 잘라낸다.
dm = xgb.DMatrix(X_final.values)
shap_values = final_model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
print("SHAP 계산 완료. 크기:", shap_values.shape)

# --------------------------------------------
# 6-4. 전역 해석: 모델이 전반적으로 어떤 센서를 보는가
# --------------------------------------------
# 각 센서의 |SHAP| 평균 = "이 센서가 판단에 평균적으로 얼마나 영향을 줬나"
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top15_idx = np.argsort(mean_abs_shap)[::-1][:15]

print("\n=== [전역] 불량 판단 기여 상위 15개 센서 ===")
for rank, i in enumerate(top15_idx, start=1):
    col = X_final.columns[i]
    print(f" {rank:2d}. 센서 {col}  (평균 기여도 {mean_abs_shap[i]:.4f})")

# --------------------------------------------
# 6-5. 개별 해석: 불량 웨이퍼 한 장은 '왜' 불량으로 예측됐나
# --------------------------------------------
# 실제 불량 중 모델이 가장 높은 확률을 준 웨이퍼를 예시로 뽑는다.
prob_all = final_model.predict_proba(X_final.values)[:, 1]
bad_idx = np.where(y == 1)[0]                     # 실제 불량들의 행 번호
example = bad_idx[np.argmax(prob_all[bad_idx])]   # 그중 확률 최고인 한 장

print(f"\n=== [개별] 웨이퍼 #{example} (실제 불량, 예측 확률 {prob_all[example]:.2f}) ===")
contrib = shap_values[example]
top5 = np.argsort(np.abs(contrib))[::-1][:5]
for i in top5:
    direction = "불량 쪽으로 밀음" if contrib[i] > 0 else "정상 쪽으로 당김"
    print(f"  센서 {X_final.columns[i]}: {contrib[i]:+.3f}  ({direction})")

# ============================================
# 7단계: 시나리오 배치 검증
# ============================================
# [이 단계 결과물]
# - 정상 웨이퍼 100장을 가져와, 일부에 '일부러 이상'을 심는다
# - 정답을 아는 문제를 만들어 모델을 채점한다
#   → 모델이 심은 이상을 잡아내는가? SHAP이 그 센서를 원인으로 지목하는가?
# - 시나리오 A: 단일 센서 이상 (센서 59)
# - 시나리오 B: 다중 센서 동시 이상 (단독으론 약한 수준을 여러 개)

# --------------------------------------------
# 7-1. 로버스트 통계 (평균/표준편차 대신)
# --------------------------------------------
# 이 데이터의 센서들은 분포가 심하게 치우쳐 있다(예: 센서59 중앙값 0.95, 최대 168).
# 이럴 때 평균·표준편차는 극단값에 끌려가 왜곡된다.
# → 중앙값과 IQR(사분위 범위)로 '치우침에 강한' 기준을 쓴다.
#   robust_sigma = IQR / 1.349  (정규분포에서 IQR과 표준편차의 환산 상수)
def robust_stats(col):
    s = X[col][y == 0]                      # 정상 웨이퍼만의 값
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return med, iqr / 1.349

THRESHOLD = 0.20   # 4단계에서 확정한 운영 임계값

# --------------------------------------------
# 7-2. 정상 웨이퍼 100장 배치 준비
# --------------------------------------------
rng = np.random.default_rng(42)             # 난수 고정 (매번 같은 100장)
normal_idx = np.where(y == 0)[0]            # 실제 정상 웨이퍼들의 행 번호
sample_idx = rng.choice(normal_idx, 100, replace=False)

batch_base = X_final.iloc[sample_idx].copy()          # 조작 전 원본 배치
prob_base = final_model.predict_proba(batch_base.values)[:, 1]

N_INJECT = 30                                          # 앞 30장에 이상 주입
base_hit_inject = (prob_base[:N_INJECT] >= THRESHOLD).sum()
base_hit_clean = (prob_base[N_INJECT:] >= THRESHOLD).sum()
print("=== 기준선 (아직 아무것도 안 심음) ===")
print(f"주입 예정 30장: {base_hit_inject}장이 이미 임계 초과")
print(f"정상 유지 70장: {base_hit_clean}장이 임계 초과 (원래 오탐)")

# --------------------------------------------
# 7-3. 시나리오 A: 센서 59 단독 이상 주입
# --------------------------------------------
# 이상 크기를 여러 단계로 바꿔가며 '주입량 vs 검출률' 관계를 본다.
med59, sig59 = robust_stats(59)
col59 = batch_base.columns.get_loc(59)      # 센서59가 표의 몇 번째 열인지

print(f"\n=== [시나리오 A] 센서 59 단독 주입 (30/100장) ===")
print(f"센서59 정상 중앙값 {med59:.2f}, 로버스트σ {sig59:.2f}")
print("주입량 | 주입값 | 검출/30 | 오탐/70 | SHAP지목/30")

for k in [1, 2, 3, 4]:
    batch = batch_base.copy()
    inject_value = med59 + k * sig59
    # .iloc[행범위, 열번호] 로 앞 30장의 센서59만 이상값으로 교체
    batch.iloc[:N_INJECT, col59] = inject_value

    prob = final_model.predict_proba(batch.values)[:, 1]
    detected = (prob[:N_INJECT] >= THRESHOLD).sum()
    false_alarm = (prob[N_INJECT:] >= THRESHOLD).sum()

    # SHAP이 '센서59'를 1순위 원인으로 지목했는지 확인
    dmat = xgb.DMatrix(batch.values)
    sv = final_model.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
    shap_hit = sum(1 for i in range(N_INJECT) if np.argmax(np.abs(sv[i])) == col59)

    print(f"  +{k}σ  | {inject_value:6.2f} |   {detected:2d}/30  |  {false_alarm:2d}/70  |    {shap_hit:2d}/30")

# --------------------------------------------
# 7-4. 시나리오 B: 다중 센서 동시 이상
# --------------------------------------------
# 각각은 '단독으로는 약한' +1.5σ 수준이지만, 동시에 여러 개가 틀어진 경우.
# → 단일 센서 감시로는 놓치고, 다변량 분석이라야 잡히는 상황을 재현한다.
multi_sensors = [59, 33, 21]
LEVEL = 1.5

print(f"\n=== [시나리오 B] 다중 센서 동시 주입 (각 +{LEVEL}σ) ===")

# (1) 각 센서를 '단독으로만' 주입했을 때
print("단독 주입 시:")
solo_results = {}
for c in multi_sensors:
    batch = batch_base.copy()
    med, sig = robust_stats(c)
    batch.iloc[:N_INJECT, batch.columns.get_loc(c)] = med + LEVEL * sig
    prob = final_model.predict_proba(batch.values)[:, 1]
    solo_results[c] = (prob[:N_INJECT] >= THRESHOLD).sum()
    print(f"  센서 {c} 단독: 검출 {solo_results[c]}/30")

# (2) 세 센서를 동시에 주입했을 때
batch_multi = batch_base.copy()
for c in multi_sensors:
    med, sig = robust_stats(c)
    batch_multi.iloc[:N_INJECT, batch_multi.columns.get_loc(c)] = med + LEVEL * sig
prob_multi = final_model.predict_proba(batch_multi.values)[:, 1]
multi_detected = (prob_multi[:N_INJECT] >= THRESHOLD).sum()
multi_false = (prob_multi[N_INJECT:] >= THRESHOLD).sum()

print(f"동시 주입 시: 검출 {multi_detected}/30, 오탐 {multi_false}/70")
print(f"  → 단독 최대({max(solo_results.values())}/30)보다 동시({multi_detected}/30)가 더 많이 검출되면,")
print(f"     '개별 센서는 정상 범위여도 조합이 이상하면 잡아낸다'는 다변량 감지 능력의 근거가 된다.")

# --------------------------------------------
# [ 이 결과를 읽는 법 - 참고용 정리 ]
# --------------------------------------------
# ● 왜 로버스트 통계인가:
#   센서 분포가 치우쳐 있어(왜도 큼) 평균+3σ는 실제 데이터에 없는 값이 된다.
#   중앙값·IQR 기준이 실제 공정 데이터의 산포를 더 정확히 반영한다.
#
# ● '포화(saturation)' 현상:
#   주입량을 +2σ 이상 키워도 검출률이 더 오르지 않는 구간이 나타난다.
#   원인: 트리 모델은 학습 데이터에서 본 '분기점'까지만 구분한다.
#         그 범위를 넘어선 값은 전부 같은 쪽으로 분류되어 반응이 멈춘다.
#   해석: 결함이 아니라 트리 계열 모델의 구조적 특성.
#         → "이 모델의 검출 민감 구간은 약 +1σ~+2σ"라고 정량화할 수 있고,
#           이는 계측에서 말하는 '검출 한계(detection limit)'와 같은 개념이다.
#
# ● SHAP 지목률의 의미:
#   모델이 '불량'이라 판정했을 때, 그 근거로 실제 주입한 센서를 짚었는지.
#   이 비율이 높을수록 "원인 규명(FA)이 신뢰할 만하다"는 근거가 된다.
#
# ● 시나리오 B의 의미:
#   단독으로는 약한 이상이라도 여러 개가 겹치면 검출된다면,
#   단일 센서 관리도(SPC)로는 놓치는 불량을 다변량 모델이 잡는다는 뜻.
#   → 기존 통계적 공정 관리 대비 ML 도입의 실질적 근거.

# ============================================
# 8단계: 시나리오 배치 라이브러리 (5종)
# ============================================
# [이 단계 결과물]
# - 서로 다른 5가지 이상 상황을 100장 배치로 각각 만든다
# - 각 배치에 대해: 검출률 / 오탐 / 원인 지목 정확도를 측정
# - 대시보드에서 "배치를 골라 진단"하는 데 쓸 재료를 파일로 저장
#
# [원인 지목 측정 방식 변경 - 중요]
#   기존: "SHAP 순위 1위가 주입 센서인가"
#   문제: 센서59가 워낙 지배적이라, 다른 센서를 조작해도 1위는 계속 센서59.
#   변경: "주입 전 대비 SHAP 기여도가 가장 많이 늘어난 센서가 주입 센서인가"
#   근거: 실무 FDC의 '변화점 분석'과 같은 논리. 절대 순위가 아니라 변화량을 본다.

# --------------------------------------------
# 8-1. 공통 준비
# --------------------------------------------
THRESHOLD = 0.20
N_INJECT = 30          # 100장 중 앞 30장에 이상 주입

rng = np.random.default_rng(42)
sample_idx = rng.choice(np.where(y == 0)[0], 100, replace=False)
batch_base = X_final.iloc[sample_idx].copy()

# 주입 전 기준 상태 (확률 + SHAP)
prob_base = final_model.predict_proba(batch_base.values)[:, 1]
shap_base = final_model.get_booster().predict(
    xgb.DMatrix(batch_base.values), pred_contribs=True)[:, :-1]

def robust_stats(col):
    """정상 웨이퍼 기준 중앙값과 로버스트 표준편차(IQR/1.349)"""
    s = X[col][y == 0]
    return s.median(), (s.quantile(0.75) - s.quantile(0.25)) / 1.349

def make_batch(sensor_levels):
    """sensor_levels: {센서번호: 몇 σ} → 앞 30장에 주입한 배치를 만든다"""
    b = batch_base.copy()
    for c, k in sensor_levels.items():
        med, sig = robust_stats(c)
        b.iloc[:N_INJECT, b.columns.get_loc(c)] = med + k * sig
    return b

def evaluate(batch, injected_sensors, name):
    """배치를 평가: 검출률/오탐/원인지목"""
    prob = final_model.predict_proba(batch.values)[:, 1]
    shap_v = final_model.get_booster().predict(
        xgb.DMatrix(batch.values), pred_contribs=True)[:, :-1]

    detected = (prob[:N_INJECT] >= THRESHOLD).sum()
    false_alarm = (prob[N_INJECT:] >= THRESHOLD).sum()

    # 원인 지목: SHAP 증가량(주입 후 - 주입 전)이 가장 큰 센서 확인
    delta = shap_v[:N_INJECT] - shap_base[:N_INJECT]
    inj_cols = [batch.columns.get_loc(c) for c in injected_sensors]

    if len(injected_sensors) == 1:
        # 단일: 증가량 1위가 주입 센서인 비율
        hit = sum(1 for i in range(N_INJECT) if np.argmax(delta[i]) == inj_cols[0])
        attr = f"{hit}/{N_INJECT} (증가량 1위 일치)"
    else:
        # 다중: 증가량 상위 3개 중 주입 센서가 평균 몇 개 포함됐나
        k = len(injected_sensors)
        avg = np.mean([len(set(np.argsort(delta[i])[::-1][:k]) & set(inj_cols))
                       for i in range(N_INJECT)])
        attr = f"평균 {avg:.1f}/{k}개 (증가량 상위 포함)"

    print(f"\n[{name}]")
    print(f"  조작 센서: {injected_sensors}")
    print(f"  검출: {detected}/{N_INJECT}  (주입 전 {(prob_base[:N_INJECT]>=THRESHOLD).sum()}/{N_INJECT})")
    print(f"  오탐: {false_alarm}/70  (주입 전 {(prob_base[N_INJECT:]>=THRESHOLD).sum()}/70)")
    print(f"  원인 지목: {attr}")
    print(f"  배치 평균 불량확률: {prob[:N_INJECT].mean():.3f} (정상부 {prob[N_INJECT:].mean():.3f})")
    return {'name': name, 'batch': batch, 'prob': prob,
            'injected': injected_sensors, 'detected': int(detected),
            'false_alarm': int(false_alarm)}

print("=" * 55)
print("시나리오 배치 라이브러리 (각 100장, 앞 30장 조작)")
print("=" * 55)
print(f"기준선: 주입 전 검출 {(prob_base[:N_INJECT]>=THRESHOLD).sum()}/30, "
      f"오탐 {(prob_base[N_INJECT:]>=THRESHOLD).sum()}/70")

scenarios = {}

# --------------------------------------------
# 시나리오 1 & 2: 단일 센서 이상 (같은 유형, 다른 센서)
# --------------------------------------------
# 단일 설비 파라미터 하나가 틀어진 상황. 센서만 다르게 두 가지.
scenarios['S1'] = evaluate(make_batch({59: 3}), [59],
                           "S1. 단일 센서 이상 - 센서 59 (+3σ)")
scenarios['S2'] = evaluate(make_batch({130: 3}), [130],
                           "S2. 단일 센서 이상 - 센서 130 (+3σ)  [S1과 같은 유형, 다른 센서]")

# --------------------------------------------
# 시나리오 3 & 4: 다중 센서 동시 이상 (같은 유형, 다른 조합)
# --------------------------------------------
# 각각은 약한 이탈(+2σ)이지만 여러 개가 동시에 틀어진 상황.
# 단일 센서 관리도로는 놓치고, 다변량 모델이라야 잡히는 케이스.
scenarios['S3'] = evaluate(make_batch({59: 2, 33: 2, 21: 2}), [59, 33, 21],
                           "S3. 다중 센서 동시 이상 - 59+33+21 (각 +2σ)")
scenarios['S4'] = evaluate(make_batch({33: 2, 130: 2, 460: 2}), [33, 130, 460],
                           "S4. 다중 센서 동시 이상 - 33+130+460 (각 +2σ)  [S3과 같은 유형, 다른 조합]")

# --------------------------------------------
# 시나리오 5: 점진적 드리프트
# --------------------------------------------
# 설비가 서서히 나빠지는 상황. 100장 전체에 걸쳐 0σ → 3σ로 점점 심해진다.
# 앞부분은 정상처럼 보이다가 뒤로 갈수록 불량률이 오르는 패턴 → 관리도로 보기 좋다.
batch_drift = batch_base.copy()
med59, sig59 = robust_stats(59)
drift_levels = np.linspace(0, 3, 100)          # 0σ부터 3σ까지 균등 증가
batch_drift.iloc[:, batch_drift.columns.get_loc(59)] = med59 + drift_levels * sig59
prob_drift = final_model.predict_proba(batch_drift.values)[:, 1]

print("\n[S5. 점진적 드리프트 - 센서 59가 0σ→3σ로 서서히 악화]")
print("  구간(웨이퍼) | 주입량 | 검출/20 | 평균확률")
for s in range(0, 100, 20):
    e = s + 20
    print(f"   {s:2d}-{e-1:2d}      | {drift_levels[s]:.1f}~{drift_levels[e-1]:.1f}σ |  {(prob_drift[s:e]>=THRESHOLD).sum():2d}/20  | {prob_drift[s:e].mean():.3f}")
scenarios['S5'] = {'name': 'S5. 점진적 드리프트', 'batch': batch_drift,
                   'prob': prob_drift, 'injected': [59],
                   'detected': int((prob_drift >= THRESHOLD).sum()), 'false_alarm': None}

# --------------------------------------------
# 8-2. 시나리오 배치를 파일로 저장 (대시보드용)
# --------------------------------------------
joblib.dump({
    'scenarios': {k: {'batch': v['batch'], 'prob': v['prob'],
                      'injected': v['injected'], 'name': v['name']}
                  for k, v in scenarios.items()},
    'base_batch': batch_base,
    'base_prob': prob_base,
    'threshold': THRESHOLD,
    'n_inject': N_INJECT,
}, 'artifacts/scenarios.joblib')
print("\n저장 완료: artifacts/scenarios.joblib (대시보드가 불러 쓸 시나리오 5종)")

# --------------------------------------------
# [ 시나리오 구성 의도 - 참고용 정리 ]
# --------------------------------------------
# ● S1 vs S2 (같은 유형, 다른 센서):
#   같은 '단일 센서 이상'이라도 어느 센서냐에 따라 검출률이 다르다.
#   → 모델의 센서별 민감도 차이를 정량화. 어느 공정을 더 촘촘히 감시해야 하는지 근거.
#
# ● S3 vs S4 (같은 유형, 다른 조합):
#   센서 조합이 달라지면 다변량 검출 성능도 달라진다.
#   → 특정 조합이 더 위험하다는 것을 보여주면, 상관 감시 대상 선정의 근거가 된다.
#
# ● S5 (드리프트):
#   급격한 이상이 아니라 서서히 나빠지는 경우. 초기에는 개별 웨이퍼가 정상 판정되지만
#   배치 단위로 추세를 보면 악화가 드러난다.
#   → 관리도(SPC)의 존재 이유이자, '선제적 수율 방어'의 근거.
#
# ● 공통: 오탐(뒤 70장)은 어느 시나리오에서도 변하지 않아야 한다.
#   변하지 않았다면, 검출 증가가 주입 때문이라는 통제 실험이 성립한다.

# ============================================
# 9단계: DOE 기반 요인설계 분석 (2^3 Factorial Design)
# ============================================
# [이 단계 결과물]
# - 센서 3개를 각각 정상(-)/이탈(+) 2수준으로 두고 8가지 조합을 전부 검증
# - 각 센서의 '주효과'와 센서 간 '교호작용'을 분리해 정량화
# - 목적: "개별 센서는 정상 범위인데 조합에서 문제가 발생"하는 구조를 수치로 확인
#
# [왜 DOE인가]
#   기존 시나리오(S1~S5)는 조합이 임의적이라 "센서 A와 B가 함께 있을 때 시너지가 있나"에
#   답할 수 없다. 요인설계는 가능한 조합을 체계적으로 전부 돌려 효과를 분리한다.
#   공정 통합(PI) 관점에서 '모듈 간 간섭'을 정량화하는 방법론에 해당한다.

import itertools

# --------------------------------------------
# 9-1. 요인 선정 및 설계 설정
# --------------------------------------------
FACTORS = [59, 21, 103]    # 요인으로 사용할 센서 3개 (SHAP 기여도 상위 중 선정)
LEVEL = 2.0                # 이탈 수준: +2σ (단독으로는 약한 이탈)
N_BATCH = 200              # 조건당 웨이퍼 수 (많을수록 추정이 안정적)
N_REP = 5                  # 반복 횟수 (효과의 표준오차 추정용)

print("=" * 60)
print(f"2^3 요인설계: 센서 {FACTORS}, 각 +{LEVEL}σ")
print("=" * 60)

def make_condition(base_batch, combo):
    """combo는 (0,1,0) 같은 형태. 1인 센서만 이탈시킨다."""
    b = base_batch.copy()
    for sensor, on in zip(FACTORS, combo):
        if on:
            med, sig = robust_stats(sensor)
            b.iloc[:, b.columns.get_loc(sensor)] = med + LEVEL * sig
    return b

# --------------------------------------------
# 9-2. 8개 조건 실행 (1회차 - 설계표 출력용)
# --------------------------------------------
pool = np.where(y == 0)[0]
batch_doe = X_final.iloc[np.random.default_rng(42).choice(pool, N_BATCH, replace=False)].copy()

design_rows = []
for combo in itertools.product([0, 1], repeat=3):
    b = make_condition(batch_doe, combo)
    p = final_model.predict_proba(b.values)[:, 1]
    design_rows.append({
        '조건': ''.join('+' if v else '-' for v in combo),
        'X1(59)': +1 if combo[0] else -1,
        'X2(21)': +1 if combo[1] else -1,
        'X3(103)': +1 if combo[2] else -1,
        '평균확률': p.mean(),
        '검출률': (p >= THRESHOLD).mean(),
    })
df_design = pd.DataFrame(design_rows)
print("\n[설계표] 8개 조건 실행 결과")
print(df_design.to_string(index=False))

# --------------------------------------------
# 9-3. 효과 계산 (반복 실험으로 표준오차까지)
# --------------------------------------------
# 코드화 계수 방식: 효과 = (부호 × 응답).평균 × 2
# 부호가 +1인 조건들의 평균에서 -1인 조건들의 평균을 뺀 값과 같다.
def compute_effects(responses, codes):
    eff = {}
    for i in range(3):
        eff[f'X{i+1}'] = (codes[:, i] * responses).mean() * 2
    for i, j in itertools.combinations(range(3), 2):
        eff[f'X{i+1}X{j+1}'] = ((codes[:, i] * codes[:, j]) * responses).mean() * 2
    eff['X1X2X3'] = ((codes[:, 0] * codes[:, 1] * codes[:, 2]) * responses).mean() * 2
    return eff

rep_effects = []
for rep in range(N_REP):
    b0 = X_final.iloc[np.random.default_rng(100 + rep).choice(pool, N_BATCH, replace=False)].copy()
    resp, codes = [], []
    for combo in itertools.product([0, 1], repeat=3):
        b = make_condition(b0, combo)
        p = final_model.predict_proba(b.values)[:, 1]
        resp.append(p.mean())
        codes.append([1 if v else -1 for v in combo])
    rep_effects.append(compute_effects(np.array(resp), np.array(codes)))

df_eff = pd.DataFrame(rep_effects)
print(f"\n[효과 분석] 응답변수: 평균 불량확률, {N_REP}회 반복")
print("효과      | 추정값    | 표준편차  | 판정")
for k in ['X1', 'X2', 'X3', 'X1X2', 'X1X3', 'X2X3', 'X1X2X3']:
    v = df_eff[k]
    verdict = "유의" if abs(v.mean()) > 2 * v.std() else "미미"
    print(f"{k:9s} | {v.mean():+.4f}  | {v.std():.4f}   | {verdict}")

# --------------------------------------------
# 9-4. 가법성 검정 (교호작용의 실질적 크기)
# --------------------------------------------
# "각각 따로 넣은 효과의 합" vs "동시에 넣은 효과"를 비교한다.
# 동시가 더 크면 시너지(양의 교호작용)가 존재한다는 뜻.
base_prob_doe = final_model.predict_proba(batch_doe.values)[:, 1].mean()

solo_effects = {}
for c in FACTORS:
    b = batch_doe.copy()
    med, sig = robust_stats(c)
    b.iloc[:, b.columns.get_loc(c)] = med + LEVEL * sig
    solo_effects[c] = final_model.predict_proba(b.values)[:, 1].mean() - base_prob_doe

b_all = make_condition(batch_doe, (1, 1, 1))
combined_effect = final_model.predict_proba(b_all.values)[:, 1].mean() - base_prob_doe
additive_sum = sum(solo_effects.values())

print(f"\n[가법성 검정]")
print(f"  기준 상태 평균 불량확률: {base_prob_doe:.4f}")
for c in FACTORS:
    print(f"  센서 {c} 단독 투입 시 증가분: {solo_effects[c]:+.4f}")
print(f"  ─────────────────────────────────────")
print(f"  단순 합 (효과가 독립이라 가정): {additive_sum:+.4f}")
print(f"  실제 동시 투입 결과          : {combined_effect:+.4f}")
print(f"  차이 (교호작용)              : {combined_effect - additive_sum:+.4f} "
      f"({(combined_effect / additive_sum - 1) * 100:+.1f}%)")

# --------------------------------------------
# 9-5. 저장
# --------------------------------------------
joblib.dump({
    'factors': FACTORS,
    'level': LEVEL,
    'design_table': df_design,
    'effects_mean': df_eff.mean().to_dict(),
    'effects_std': df_eff.std().to_dict(),
    'solo_effects': solo_effects,
    'additive_sum': float(additive_sum),
    'combined_effect': float(combined_effect),
    'base_prob': float(base_prob_doe),
}, 'artifacts/doe.joblib')
print("\n저장 완료: artifacts/doe.joblib")

# --------------------------------------------
# [ DOE 결과 해석 - 참고용 정리 ]
# --------------------------------------------
# ● 주효과(main effect): 해당 센서 하나만 이탈시켰을 때 응답이 얼마나 변하는가
#   - X1(센서59)이 가장 크다 → 단일 감시 우선순위가 가장 높은 항목
#
# ● 교호작용(interaction): 두 센서가 함께 이탈했을 때, 각각의 효과 합보다
#   더 크거나 작게 나타나는 부분
#   - 모두 양(+)으로 나타남 → 함께 이탈하면 효과가 증폭되는 시너지 구조
#   - 공정 통합(PI) 관점에서 '모듈 간 간섭'에 해당하는 현상
#
# ● 가법성 검정이 더 직관적이다:
#   각각 따로 넣으면 합계 +0.09 수준인데, 동시에 넣으면 +0.11이 나온다.
#   → 개별 항목이 모두 관리 범위 내여도, 조합에서는 예상보다 큰 이탈이 발생한다.
#   → 단일 센서 관리도(SPC)만으로는 이 구조를 감지할 수 없다.
#
# ● 응답변수 선택에 대한 주석:
#   '검출률'(임계값 초과 비율)로 분석하면 일부 교호작용이 통계적으로 유의하지 않게 나온다.
#   이진 판정 과정에서 정보가 손실되기 때문이다.
#   '평균 불량확률'(연속형)이 더 민감한 응답변수이며, DOE에서는 가능한 한
#   연속형 응답을 사용하는 것이 표준이다.
#
# ● 한계:
#   교호작용의 크기는 모델 구조에 영향받는다. 본 모델은 max_depth=2로
#   트리 하나가 최대 2개 변수까지만 조합하며, colsample_bytree=0.3으로
#   트리마다 일부 센서만 참조한다. 따라서 3차 교호작용(X1X2X3)은 거의 0에 가깝다.
#   이는 데이터에 3차 상호작용이 없다는 뜻이 아니라, 모델이 그것을 학습할
#   구조가 아니라는 뜻이다.

# ============================================
# 10단계: 배치 관리도 기준선 산출 (SPC)
# ============================================
# [이 단계 결과물]
# - 정상 웨이퍼 배치를 반복 추출해 '위험 판정 개수'의 자연 변동을 측정
# - 이 분포로부터 관리한계(UCL)를 산출
# - 각 시나리오가 기준선에서 얼마나 벗어났는지 Z값과 p값으로 평가
#
# [왜 필요한가]
#   어떤 배치에서 위험 판정이 10장 나왔을 때, 이것이 이상인지 정상 변동인지
#   판단하려면 '정상 공정에서는 몇 장이 나오는가'라는 기준선이 필요하다.
#   이 기준선 없이 관리도를 그리면 선을 그어놓고 근거를 대지 못한다.

from scipy import stats

BATCH_SIZE = 30      # 판정 단위 (시나리오의 주입 구간과 동일)
N_TRIALS = 200       # 반복 추출 횟수

# --------------------------------------------
# 10-1. 정상 배치의 자연 변동 측정
# --------------------------------------------
normal_pool = np.where(y == 0)[0]
baseline_counts = []

for seed in range(N_TRIALS):
    r = np.random.default_rng(seed)
    idx = r.choice(normal_pool, BATCH_SIZE, replace=False)
    p = final_model.predict_proba(X_final.iloc[idx].values)[:, 1]
    baseline_counts.append((p >= THRESHOLD).sum())

baseline_counts = np.array(baseline_counts)
bl_mean = baseline_counts.mean()
bl_std = baseline_counts.std()
UCL = bl_mean + 3 * bl_std          # 관리 상한
LCL = max(0, bl_mean - 3 * bl_std)  # 관리 하한 (개수는 음수 불가)

print("=" * 60)
print(f"배치 관리도 기준선 ({BATCH_SIZE}장 배치, {N_TRIALS}회 반복 추출)")
print("=" * 60)
print(f"정상 배치의 위험 판정 개수")
print(f"  평균 {bl_mean:.2f}장, 표준편차 {bl_std:.2f}장")
print(f"  관측 범위 {baseline_counts.min()}~{baseline_counts.max()}장")
print(f"  95% 구간 {np.percentile(baseline_counts, 2.5):.0f}~{np.percentile(baseline_counts, 97.5):.0f}장")
print(f"  관리 상한(UCL, 평균+3σ) = {UCL:.1f}장")
print(f"  관리 하한(LCL, 평균-3σ) = {LCL:.1f}장")

# --------------------------------------------
# 10-2. 각 시나리오를 기준선과 비교
# --------------------------------------------
# Z값: 기준선 평균에서 표준편차 몇 개만큼 벗어났는가
# p값: 정상 공정에서 이 정도 개수가 우연히 나올 확률 (이항검정)
print(f"\n각 시나리오의 통계적 평가")
print("시나리오                  | 검출 | Z값    | p값     | 관리한계 판정")

scenario_eval = {}
for key, sc in scenarios.items():
    if key == 'S5':
        continue   # 드리프트는 배치 전체가 변하므로 별도 처리
    detected = int((sc['prob'][:N_INJECT] >= THRESHOLD).sum())
    z = (detected - bl_mean) / bl_std
    pval = stats.binomtest(detected, BATCH_SIZE, bl_mean / BATCH_SIZE,
                           alternative='greater').pvalue
    verdict = "초과 (이상 신호)" if detected > UCL else "한계 내"
    scenario_eval[key] = {'detected': detected, 'z': float(z), 'p': float(pval),
                          'over_ucl': bool(detected > UCL)}
    print(f"{sc['name'][:24]:24s} | {detected:3d}  | {z:+.2f}  | {pval:.4f}  | {verdict}")

# --------------------------------------------
# 10-3. 저장
# --------------------------------------------
joblib.dump({
    'baseline_counts': baseline_counts,
    'mean': float(bl_mean),
    'std': float(bl_std),
    'ucl': float(UCL),
    'lcl': float(LCL),
    'batch_size': BATCH_SIZE,
    'n_trials': N_TRIALS,
    'p95_low': float(np.percentile(baseline_counts, 2.5)),
    'p95_high': float(np.percentile(baseline_counts, 97.5)),
    'scenario_eval': scenario_eval,
}, 'artifacts/spc_baseline.joblib')
print("\n저장 완료: artifacts/spc_baseline.joblib")

# --------------------------------------------
# [ 결과 해석 - 참고용 정리 ]
# --------------------------------------------
# ● 이 기준선이 의미하는 것:
#   모델은 정상 웨이퍼도 일정 비율로 위험 판정한다(오탐). 중요한 것은
#   그 오탐 비율이 '일정하다'는 점이다. 정상 배치 30장에서 평균 5.25장이
#   꾸준히 나온다면, 그 자체가 기준선이 된다.
#   → 개별 판정이 부정확해도 '개수의 분포'는 예측 가능하다.
#
# ● 다만 한계도 분명하다:
#   표준편차가 1.94장으로, 평균 대비 37% 수준이다. 30장 배치로는 변동이 크다.
#   실제로 약한 신호(S2)는 기준선 안에 묻혀 감지되지 않는다.
#   → 배치 크기를 키우면 개선된다. 표준오차는 표본 수의 제곱근에 반비례하므로,
#     100장 배치로 판정하면 상대 변동이 약 절반으로 줄어든다.
#
# ● 앞선 해석의 정정:
#   시나리오 기준 배치의 주입 구간에서 관측된 '주입 전 2장'은
#   정상 배치 분포(평균 5.25장)의 하위 2.5% 수준에 해당하는 값이다.
#   따라서 '2장 → 13장'이라는 비교는 기준값이 우연히 낮아 효과가 과대평가된 것이며,
#   기대값 기준으로는 '5.25장 → 13장'으로 약 2.5배 증가가 정확한 표현이다.