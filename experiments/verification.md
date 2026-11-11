# 검증 기준 — 내가 제대로 했는지 확인하는 방법

목적: 내 말을 믿지 않고 **직접 확인**할 수 있게 하는 것. 명령과 기대 출력을
같이 적고, **틀렸을 때 무엇을 뜻하는지**도 적는다. 통과만 적어 두면 통과가
무엇을 보장하는지 알 수 없다.

전제: 저장소 루트에서 `PYTHONPATH=.` 로 실행한다.

---

## 0. 한 줄 요약 — 세 개만 보면 된다

```bash
PYTHONPATH=. python -m pytest tests/ -q                         # ① 164개 통과
PYTHONPATH=. python experiments/grid_convergence.py --quick  # ② A*/h ≈ 1.8
PYTHONPATH=. python -m pytest tests/test_input_check.py -q      # ③ 29개 통과
```

①이 깨지면 내가 뭔가 망가뜨렸다. ②가 다른 값을 주면 바닥 측정이 재현되지
않는다. ③이 깨지면 새 검증 루프가 작동하지 않는다.

**①은 11 분쯤 걸린다.** 한 번에 돌리기 부담스러우면 세 묶음으로 쪼갠다 —
개수까지 맞춰 보면 빠뜨린 파일이 없는지도 같이 확인된다.

```bash
PYTHONPATH=. python -m pytest tests/test_analytic_bates.py tests/test_ballistics.py \
  tests/test_burnrate.py tests/test_code_defects.py tests/test_diff_burnback.py \
  tests/test_export.py tests/test_grain_multiseg.py tests/test_input_check.py -q
#   → 116 passed  (약 35 s)
PYTHONPATH=. python -m pytest tests/test_ode_judge.py tests/test_convergence.py -q
#   →  13 passed  (약 375 s — 가장 느린 묶음)
PYTHONPATH=. python -m pytest tests/test_optimize.py tests/test_reference_compare.py -q
#   →  35 passed  (약 195 s)
```

합 **164**. 시작 시점의 기준선은 **133** 이었고 이번에 31 개를 더했다
(`test_input_check.py` 29 + 새 바닥 시험 1 + 부스트-서스테인 모순 시험 1).
164 가 아니면 시험 파일이 빠졌거나 내가 적은 개수가 틀렸다.

---

## 1. 격자 수렴 — 진폭 바닥

### 1.1 재현

```bash
PYTHONPATH=. python experiments/grid_convergence.py            # 약 240 s
PYTHONPATH=. python experiments/grid_convergence.py --eps      # 약 240 s
```

기대 (± 는 재현 허용폭이고, 그 밖이면 **보고와 다르다**):

| 실행 | 봐야 할 줄 | 기대 |
|---|---|---|
| 기본 | `A_abs/h` 범위 | **0.84 ~ 1.02** (폭 1.21×) |
| 기본 | n=121 격자오차 중앙 | **3.65 %** ±0.05 |
| 기본 | 격자오차의 차수 | n 121→321 에서 4.0 배 감소 (≈ h^1.4) |
| `--eps` | 마지막 줄 | "**바닥은 완화 폭을 따라간다**" |
| `--eps` | `A/eps` 범위 | **0.62 ~ 0.79** (폭 1.27×) |

**틀리면 무슨 뜻인가.** `A/eps` 의 폭이 `A/h` 의 폭보다 **크게** 나오면 내
결론이 뒤집힌다 — 그러면 `MODEL_FLOOR_K · eps_cells · h` 가 아니라 `h` 만의
식이어야 한다. 폭이 둘 다 2 배를 넘으면 둘 다 아니고 다른 변수가 있다.

### 1.2 식이 실제로 바뀌었는지

```bash
PYTHONPATH=. python -c "
from gre.optimize.runner import amplitude_floor
print(amplitude_floor(14.0, h=0.5,  eps_cells=1.5))  # (0.6,  '모델')
print(amplitude_floor(99.0, h=0.5,  eps_cells=1.5))  # r0 를 7배로 — 같아야 한다
print(amplitude_floor(14.0, h=0.25, eps_cells=1.5))  # h 절반 — 0.3
print(amplitude_floor(14.0, h=0.01, eps_cells=1.5))  # 아주 고운 격자 — 0.012
print(amplitude_floor(14.0))                         # 격자 없음 → 휴리스틱 꼬리표
"
```

기대: `(0.6, '모델')` / `(0.6, '모델')` / `(0.3, '모델')` / `(0.012, '모델')` /
`(0.14, '모델(휴리스틱)')`.

**둘째 줄이 첫째와 다르면** 식에 아직 `r₀` 가 남아 있다는 뜻이고, 이번 수정의
핵심이 안 들어간 것이다. **넷째 줄이 `(0.02, '제작')` 이면** δ 가 아직 판정에
끼어 있다는 뜻이다 — δ 는 2026-10-16 에 판정에서 뺐다 (지배항이 렌더링 공차가
만든 자기충족적 값이다). **다섯째 줄의 꼬리표가 `'모델'` 이면** 호출자가 격자를
안 넘긴 사실이 보고서에서 사라진다 — 그 구분이 있어야 "휴리스틱으로 판정된
결과" 를 나중에 걸러낼 수 있다.

### 1.3 결론이 문서와 코드에서 같은지

```bash
grep -n "MODEL_FLOOR_K" gre/optimize/runner.py | head -3
grep -n "0.80 · eps_cells · h\|0.80 · eps" experiments/grid_results.md
```

`runner.py` 의 상수와 결과 문서의 식이 같아야 한다. 다르면 둘 중 하나가 stale.

---

## 2. 입력 검증 루프

### 2.1 자기 일관된 목표는 통과해야 한다 (거짓 양성 없음)

```bash
PYTHONPATH=. python -c "
import torch
from gre.optimize.forward import DiffMotor
from gre.optimize.loss import Target
from gre.optimize.constraints import ManufacturingLimits
from gre.validate.input_check import check_inputs
m=DiffMotor(d_outer=60.0,n_grid=121); z=torch.zeros(8,dtype=m.dtype)
t,p,f,_=m.simulate(torch.tensor(14.0,dtype=m.dtype),z,z,[95.0],torch.tensor(80.0,dtype=m.dtype))
r=check_inputs(Target(t=t.detach(),f=f.detach()),m,ManufacturingLimits(),(95.0,),a_throat=80.0)
print(r.text())
"
```

기대: **확정 불가 0**, `runnable = True`. 경고는 `FMT_RISE_RES` 하나만 (정방향
모델의 점화 상승이 표본 3 개로 분해되는 것은 **사실**이다 — 거짓 양성이 아니다).

**확정 불가가 하나라도 나오면** 검사가 자기 모델의 출력을 거부하는 것이므로
그 검사가 틀렸다. 이 항목이 검증 루프의 가장 중요한 기준이다 — 멀쩡한 입력을
거부하는 쪽이 못 잡는 쪽보다 나쁘다.

### 2.2 역산이 정방향 모델과 맞는지 (루프 전체의 기반)

```bash
PYTHONPATH=. python -m pytest tests/test_input_check.py -q \
  -k "inversion_matches_the_forward_model"
```

기대: 2 개 통과. 그 시험이 단언하는 것:

| 양 | 중앙 오차 | 최대 |
|---|---|---|
| 압력 역산 `P = F/(C_f·A_t)` | < 1 % (실측 **0.24 %**) | < 3 % (실측 1.03 %) |
| 연소면적 역산 `ρ_p·A_b·r = P·A_t/c*` | < 2 % (실측 **0.73 %**) | < 5 % (실측 2.39 %) |

**이게 깨지면 나머지 검사를 읽을 필요가 없다.** 요구 압력·면적·웹이 전부 이
역산 위에 서 있다.

### 2.3 깨진 입력을 실제로 잡는지 (거짓 음성 없음)

```bash
PYTHONPATH=. python -m pytest tests/test_input_check.py -q
```

29 개 통과. 각 검사가 **무엇을 잡는지**를 직접 보려면:

```bash
PYTHONPATH=. python - <<'EOF'
import numpy as np, torch
from gre.optimize.forward import DiffMotor
from gre.optimize.loss import Target
from gre.optimize.constraints import ManufacturingLimits
from gre.validate.input_check import check_inputs
m=DiffMotor(d_outer=60.0,n_grid=121); lim=ManufacturingLimits()
z=torch.zeros(8,dtype=m.dtype)
t0,_,f0,_=m.simulate(torch.tensor(14.0,dtype=m.dtype),z,z,[95.0],torch.tensor(80.0,dtype=m.dtype))
t0,f0=t0.detach().numpy(),f0.detach().numpy()
def fire(tag,t,f,**kw):
    r=check_inputs(Target(t=torch.tensor(t),f=torch.tensor(f)),m,lim,(95.0,),**kw)
    print(f'{tag:24s} runnable={str(r.runnable):5s} {[x.code for x in r.findings if not x.ok]}')
fire('원본',t0,f0,a_throat=80.0)
fire('시간 역행',np.concatenate([t0[:300],t0[200:]])[:len(f0)],f0)
fire('큰 음수 추력',t0,np.where(np.arange(len(f0))<50,-0.3*f0.max(),f0))
fire('잡음 5%',t0,f0+0.05*f0.max()*np.sin(np.arange(len(f0))*2.1))
fire('연소 5배 (웹)',t0*5,f0,a_throat=80.0)
fire('추력 10배',t0,f0*10,a_throat=80.0)
EOF
```

기대: 원본만 `runnable=True`. 나머지는 각각 `FMT_TIME` / `FMT_SIGN` /
`FMT_NOISE` / `PHYS_WEB`+`BOUNDS` / `BOUNDS` 가 걸린다.

**"연소 5배" 가 `PHYS_WEB` 없이 통과하면** 자유 목 웹 검사가 다시 공허해진
것이다 (이번에 고친 결함 #3 의 재발).

### 2.4 등급의 의미가 지켜지는지

"확정 불가" 는 **상한이 엄밀할 때만** 쓴다는 규약이 있다. 확인:

```bash
grep -n "FATAL\|WARN" gre/validate/input_check.py | grep "Finding(" 
```

`AB_REACH` 와 `PHYS_THROAT_FIXED` 가 **WARN** 이어야 한다 (둘 다 경계가
하한/비엄밀). 이 둘이 FATAL 로 바뀌어 있으면 만들 수 있는 목표를 거부할 수
있다. §4.6.3 표가 "연소면적 실현 가능성" 을 하드 검사처럼 적어 둔 것은
**과한 표현**이고 그 이유는 `_ab_upper_bound` 주석에 적혀 있다.

---

## 3. 전면 검토 — 보고한 문제가 실재하는지

전면 검토에서 지적한 문제를 직접 확인하는 방법. **네 개만 재현해도
보고서의 신뢰도를 가늠할 수 있다.**

### 3.1 평균 스텝 웹 궤적 (§2.1, 가장 큰 지적)

```bash
PYTHONPATH=. python - <<'EOF'
import torch, numpy as np
from gre.optimize.forward import DiffMotor
m=DiffMotor(d_outer=60.0,n_grid=121); z=torch.zeros(8,dtype=m.dtype)
a=z.clone(); a[2]=2.2
t,p,f,d=m.simulate(torch.tensor(11.0,dtype=m.dtype),a,z,[95.0],torch.tensor(78.0,dtype=m.dtype))
r=m.br.rate(p).detach().numpy(); hl=np.asarray(d['h_list'],float)
w_mean=np.concatenate([[0],np.cumsum(r[:-1]*float(d['h']))])        # evaluate 가 쓰는 것
w_true=np.concatenate([[0],np.cumsum(r[:-1]*hl[:len(r)-1])])        # 실제 스케줄
cap=float(d['w_tab'][-1]); tt=t.detach().numpy()
def hit(w): 
    i=np.flatnonzero(w>=0.999*cap); return tt[i[0]] if i.size else float('nan')
print(f'평균 스텝 소진 {hit(w_mean):.2f} s   실제 스케줄 {hit(w_true):.2f} s   h 평균 {float(d["h"])*1e3:.2f} ms')
EOF
```

**고치기 전**에는 두 시각이 2 배 가까이 달랐다 (평균 4.26 s / 실제 8.76 s).
**고친 뒤**에도 이 비교 자체는 같은 값을 준다 — 이 스크립트가 두 방식을 **직접**
계산하기 때문이다. 확인할 것은 `evaluate` 가 어느 쪽을 쓰는지다:

```bash
grep -n 'h_list' gre/optimize/runner.py | head -3
```

기대: `evaluate` 안에서 `diag["h_list"]` 를 쓰는 줄이 나온다. `diag["h"]` 로
궤적을 쌓는 줄이 남아 있으면 수정이 안 들어간 것이다.

### 3.2 `fp_ratio_spread` 가 모순된 쌍을 통과시키는지 (§2.2)

```bash
PYTHONPATH=. python - <<'EOF'
import numpy as np, torch
from gre.optimize.loss import fp_ratio_spread
from gre.core.nozzle import thrust_coefficient, optimum_area_ratio
t=np.linspace(0,2,2000); P=np.where(t<0.5,5.0,1.5); P[0]=P[-1]=0.2
pa,g,At=0.1013,1.133,80.0; eps=optimum_area_ratio(3.0,pa,g)
cf=np.array([thrust_coefficient(max(x,0.15),pa,eps,g) for x in P])
F=cf*P*At; F[0]=F[-1]=0.0
Pb=P.copy(); Pb[t<0.5]*=2.0; Pb[0]=0.2          # 부스트 압력만 2배 — 모순된 쌍
r=fp_ratio_spread(torch.tensor(t),torch.tensor(F),torch.tensor(Pb))
print(f'변동 {100*r["spread"]:.2f} %   창 안 {r["n"]}/2000   (문턱 20 %)')
EOF
```

**고친 뒤의 기대: 변동 61.45 %, 창 안 1998/2000** — 걸려야 한다.
(2026-10-16 이전에는 **0.00 %, 창 안 499/2000** 으로 **통과**시켰다. 그 숫자가
나오면 수정이 안 들어간 것이다.)

같은 자리에서 자기 일관된 쌍도 확인한다 — 이쪽이 더 중요하다:

```bash
PYTHONPATH=. python -c "
import numpy as np, torch
from gre.optimize.loss import fp_ratio_spread
from gre.core.nozzle import thrust_coefficient, optimum_area_ratio
t=np.linspace(0,2,2000); P=np.where(t<0.5,5.0,1.5); P[0]=P[-1]=0.2
pa,g,At=0.1013,1.133,80.0; eps=optimum_area_ratio(3.0,pa,g)
cf=np.array([thrust_coefficient(max(x,0.15),pa,eps,g) for x in P])
F=cf*P*At; F[0]=F[-1]=0.0
r=fp_ratio_spread(torch.tensor(t),torch.tensor(F),torch.tensor(P),eps=eps)
print(f'참 ε: A_t={r[\"at_mean\"]:.3f} mm² (참 80.0), 변동 {100*r[\"spread\"]:.4f} %')
"
```

기대: `A_t=80.000 mm², 변동 0.0000 %`. **추정자의 검산이다** — 참 ε 를 주면
함의 노즐목이 정확히 복원되어야 한다. 안 되면 `C_f` 를 나누는 식이 틀렸다.

### 3.3 `feasibility_bounds` 가 최대추력을 안 보는지 (§1 #9)

```bash
PYTHONPATH=. python -c "
import torch
from gre.optimize.forward import DiffMotor
from gre.optimize.loss import Target
from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.runner import feasibility_bounds
m=DiffMotor(d_outer=60.0,n_grid=121)
t=torch.linspace(0,2,500,dtype=m.dtype); f=torch.full((500,),30.0,dtype=m.dtype); f[250]=1e5
r=feasibility_bounds(m,Target(t=t,f=f),ManufacturingLimits(),(95.0,))
print('F_max',float(f.max()),'총역적',round(r['I_target'],1),'feasible',r['feasible'])
"
```

기대: `F_max 100000.0` 인데 **`feasible True`**. 100 kN 스파이크가 통과한다 —
최대추력 한계를 검사하지 않는다는 증거다.

### 3.4 세장비 결과가 설계 제약에 안 걸려 있는지 (§2.3)

```bash
PYTHONPATH=. python -c "
import torch
from gre.optimize.constraints import fill_feasible, FILL_SEVERITY_MAX
z=torch.zeros(8,dtype=torch.float64); a=z.clone(); a[5]=7.0   # 6-로브 깊음
g=fill_feasible(torch.tensor(14.0,dtype=torch.float64),a,z)
print(f'한계 {FILL_SEVERITY_MAX}  깊이 {g[\"depth\"]:.2f}  폭 {g[\"width\"]:.2f}  "
      f"깊이²/폭 {g[\"severity\"]:.1f}  ok={g[\"ok\"]}')
"
grep -n '로브 충전' gre/optimize/runner.py
```

기대: `한계 267.0  깊이 14.00  폭 7.39  깊이²/폭 26.5  ok=True`, 그리고 둘째
명령이 `admissibility` 안의 **"로브 충전"** 행을 출력한다.

**둘째 명령이 비면** 게이트가 안 들어간 것이다 (2026-10-16 이전 상태 — 그때는
폭 0.5 × 깊이 20 mm 로브가 전부 통과했다).

⚠ **`ok=True` 가 정상이다.** 이 게이트는 현 설계 족에서 걸리지 않는다 — 푸리에
로브가 넓어서 `깊이²/폭` 이 한계와 한 자리 차이다. 그래도 둔 이유는 보고서
§0 ★(나) 에 있다. 실제로 걸리는지 보려면 좁은 로브를 넣어야 한다:

```bash
PYTHONPATH=. python -c "
from gre.export.coupon import fill_limit
r=fill_limit()
print('안 찬다고 판정:', r['short'], '  (관측: C1, C2)')
"
```

기대: `['C1', 'C2']` — 관측과 **정확히** 같아야 한다. 18 개 중 한 칸이라도
어긋나면 기준이 틀렸다 (`test_fill_limit_reproduces_all_eighteen_specimens`
가 같은 것을 단언한다).

---

## 3.5 (A)(B) 재측정 — 결론이 재현되는지

둘 다 비싸다 (각 ~70 분, ~45 분, 이 컨테이너 2 코어 기준). **결정론적이므로
같은 숫자가 나와야 한다** — 컨테이너 재시작으로 웜 팔을 두 번 돌렸고
149 반복·모양 9.54 %·복잡도 0.54 가 **글자까지 같게** 재현됐다.

```bash
PYTHONPATH=. python experiments/warmstart_vs_multistart.py \
    --iters-per-k 60 --k-compare 2        # (A)
PYTHONPATH=. python experiments/k3_at_n241.py --grid 241            # (B-1)
PYTHONPATH=. python experiments/k3_at_n241.py --multistart \
    --grid 241 --k 3 --iters 60 --seeds 4 # (B-2)
```

꼭 봐야 할 네 숫자:

| 어디 | 기대 | 틀리면 |
|---|---|---|
| (A) 6-로브 웜 | 집합 **안**, 부족 0.00 %p, 복잡도 0.54, 149 반복 | 웜 사슬 구현이 `minimal_complexity_search` 와 어긋났다 |
| (A) 6-로브 독립 | 집합 **밖**, 부족 9.92 %p, 복잡도 4.99 | 예산이 안 맞거나 씨앗이 독립이 아니다 |
| (A) 예산 열 | 두 팔이 **같은 수** (149=147, 138=138) | 비교가 불공정하다 — 가장 중요한 확인이다 |
| (B-1) K=3 | 복잡도 **5.42 mm**, 부족 0.2 %p | 목표를 다른 격자에서 만들었을 가능성 |

⚠ **(A) 를 모양 오차로 읽으면 결론이 뒤집힌다** (독립이 7.67 < 웜 9.54). 그게
설계 문서 이전까지의 숫자가 쓴 측도이고, 철회한 이유다. 비교는 **집합까지의
거리**로 할 것.

⚠ **(B-2) 가 0/4 로 나오는 것이 정상이다.** 이 시험은 "재시작으로 0.2 %p 를
넘을 수 있나" 를 묻고 답이 '아니다' 다. 1/4 이상 들어가면 결론이 바뀐다 —
그때는 0.2 %p 가 나쁜 국소 최소였다는 뜻이다.

---

## 4. 회귀 — 이번에 고친 것이 다시 깨지는지

이번 수정마다 **그 결함 자체를 잡는 시험**을 붙였다. 결함을 되돌리면 그 시험이
깨져야 한다. 확인하려면 해당 줄을 되돌리고 그 시험만 돌리면 된다.

| 고친 것 | 깨져야 하는 시험 |
|---|---|
| `amplitude_floor` 에 `h` 없음 | `test_model_floor_scales_with_relaxation_width_not_r0` |
| 자유 목 웹 검사 공허 | `test_web_overrun_is_fatal_in_both_throat_modes` (A_t=None 쪽) |
| 점화·테일오프 거짓 양성 | `test_vertical_ignition_and_tailoff_are_never_rejected` |
| 도달 범위가 케이스압 무시 | `test_throat_reach_is_capped_by_the_case_pressure` |
| `dt ≥ τ` 에서 '통과' 로 적음 | `test_rise_check_cannot_judge_when_dt_exceeds_tau` |
| τ 식이 정방향과 다름 | `test_rise_check_tau_is_the_forward_models_formula` |
| 창이 준위 마스크 (input_check 쪽) | `test_window_is_an_interval_not_a_level_mask_for_boost_sustain` |
| float64 경계 판정 | `test_model_floor_scales_with_relaxation_width_not_r0` (격자 7201/1801) |
| **평균 스텝 웹 궤적** | `test_optimize.py` 전체 — 창 경계가 틀어지면 모양 오차가 바뀌어 여러 시험이 동시에 깨진다 |
| **F/P 추정자 (부스트-서스테인)** | `test_contradiction_check_sees_a_boost_sustain_pair` |
| **충전 기준 (세장비 → 깊이²/폭)** | `test_fill_limit_reproduces_all_eighteen_specimens` |
| **죽은 단언 `or True`** | `test_export_coupon_writes_files_and_reports` |
| **δ 가 판정에 끼어듦** | `test_model_floor_scales_with_relaxation_width_not_r0` (`DELTA_TRANSFER_MM` 이 `amplitude_floor` 소스에 없음을 단언) |

**이 중 하나라도 결함을 되돌렸는데 통과하면** 그 시험은 거짓 확신이다 —
보고서 §3.2 가 기존 시험 11 개에서 바로 그것을 찾아낸 목록이다.

---

## 5. 이 작업이 **보장하지 않는** 것

기준을 적는 자리에 한계도 적는다. 아래는 확인해도 안 나온다.

1. **진폭 바닥은 필요조건이다.** "S > E" 는 모드가 곡선을 격자 오차보다 크게
   바꾼다는 뜻이고, **최적화기가 그 차이를 쓸 수 있다**는 뜻이 아니다. `r₀` 와
   `A_t` 가 자유롭기 때문에 작은 모드 변화는 보상될 수 있다.
2. **바닥 위에 있다고 진폭이 정확히 복원되지 않는다.** 거친 격자는 응답을 전
   구간에서 깎는다 (n=121 은 기준의 25~60 %) — 진폭을 체계적으로 과대 추정한다.
   이 보정은 **안 했다.**
3. **진폭 바닥의 `k`·`r₀` 의존성은 안 쟀다** (`k=3`, `r₀=14` 하나뿐).
4. **입력 검증은 전부 필요조건이다.** 통과는 "불가능이 확정되지 않았다" 일
   뿐이고 실현 가능을 뜻하지 않는다. 등급 규약이 그 구분이다.
5. **`AB_REACH` 의 두 경계는 둘 다 엄밀하지 않다.** 도달 범위는 단일 모드
   훑기의 하한, 최소 특징 상한은 띠 비겹침 가정이 쐐기 끝(`r_tip_min` 0.8 <
   `t_prop_min` 2.0)에서 깨진다.
6. **전면 검토는 전수 검사가 아니다.** 네 방향으로 나눠 봤지만 §5 에 적은
   미확인 항목이 남아 있고, 특히 설계 문서 힌지 궤적 표는 둘레 수정의 영향을
   정면으로 받는 양인데 재측정 비용 때문에 못 돌렸다.
7. **승인 7 건은 2026-10-16 에 전부 적용했다** (보고서 §0). 그러면서 §2.4 의
   처방이 틀렸던 것이 드러나 기준을 `깊이²/폭` 으로 다시 잡았다 — **내 진단이
   맞아도 처방은 틀릴 수 있다**는 사례다.
8. **로브 충전 게이트는 현 설계 족에서 걸리지 않는다.** 들어갔다는 것이
   "설계 공간이 그 한계로 묶인다" 는 뜻은 아니다 (보고서 §0 ★(나)).
9. **K=3 의 진폭 근거는 n=241 에서 복구됐지만 집합 진입은 0.2 %p 모자란다.**
   복잡도 5.42 mm 로 바닥 0.300 의 18 배다. 다만 재시작으로는 그 0.2 %p 를
   못 넘는다 (독립 씨앗 0/4). 더 고운 격자·τ 재검토·다른 최적화 설정 중
   **어느 것도 안 했다.**
10. **운용 격자(n=121)는 참 설계보다 16 배 단순한 형상을 복원한다** (0.428 vs
   참값 7.0). n=241 이 0.77 배로 거의 맞다. **운용 격자를 올리는 결정은 안 했다** —
   비용이 반복당 9.8 → 11.6 s 로 크지 않으므로 다음 후보다.
11. **`iters_per_k` 를 120 으로 올린 효과는 재측정하지 않았다.** 기본값으로
   호출하는 경로의 판정이 달라질 수 있다.
12. **(A) 는 K=2 한 점, 씨앗 3~4 개에서만 쟀다.** "독립 다중 출발이 나쁘다" 의
   통계적 강도는 제한적이다 (다만 6-로브 3/3, K=3 4/4 가 전부 집합 밖이었다).
13. **`k0_recheck` 가 n=241 에서 돌지 않았다** — 집합 진입을 못 했기 때문이다.
   H1(예산이 한 일) vs H2(모드가 한 일) 판정은 그 격자에서 **열려 있다.**
