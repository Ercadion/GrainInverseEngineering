# GrainInverseEngineering

**목표 추력 곡선을 넣으면 그 곡선을 내는 고체 로켓 모터의 그레인(포트) 형상을 찾아 주는 역설계 도구**입니다.

보통의 내탄도 시뮬레이터는 "이 형상이면 추력이 어떻게 나오는가"(정문제)를 계산합니다.
이 도구는 그 반대로 **"이 추력 곡선을 내려면 포트를 어떤 모양으로 만들어야 하는가"**(역문제)를 풉니다.
케이스 내경, 세그먼트 길이, 노즐목은 사용자가 정하는 **고정 입력**이고, 도구는 그 조건 안에서 포트 단면 형상만 찾습니다.
목표 곡선이 주어진 케이스로는 실현할 수 없으면 조건을 몰래 넓히지 않고 **실현 불가로 보고**합니다.

## 주요 기능

- **미분가능 정방향 모델**: 2D 거리장 burnback, 내탄도(챔버 충전 ODE), 노즐 모델을 PyTorch로 구현해 경사 기반 최적화를 합니다.
- **역설계 실행 모드 3가지**
  - *단일 실행*: 초기 형상(씨앗) 하나로 한 번 최적화합니다. 가장 빠릅니다.
  - *해 족*: 여러 출발점에서 최적화해 허용 오차 안에 드는 설계들을 표로 비교합니다.
  - *최소 복잡도 K 탐색*: 포트 윤곽의 모드 수 K를 늘려 가며 처음 통과하는 가장 단순한 형상을 찾습니다.
- **합격 판정**: 추력·압력의 모양, 총역적, 최대 추력, 최대 압력을 허용 오차(기본 10 %)로 판정합니다.
- **입력 자동 점검**: 목표 곡선이 물리적으로 그럴듯한지, 주어진 케이스·노즐목으로 실현 가능한지 미리 검사합니다.
- **제작 제약**: 최소 살 두께, 꼭짓점 반경 등 실제로 만들 수 있는 형상만 허용합니다.
- **CAD 내보내기**: 찾은 형상으로 그레인, 코어 핀, 2분할 실리콘 주형 마스터 등 몰드 일습을 STL(선택적으로 STEP)로 출력합니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
pip install -r requirements.txt
pip install flask
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- `flask`는 웹 UI에, `torch`는 역설계 계산에 필요합니다 (CPU 버전이면 충분합니다).
- STEP 파일까지 내보내려면 `pip install cadquery`가 추가로 필요합니다. 없으면 STL만 출력됩니다.
- `triangle`은 선택 의존성입니다. 설치하지 않아도 CAD 출력의 주 경로(`mapbox_earcut`)는 동작합니다.

## 사용 방법

### 1. 웹 UI 실행

```bash
python gre_web.py
```

브라우저에서 `http://localhost:5080`이 열립니다. 다른 옵션:

```bash
python gre_web.py --port 5085        # 포트 지정
python gre_web.py --no-browser       # 브라우저 자동 열기 끄기
python gre_web.py --selftest         # 서버 없이 계산 경로만 검증
```

### 2. 목표 곡선 입력

화면의 **목표 곡선** 항목에서 입력 방식을 고릅니다.

| 방식 | 설명 |
|---|---|
| 알려진 형상에서 생성 | 정해진 형상으로 곡선을 만든 뒤 다시 복원해 보는 시연용 |
| 사다리꼴 생성 | 최대 추력, 연소 시간 등으로 단순한 목표 곡선을 생성 |
| 표 붙여넣기 | `t, F[, P]` 표를 직접 붙여넣기 |
| CSV 파일 | `t, F[, P]` 열을 가진 CSV/TSV 파일 업로드 |

CSV 형식은 다음과 같습니다. 첫 줄 헤더와 `#` 주석 줄은 무시됩니다. 압력 열은 선택이며, 넣으면 압력도 판정 대상이 됩니다.

```
t_s,F_N,P_MPa
0.00000,0.547,0.10130
0.00590,127.101,2.98411
...
```

- 시간 `t` [s], 추력 `F` [N], 챔버 **절대압** `P` [MPa]
- 압력 열이 있어도 추력만으로 맞추고 싶으면 화면에서 *압력 열을 무시*를 선택합니다.

### 3. 케이스 · 노즐목 입력

CSV에는 곡선만 들어 있으므로 아래 값을 화면에 함께 넣어야 합니다.

- **케이스 내경 Ø** [mm]
- **세그먼트 길이** [mm]: 여러 세그먼트면 쉼표로 구분 (예: `74,74,74`)
- **노즐목**: 보유한 노즐이 있으면 *고정 직경*, 아니면 *범위*를 지정해 그 안에서 고르게 합니다.

추진제 종류(KNSB, KNDX, KNSU), 제작 제약, 허용 오차는 고급 항목에서 바꿀 수 있습니다. 기본값은 코드에서 읽어 채워집니다.

### 4. 실행 · 결과 확인

실행 모드를 고르고 실행하면 목표 곡선과 역설계 결과의 추력·압력 곡선, 복원된 포트 단면, 항목별 사양 오차, 합격 여부가 표시됩니다.
계산은 수 분이 걸릴 수 있으며 한 번에 하나의 작업만 실행됩니다.

### 5. CAD 내보내기

결과 화면에서 출력 폴더를 지정하고 내보내면 그레인과 몰드 부품이 STL로 저장됩니다. *STEP도 함께*를 선택하면 STEP 파일도 만듭니다 (`cadquery` 필요).

### 예시 데이터로 시작하기

`data/` 폴더에 바로 넣어 볼 수 있는 목표 곡선이 있습니다. 모두 알려진 설계를 정방향으로 계산해 만든 곡선이라 **답이 존재하는 것이 보장**되어 있습니다.

| 파일 | 케이스 내경 | 세그먼트 길이 | 노즐목 (고정) |
|---|---|---|---|
| `example_target_3lobe.csv` | 60 | 90 | Ø6.20 |
| `example_target_D44_3seg74.csv` | 44 | 74,74,74 | Ø10.5 |
| `example_target_D40_3seg74_x08.csv` | 40 | 74,74,74 | Ø10.5 |

각 파일 머리의 주석에 곡선을 만든 원래 설계와 확인된 결과가 적혀 있습니다.

### Python에서 직접 사용

웹 UI는 `gre` 패키지를 호출만 하므로, 같은 기능을 코드에서 바로 쓸 수 있습니다. 예를 들어 정방향 burnback 계산과 CAD 출력은 다음과 같습니다.

```python
from gre.core.burnback2d import SegmentBurnback2D
from gre.geometry.port import star_port
from gre.export import export_set

port = star_port(12.0, 6, 3.0)                       # 6각 스타 포트
seg = SegmentBurnback2D(port, d_outer=60.0, n=601)
w = seg.web_grid(200)
ab = seg.burn_area(w, 80.0)                          # 웹 w에 따른 연소 면적

rep = export_set(port, d_outer=60.0, length=95.0, outdir="out/parts")
```

`python demo_burnback.py`를 실행하면 원형·스타 포트의 거리장과 연소 면적 곡선을 그린 그림이 만들어집니다.

## 프로젝트 구조

```
gre_web.py      역설계 웹 UI (입력 · 실행 · 결과 · CAD 내보내기)
t2_input_web.py 몰드 탈형 시험에서 잰 값을 제작 제약(t_prop_min)으로 환산하는 보조 도구
demo_burnback.py 정방향 burnback 데모

gre/core        미분가능 burnback, 내탄도, 연소 속도, 노즐
gre/geometry    포트 윤곽 표현(푸리에 등)
gre/optimize    정방향 모델 래퍼, 손실 함수, 제작 제약, 최적화 실행기
gre/validate    입력 자동 점검, 식별 가능성, 침식 연소 진단
gre/export      STEP/STL, 몰드 부품, 시편
data/           예시 목표 곡선
experiments/    검증 · 분석 스크립트
reference/      비교 검증용 기존 시뮬레이터 사본 (수정 금지)
tests/          해석해 대조, 수렴, 회귀 테스트
```

## 테스트

```bash
python -m pytest tests -q
```

`torch`가 없으면 미분가능 모델 관련 테스트는 건너뜁니다.

## 참고 문헌

저작권 때문에 논문 원문(PDF)은 저장소에 넣지 않았습니다. 아래 제목으로 찾을 수 있습니다.

- *Burnback Analysis of Solid Propellant Rocket Motors*
- Wei et al. (2018), *Combined Acceleration Methods for Solid Rocket Motor Grain …*, International Journal of Aerospace Engineering
- *Solid rocket motor propellant grain burnback simulation based on fast minimum distance function calculation*
- 「인공신경망에 기반한 고체 연료 추진제 형상 최적화」

## 라이선스

[MIT](LICENSE) © 2026 SongSangHyun

선택 의존성 `triangle`은 Jonathan Shewchuk의 Triangle을 감싼 것이며, Triangle 자체는 상업적 사용에 별도 허락이 필요합니다.
이 저장소는 그 코드를 포함하지 않습니다.
