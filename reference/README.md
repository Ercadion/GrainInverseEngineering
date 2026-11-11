# reference/ — 기존 시뮬레이터 사본 (읽기 전용 참조)

출처: `<프로젝트 폴더>\6DOF RocketSim with Unreal`
복사일: 2026-09-26

V1-b 독립 구현 대조용 사본이다. **원본 폴더는 수정하지 않는다.**
이 사본도 기능 개발에 쓰지 않고, 대조 실행에만 사용한다.

- `SRM_simul.py` — SDF 격자 burnback + 내탄도 (R_dot_calculation 이 Nakka KNSB 구간표 하드코딩)
- `srm_multiseg.py` — 다중 세그먼트 엔진
- `srm_bates_fast.py` — BATES 원형 코어 해석식 고속 경로
