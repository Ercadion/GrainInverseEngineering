"""설계 → 제작 파일. 그레인과 몰드 일습을 한 번에 뽑는다.

제작 경로 (§5.1) — **형상이 두 번 반전된다.**

    ① 코어 몰드 (3D 프린팅, 강체)   ─실리콘 주입→  ② 코어 핀 (실리콘, 포트 양형)
    ② 코어 핀 + 지관 + 엔드 플레이트 ─KNSB 압축 캐스팅→  ③ 그레인 (포트 음형)

따라서 **포트 윤곽은 ① 에서 음형, ② 에서 양형, ③ 에서 다시 음형**이다. 반전마다
수축·전사 오차가 쌓이므로 시편 시험 T3 이 그 누적을 잰다.

한 벌에 들어가는 것

| 파일 | 무엇 | 왜 |
|---|---|---|
| `grain` | 설계 그레인 자체 | 검증·시각화·체적 확인용 (출력하지 않는다) |
| `core_mold` | 코어 핀을 뜨는 음형 | 3D 프린팅 |
| `core_pin` | 실리콘 코어 (참고 형상) | 치수 확인·간섭 점검 |
| `piston` | 압축 캐스팅 피스톤 | 지관 안에서 추진제를 누른다 |
| `end_plate` | 엔드 플레이트 | 코어 핀을 중심에 세운다 |

⚠ **수축률과 간극은 입력이다.** 시편 시험 T4 전에는 수축률을 모르므로 기본 0 이고,
   그때 나온 파일에는 "수축 미보정" 이 보고서에 적힌다. 간극을 0 으로 두면 부품이
   서로 안 들어간다 — 기본값은 보수적으로 잡되 반드시 실측으로 확인할 것.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..geometry.port import FourierPort
from . import profile as P
from . import solid as S


@dataclass
class FitSpec:
    """끼워맞춤 간극과 공정 보정 [mm, %]. 간극은 **실측으로 확정해야 하는 값이다.**

    ⚠ **수축은 사슬에 두 번 있다.** 종전에는 하나로 셌는데, 형상이 두 번
       반전되므로 수축도 두 번 일어난다.

           코어 몰드 ──실리콘 경화(−`silicone_shrink_pct`)→ 코어 핀
           코어 핀   ──추진제 캐스팅(−`shrink_pct`)────────→ 그레인 포트

       둘을 합쳐 놓으면 T4 에서 추진제 수축을 잴 때 실리콘 몫이 섞여 들어간다.
       그래서 따로 둔다 — `silicone_shrink_pct` 는 제품 자료로 **이미 아는 값**이고
       (Mold Max 30: 0.002 in/in = 0.2 %), `shrink_pct` 만 T4 로 재면 된다.
    """

    #: [%] 실리콘 경화 수축. 제품 자료 0.2 % **+ 실측 검증**:
    #: 배킹층 긴 변 173.326 → 172.97 mm = 0.2055 %. 자료값과의 차 0.0055 %p 는
    #: 캘리퍼스 한 눈금(0.01 mm on 173 mm)이라 **분해능 안에서 일치**한다.
    #: ⚠ 0.2055 로 올리지 않는다 — 한 번의 측정이 주지 않는 정밀도를 가장하게 된다.
    #: 전제: 마스터의 FDM 출력 오차는 **0 으로 본다** (2026-10-15 결정). 그래서
    #: 위 0.2055 % 는 순수 실리콘 수축이다. 전사 오차가 나중에 설명되지 않으면
    #: **가장 먼저 의심할 가정**이고, 마스터의 같은 치수를 재면 되돌릴 수 있다.
    silicone_shrink_pct: float = 0.2
    shrink_pct: float = 0.0        # 추진제 수축, T4 전까지 0 — 핀을 이만큼 키운다
    pin_clearance: float = 0.15    # 피스톤 구멍 ↔ 코어 핀 (실리콘은 눌리므로 작게)
    tube_clearance: float = 0.30   # 피스톤·엔드플레이트 외경 ↔ 지관 내경
    mold_wall: float = 6.0         # 코어 몰드 벽 두께
    mold_base: float = 5.0         # 코어 몰드 바닥 두께
    #: [mm] 핀이 그레인 **위쪽**으로 더 나오는 길이. 피스톤(단차 + 플랜지)을
    #: 통과하고 그 위로 뽑기 공구가 물 만큼 있어야 한다 — `pin_overhang_needed()`.
    pin_overhang: float = 28.0
    #: [mm] 핀이 그레인 **아래쪽**으로 나오는 길이 — 엔드 플레이트 자리에 앉을
    #: 만큼만 있으면 된다. 위아래를 같게 두면 코어 몰드가 그만큼 길어지고
    #: 실리콘을 그만큼 더 먹는다. 비대칭이지만 **조용히 틀릴 수 없다** —
    #: 거꾸로 끼우면 짧은 쪽이 위로 와서 공구가 물 자리가 없다.
    pin_overhang_bot: float = 7.0
    plate_thick: float = 8.0       # 엔드 플레이트 플랜지 두께
    piston_thick: float = 10.0     # 피스톤 플랜지 두께
    seat_depth: float = 5.0        # 엔드 플레이트의 핀 자리 깊이
    chord_tol: float = P.DEFAULT_CHORD_TOL

    # ───────────────────────────────────────────────────── 세그먼트 간격 (입력)
    #: [mm] **단차** — 플러그가 지관 안으로 들어가는 깊이. 플랜지가 지관 테두리에
    #: 닿아 멈추므로 이 값이 **기계적 하드 스톱**이고, 그래서
    #:
    #:     추진제 길이 = 지관 길이 − 2 · end_inset
    #:     세그먼트 간격 = 2 · end_inset        ← 연료끼리 만나는 부분
    #:
    #: 사용자 확인 (2026-10-29): 외부 시뮬은 세그먼트 **사이**를 4 mm 로 보지만,
    #: 공정 편의상 **양끝 각 4 mm** 로 찍으므로 실제 간격은 **8 mm** 다. 고정값이
    #: 아니라 **입력**이다 — 간격을 바꾸려면 이 값만 바꾼다.
    #:
    #: ⚠ 체적이 이 값에 **직접** 걸린다. 92 % 충전으로 넣을 질량은
    #:   `export_report` 의 `propellant` 블록이 계산해 준다.
    end_inset: float = 4.0
    #: [mm] 지관 벽 두께 — 플랜지가 **테두리를 덮어야** 하드 스톱이 생기므로
    #: 필요하다. 플랜지 반경 = 지관 외경/2 + `flange_margin`.
    #:
    #: **2026-10-29 사용자 실측 확정: 지관 내경 40.0 mm · 벽 2.0 mm.**
    #: 그 전까지는 가정값 2.0 이었고 보고서에 "가정값" 경고가 붙었다 — 공교롭게
    #: 실측이 같은 값이라 이미 나간 파일의 형상은 바뀌지 않았다.
    #:
    #: ⚠ **이것이 틀리면 코드는 잡을 수 없다.** 플랜지 반경은 이 값을 더해서
    #: 정하므로, 플랜지는 벽 두께와 무관하게 늘 `flange_margin` 만큼 내민다 —
    #: 즉 입력이 틀렸다는 신호가 기하학에 남지 않는다. 실제 벽이 입력보다 두꺼우면
    #: 플랜지가 실물 테두리를 덜 덮고, `flange_margin` 보다 두꺼우면 아예 못 덮어
    #: **플러그가 끝까지 들어가** 추진제 길이가 공정마다 달라진다. 그래서 이 값은
    #: 반드시 실측이어야 한다.
    #: [mm] 지관 벽 두께 — **2.0 고정** (사용자 2026-10-30 "지관 벽
    #: 두께는 2mm 고정으로 하자"; 2026-10-29 실측 확정값과 같다).
    #: 몰드 홈 치수가 전부 여기서 파생되므로 바꾸면 전부 따라 바뀐다.
    #: ⚠ 틀린 값을 넣어도 코드는 못 잡는다 — 입력 검증의 사각이다.
    tube_wall: float = 2.0
    flange_margin: float = 5.0     # 플랜지가 지관 외경보다 더 내미는 폭

    # ───────────────────────────────────────────────────── 탈형 관찰용 기능
    #: [mm] **씰 압입량** — 추진제 쪽 끝면에서 구멍을 포트보다 이만큼 작게 뚫어
    #: 실리콘 핀을 물린다. 페이스트가 핀↔판 틈으로 올라가 **플래시 고리**를 만들면
    #: 탈형 때 그 고리가 가시를 부러뜨리므로, 그 경로를 막는다.
    #: 실리콘(Shore A 30)은 눌리므로 0.2 mm 는 조립을 막지 않는다.
    pin_seal: float = 0.20
    seal_len: float = 2.0          # 씰 띠의 축방향 길이
    #: [mm] **각도 기준 구멍** — 양끝 판의 θ=0 을 맞추기 위한 블라인드 구멍.
    #: 핀이 포트 형상이라 자리와 키가 이미 맞물려 있어 **회전 방지는 내재**지만,
    #: 탈형 뒤 "몇 번 가시가 부러졌나" 를 말하려면 θ 원점이 남아 있어야 한다.
    index_hole_d: float = 2.5
    #: 핀 뽑기 공구 — 보어가 단계적으로 좁아져 실리콘을 물고, 위에서 볼트 두 개로
    #: **축방향**으로 당긴다. 비스듬히 당기면 가시가 전단되므로 손으로 잡지 않는다.
    puller_steps: tuple = (0.30, 0.0, -0.30, -0.60)
    puller_step_len: float = 3.0
    puller_wall: float = 4.0
    puller_cap_thick: float = 8.0
    puller_bar_w: float = 12.0     # 손잡이 막대 폭
    puller_bar_h: float = 10.0     # 손잡이 막대 높이
    #: 파편 받이 접시 — 탈형 중 떨어지는 가시 파편을 받아 센다 (파단 vs 미충전 판별).
    tray_radius: float = 45.0
    tray_depth: float = 8.0
    tray_wall: float = 4.0

    # ───────────────────────────────────────────────────── 충전 질량 환산
    # ───────────────────── 2분할 실리콘 주형 — **사용자 CAD 치수 그대로**
    #: 사용자가 올린 `Silicone_Casting_Mold_Plate/Top` 의 **지름 사다리**를
    #: 그대로 옮긴 것이다 (2026-10-29 사용자 지시).
    #:
    #:     원본  Ø72 · Ø64 · Ø54 · Ø50 · 코어 Ø20
    #:     우리  Ø62 · Ø54 · Ø44 · Ø40 · 코어 **별 단면**
    #:
    #: 간격이 그대로다: 72−64 = 8 = 62−54,  64−54 = 10 = 54−44,
    #: 54−50 = 4 = 44−40. 즉 전부 **10 mm 씩 내린** 것이고, 그 결과
    #: **Ø40/Ø44 가 우리 지관의 내경/외경과 정확히 맞는다.**
    #:
    #: 구조 (바깥 → 안)
    #:
    #:     Ø62 ~ Ø54   벽
    #:     Ø54 ~ Ø44   **바깥 홈** — 지관 벽이 들어가 앉는다 (깊이 `groove_depth`)
    #:     Ø44 ~ Ø40   **서 있는 2 mm 링** — 지관을 안쪽에서 잡는다
    #:     Ø40 ~ 코어  보어 (깊이 `bore_depth`)
    #:     코어        별 단면 공동
    mold_wall_w: float = 4.0        # Ø54 → Ø62 (반경 4)
    mold_groove_w: float = 5.0      # Ø44 → Ø54 (반경 5) — 바깥 홈
    mold_groove_depth: float = 15.0  # 위 면에서 홈 바닥까지
    mold_bore_depth: float = 9.0    # 위 면에서 Ø40 보어 바닥까지
    mold_ring_recess: float = 4.0   # 위 면에서 **서 있는 링 윗면**까지 (사용자 지정)
    mold_floor: float = 3.0         # 코어 공동 아래 바닥
    #: [mm] Plate 높이가 **그레인 길이보다 얼마나 높은가** (사용자 2026-10-30:
    #: "마스터 높이가 입력으로 받은 그레인 길이를 받아서 이 길이보다 10mm 더
    #: 높게"). 그래서 `mold_h` 라는 **상수는 더 이상 없다** — 높이는
    #: `mold_height(length, fit)` 로 길이에서 나온다.
    #:
    #: ⚠ 초판은 `mold_h = 53.0` 상수였다. 사용자 Onshape 원본의 높이를 그대로
    #:   베낀 값이라, 세그먼트 길이를 바꿔도 마스터가 53 mm 로 고정이었다 —
    #:   74 mm 그레인에 53 mm 마스터가 조용히 나왔다.
    #:
    #: 별 공동 깊이는 `h − bore_depth − floor` = `length + 10 − 9 − 3`
    #: = **`length − 2`** 다 (별 구멍이 홈 띠까지 이어지므로 홈 바닥이 아니라
    #: **보어 바닥**까지다). 추진제 길이 `length − 2·end_inset` = `length − 8`
    #: 보다 6 mm 길어 코어가 추진제를 덮는다.
    #:
    #: ⚠ 초판 주석은 공동이 `length − 8` 이라 **추진제와 정확히 같다**고 적었다.
    #:   틀렸다 — 별 구멍이 band 1 과 2 **둘 다**에 들어간다는 것을 안 보고
    #:   홈 바닥까지로 셌다. `test_star_cavity_depth_...` 가 38 ≠ 32 로 잡았다.
    mold_head: float = 10.0
    mold_top_back: float = 3.0      # Top 에서 홈 위에 남는 살
    #: [mm] Top 의 **코어 기둥이 가장 바깥쪽 벽보다 솟는 높이**. 원본의 Ø20
    #: 포스트가 테(Ø72, 높이 18)보다 5 mm 솟아 있던 것과 같다 (사용자
    #: 2026-10-29 "가운데가 올라와 있어야지", 2026-10-30 "가장 바깥쪽 벽의
    #: 높이보다 5mm 더 높게 **항상**").
    #: Top 의 바깥 벽 높이는 `groove_depth + top_back` 이고 기둥은 그 위에서
    #: 이만큼 더 간다 — `test_top_post_rises_above_the_outer_wall` 이 건다.
    mold_top_post: float = 5.0
    chord_tol_mold: float = 0.02

    rho_ideal_g_cc: float = 1.841  # KNSB 이론 최대 밀도
    pack_frac: float = 0.92        # 목표 충전율 — 넣는 질량으로 맞춘다

    def seal_len_eff(self) -> float:
        """[mm] 실제로 쓰는 씰 띠 길이 — 단차의 절반을 넘지 못한다.

        씰 띠는 플러그 **안쪽**에 있어야 한다. 단차가 작으면 요청한 `seal_len` 이
        플러그보다 길어지므로 여기서 줄인다 — 띠가 짧아도 씰은 씰이고, 거부하면
        작은 단차를 아예 못 쓴다. 줄어들면 `export_report` 에 적힌다.
        """
        return min(float(self.seal_len), 0.5 * float(self.end_inset))

    @property
    def seg_gap(self) -> float:
        """[mm] 연료끼리 만나는 부분의 간격 = 2 · `end_inset`."""
        return 2.0 * float(self.end_inset)

    def plug_radius(self, d_outer: float) -> float:
        """지관에 들어가는 플러그 반경 [mm]."""
        return float(d_outer) / 2.0 - float(self.tube_clearance)

    def flange_radius(self, d_outer: float) -> float:
        """지관 테두리를 덮는 플랜지 반경 [mm] — 테두리에 **닿아서 멈춘다**."""
        return float(d_outer) / 2.0 + float(self.tube_wall) + float(self.flange_margin)

    def pin_overhang_needed(self) -> float:
        """[mm] 핀 오버행이 적어도 이만큼은 돼야 한다 — 피스톤을 통과하고 공구가 물 만큼."""
        return (float(self.end_inset) + float(self.piston_thick)
                + len(self.puller_steps) * float(self.puller_step_len))

    @property
    def mold_shrink_pct(self) -> float:
        """코어 몰드 캐비티에 걸리는 **합성** 보정 [%] — 두 수축을 곱한다.

        `(1+a)(1+b) − 1` 이지 `a + b` 가 아니다. 지금 크기에선 차이가 작지만
        (0.2 % + 2 % → 2.204 % vs 2.2 %), 더하기로 적어 두면 T4 값이 커졌을 때
        조용히 틀린다.
        """
        a = 1.0 + float(self.silicone_shrink_pct) / 100.0
        b = 1.0 + float(self.shrink_pct) / 100.0
        return (a * b - 1.0) * 100.0


@dataclass
class ExportReport:
    """파일과 함께 나가는 숫자들 — 이게 없으면 파일을 믿을 근거가 없다."""

    design: dict
    fit: dict
    fidelity: dict
    parts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    propellant: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["제작 파일 일습", ""]
        d, f = self.design, self.fidelity
        lines.append(f"  설계   포트 r₀ {d['r0']:.3f} mm, 모드 {d['n_modes']} 개, "
                     f"그레인 외경 {d['d_outer']:.1f} mm, 길이 {d['length']:.1f} mm")
        lines.append(f"  충실도 표본 {f['n_theta']} 점 · 최대 반경 편차 "
                     f"{f['sagitta_mm']:.4f} mm · 포트 면적 {f['area_err_pct']:+.4f} % · "
                     f"초기 둘레 {f['perimeter_err_pct']:+.4f} %")
        lines.append(f"  보정   실리콘 수축 {self.fit['silicone_shrink_pct']:.2f} % + "
                     f"추진제 수축 {self.fit['shrink_pct']:.2f} % "
                     f"→ 몰드 캐비티 {self.fit['mold_shrink_pct']:+.3f} % · "
                     f"핀 간극 {self.fit['pin_clearance']:.2f} mm · "
                     f"지관 간극 {self.fit['tube_clearance']:.2f} mm")
        if self.propellant:
            p = self.propellant
            lines.append(f"  단차   양끝 각 {p['end_inset_mm']:.1f} mm "
                         f"→ **세그먼트 간격 {p['seg_gap_mm']:.1f} mm** · "
                         f"추진제 길이 {p['length_mm']:.1f} mm "
                         f"(지관 {p['tube_length_mm']:.1f} − 2·{p['end_inset_mm']:.1f})")
            lines.append(f"  충전   체적 {p['volume_mm3']:,.1f} mm³ · "
                         f"이론 {p['rho_ideal_g_cc']:.3f} g/cc × "
                         f"{p['pack_frac'] * 100:.0f} % "
                         f"→ **넣을 질량 {p['mass_target_g']:.2f} g** "
                         f"(100 % 면 {p['mass_full_g']:.2f} g)")
        lines.append("")
        for p in self.parts:
            wt = "수밀" if p["watertight"] else "**수밀 아님**"
            lines.append(f"  {p['name']:<12} 삼각형 {p['faces']:>7,d}  "
                         f"체적 {p['volume_mm3']:>12,.1f} mm³  {wt}  → {p['files']}")
        if self.warnings:
            lines.append("")
            lines += [f"  ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════ 부품
def grain_bands(port: FourierPort, d_outer: float, length: float,
                fit: FitSpec = None, n_theta: int = None):
    """설계 그레인의 띠 정의 — 단면이 일정하므로 한 장이다. 반환: (띠들, 충실도)."""
    fit = fit or FitSpec()
    pts, fid = P.polygon(port, fit.chord_tol, n_theta)          # shrink 0
    outer = P.circle(d_outer / 2.0, fit.chord_tol)
    return [(outer, [pts], 0.0, float(length))], fid


def grain(port: FourierPort, d_outer: float, length: float,
          fit: FitSpec = None, n_theta: int = None):
    """설계 그레인 — 바깥 원통에서 포트를 뺀 것. **수축 보정 없음**(설계 치수)."""
    bands, fid = grain_bands(port, d_outer, length, fit, n_theta)
    return S.stepped_plate(bands, name="grain"), fid


def core_pin(port: FourierPort, length: float, fit: FitSpec = None,
             n_theta: int = None):
    """실리콘 코어 핀 — 포트의 **양형**. 양 끝으로 `pin_overhang` 만큼 길다.

    수축 보정을 적용한다: 핀이 줄어든 뒤의 치수가 설계 포트가 되어야 한다.
    """
    fit = fit or FitSpec()
    pts, _ = P.polygon(port, fit.chord_tol, n_theta, shrink_pct=fit.shrink_pct)
    z0 = -float(fit.pin_overhang_bot)
    z1 = float(length) + float(fit.pin_overhang)
    return S.prism(pts, z0, z1, name="core_pin")


def core_mold_bands(port: FourierPort, length: float, fit: FitSpec = None,
                    n_theta: int = None):
    """코어 핀을 뜨는 음형 — 바닥 있는 통. 위가 열려 있어 실리콘을 붓는다.

    바깥은 사각 블록으로 둔다 (프린터 베드에 안정적이고 벽 두께가 균일하다).
    캐비티 깊이는 핀 전체 길이다.

    ⚠ **수축 보정이 두 번 걸린다.** 이 캐비티에서 굳은 실리콘이 줄어 핀이 되고
       (`silicone_shrink_pct`), 그 핀으로 뜬 포트가 또 줄어 그레인이 된다
       (`shrink_pct`). 캐비티는 둘을 **곱한 만큼** 커야 한다.
    """
    fit = fit or FitSpec()
    pts, _ = P.polygon(port, fit.chord_tol, n_theta,
                       shrink_pct=fit.mold_shrink_pct)
    rmax = float(np.linalg.norm(pts, axis=1).max())
    half = rmax + fit.mold_wall
    block = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    depth = float(length) + float(fit.pin_overhang) + float(fit.pin_overhang_bot)
    # ⚠ 종전에는 `union_z([벽, 바닥])` 이었다 — 두 껍질이 z=mold_base 에서 맞닿아
    #   그 접촉면이 **두 번** 들어 있었고, 블록 외곽 네 모서리는 **네 번** 쓰였다.
    #   `Mesh.watertight()` 는 꼭짓점 **인덱스**로 보므로 이걸 통과시켰다 (두 껍질의
    #   인덱스가 달라 각각 닫혀 있다). STL 을 읽어 **좌표로** 병합해 보고 잡았다
    #   출력에서 제일 중요한 부품이 단일 솔리드가 아니었던 것이다.
    return [
        (block, [None], 0.0, float(fit.mold_base)),
        (block, [pts],  float(fit.mold_base), float(fit.mold_base) + depth),
    ]


def core_mold(port: FourierPort, length: float, fit: FitSpec = None,
              n_theta: int = None):
    """코어 몰드 메쉬 — 띠 정의(`core_mold_bands`)를 단일 셸로 꿰맨 것."""
    return S.stepped_plate(core_mold_bands(port, length, fit, n_theta),
                           name="core_mold")


#: [mm] FDM 최소 벽 두께 — 노즐 0.4 mm × 2 둘레. 이보다 얇으면 슬라이서가 빼먹는다.
WALL_MIN_FDM: float = 0.8


def index_hole_walls(port: FourierPort, fit: FitSpec, d_outer: float) -> tuple:
    """각도 기준 구멍의 양쪽 벽 두께 (자리 쪽, 플러그 쪽) [mm]."""
    r_mid = 0.5 * (float(port.max_radius()) + fit.plug_radius(d_outer))
    rad = float(fit.index_hole_d) / 2.0
    r_seat = float(port.max_radius()) + float(fit.pin_clearance)   # 자리 최대 반경
    return (r_mid - rad) - r_seat, fit.plug_radius(d_outer) - (r_mid + rad)


def _index_hole(port: FourierPort, fit: FitSpec, d_outer: float):
    """각도 기준 블라인드 구멍 — θ=0, 포트 최대 반경과 플러그 반경의 중간.

    핀 자리가 포트 형상이라 **회전 방지는 이미 내재**다. 이 구멍은 그게 아니라
    **θ 원점을 남기기 위한 것**이다 — 탈형 뒤 "몇 번 가시가 부러졌나" 를 말하려면
    양끝 판이 같은 각도로 꽂혀 있었다는 기준이 필요하다.
    """
    r_mid = 0.5 * (float(port.max_radius()) + fit.plug_radius(d_outer))
    w_seat, w_plug = index_hole_walls(port, fit, d_outer)
    if min(w_seat, w_plug) < WALL_MIN_FDM:
        raise ValueError(
            f"각도 기준 구멍 Ø{fit.index_hole_d:g} (중심 r={r_mid:.2f}) 의 벽이 "
            f"너무 얇다 — 자리 쪽 {w_seat:.2f} mm, 플러그 쪽 {w_plug:.2f} mm "
            f"(최소 {WALL_MIN_FDM:g}). `index_hole_d` 를 줄이십시오")
    return P.circle(float(fit.index_hole_d) / 2.0, fit.chord_tol) \
        + np.array([r_mid, 0.0])


def piston_bands(port: FourierPort, d_outer: float, fit: FitSpec = None,
                 n_theta: int = None):
    """압축 캐스팅 피스톤 — **플랜지 달린 플러그**.

    아래 `end_inset` 만큼이 지관 안으로 들어가고, 그 위 플랜지가 지관 테두리에
    **닿아서 멈춘다**. 그 하드 스톱이 추진제 길이(= 지관 길이 − 2·`end_inset`)를
    정하므로 **체적이 공정에서 재현된다** — 눌린 깊이를 눈으로 맞추지 않는다.

    띠 구성 (아래 → 위)

        ① z[0, seal_len]            플러그·**씰 띠** (구멍 = 포트 − `pin_seal`)
        ② z[seal_len, end_inset]    플러그·헐거운 띠 (구멍 = 포트 + `pin_clearance`)
        ③ z[end_inset, +thick]      플랜지 (+ 각도 기준 블라인드 구멍)
    """
    fit = fit or FitSpec()
    pts, _ = P.polygon(port, fit.chord_tol, n_theta, shrink_pct=fit.shrink_pct)
    seal = P.offset_inward(pts, fit.pin_seal)
    loose = P.offset_outward(pts, fit.pin_clearance)
    r_plug = fit.plug_radius(d_outer)
    plug = P.circle(r_plug, fit.chord_tol)
    flange = P.circle(fit.flange_radius(d_outer), fit.chord_tol)
    idx = _index_hole(port, fit, d_outer)
    z1 = fit.seal_len_eff()
    z2 = float(fit.end_inset)
    z3 = z2 + float(fit.piston_thick)
    if not (0.0 < z1 < z2):
        raise ValueError(f"단차 {z2:g} mm 가 씰 띠를 품지 못한다")
    return [
        (plug,   [seal,  None], 0.0, z1),
        (plug,   [loose, None], z1,  z2),
        (flange, [loose, idx],  z2,  z3),
    ]


def piston(port: FourierPort, d_outer: float, fit: FitSpec = None,
           n_theta: int = None):
    """피스톤 메쉬 — 띠 정의(`piston_bands`)를 단일 셸로 꿰맨 것."""
    return S.stepped_plate(piston_bands(port, d_outer, fit, n_theta),
                           name="piston")


def end_plate_bands(port: FourierPort, d_outer: float, fit: FitSpec = None,
                    n_theta: int = None):
    """엔드 플레이트 — **플랜지 달린 플러그**, 핀을 세우는 블라인드 자리.

    피스톤을 위아래로 뒤집은 것과 같다: 플랜지가 아래(지관 테두리를 받친다),
    플러그가 위로 `end_inset` 만큼 지관에 들어간다. 자리는 **관통이 아니라**
    깊이 `seat_depth` 의 음각이고, 자리 **입구**(추진제 쪽)에 씰 띠가 있다.

        ① z[0, z_s]                 플랜지 (+ 각도 기준 블라인드 구멍, 아래로 열림)
        ② z[z_s, plate_thick]       플랜지 + 자리 (헐거운 띠)
        ③ z[plate_thick, z_t−seal]  플러그 + 자리 (헐거운 띠)
        ④ z[z_t−seal, z_t]          플러그 + **씰 띠**
    """
    fit = fit or FitSpec()
    pts, _ = P.polygon(port, fit.chord_tol, n_theta, shrink_pct=fit.shrink_pct)
    seal = P.offset_inward(pts, fit.pin_seal)
    loose = P.offset_outward(pts, fit.pin_clearance)
    plug = P.circle(fit.plug_radius(d_outer), fit.chord_tol)
    flange = P.circle(fit.flange_radius(d_outer), fit.chord_tol)
    idx = _index_hole(port, fit, d_outer)
    z_f = float(fit.plate_thick)                          # 플랜지 → 플러그
    z_t = z_f + float(fit.end_inset)                      # 맨 위
    z_s = z_t - float(fit.seat_depth)                     # 자리 바닥
    z_seal = z_t - fit.seal_len_eff()                     # 씰 띠 시작
    if not (0.0 < z_s < z_seal < z_t):
        raise ValueError(
            f"자리 깊이 {fit.seat_depth:g} / 씰 길이 {fit.seal_len_eff():g} 가 "
            f"플랜지 {fit.plate_thick:g} + 단차 {fit.end_inset:g} 와 안 맞는다 "
            f"(필요: 0 < 씰 길이 < 자리 깊이 < 플랜지 + 단차)")
    # ⚠ 자리 바닥이 플랜지 안에 있을 수도 있고(얕은 단차) 플러그 안에 있을 수도
    #    있다(깊은 단차). 띠를 손으로 나열하면 그 중 한 쪽에서 조용히 틀리므로
    #    **z 분기점에서 잘라** 띠를 만든다.
    cuts = sorted({0.0, z_s, z_f, z_seal, z_t})
    bands = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < 1e-9:
            continue
        zm = 0.5 * (lo + hi)
        bands.append((
            flange if zm < z_f else plug,
            [None if zm < z_s else (seal if zm >= z_seal else loose),
             idx if zm < z_f else None],
            lo, hi))
    return bands


def end_plate(port: FourierPort, d_outer: float, fit: FitSpec = None,
              n_theta: int = None):
    """엔드 플레이트 메쉬 — 띠 정의(`end_plate_bands`)를 단일 셸로 꿰맨 것."""
    return S.stepped_plate(end_plate_bands(port, d_outer, fit, n_theta),
                           name="end_plate")


def pin_puller_bands(port: FourierPort, fit: FitSpec = None, n_theta: int = None):
    """핀 뽑기 공구 — 보어가 단계적으로 좁아지는 슬리브 + 막힌 뚜껑 + 손잡이 막대.

    탈형 시험의 측정 대상은 **가시의 생존**이다. 손으로 핀을 비스듬히 잡아 뽑으면
    가시가 전단되어 **공구가 데이터를 만들어 낸다**. 그래서

        ① 보어가 포트 형상이라 각도가 저절로 맞는다 — **회전 방지가 내재**다
          (실리콘 핀을 비틀면 가시 뿌리가 먼저 전단된다).
        ② 보어가 `puller_steps` 만큼 단계적으로 좁아져, 밀어 끼우는 동안 실리콘을
          **길이 방향으로 고르게** 문다. 한 자리만 물면 그 자리만 늘어난다.
        ③ 위가 막힌 뚜껑 + 가로 막대. 막대 양끝에 봉을 걸면 **양손/바이스로
          대칭 축하중**을 줄 수 있다.

    `puller_steps` 는 포트 기준 오프셋 [mm] 이다 (양수 = 헐겁게, 음수 = 물린다).

    ⚠ 막대만으로 축방향이 **보장되지는 않는다** — 손으로 비틀면 비틀린다. 축하중이
      결론을 가르면 막대를 지그에 걸 것. 막대는 그 걸 자리를 만들어 주는 것이다.
    """
    fit = fit or FitSpec()
    pts, _ = P.polygon(port, fit.chord_tol, n_theta, shrink_pct=fit.shrink_pct)
    r_out = float(port.max_radius()) + float(fit.puller_wall)
    outer = P.circle(r_out, fit.chord_tol)
    bands, z = [], 0.0
    for d in fit.puller_steps:
        bore = (P.offset_outward(pts, d) if d >= 0.0
                else P.offset_inward(pts, -d))
        bands.append((outer, [bore], z, z + float(fit.puller_step_len)))
        z += float(fit.puller_step_len)
    bands.append((outer, [None], z, z + float(fit.puller_cap_thick)))   # 막힌 뚜껑
    z += float(fit.puller_cap_thick)
    # 손잡이 막대 — 원판 안에 **완전히 들어가야** 한다 (단차 면을 덮기 위해).
    hw = float(fit.puller_bar_w) / 2.0
    r_in = r_out - 1.0
    if hw >= r_in:
        raise ValueError(f"손잡이 막대 폭 {fit.puller_bar_w:g} 이 외경 "
                         f"{2 * r_out:g} 보다 두껍다")
    hl = float(np.sqrt(r_in ** 2 - hw ** 2))                 # 모서리가 원 안에
    bar = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    bands.append((bar, [None], z, z + float(fit.puller_bar_h)))
    return bands


def pin_puller(port: FourierPort, fit: FitSpec = None, n_theta: int = None):
    """뽑기 공구 메쉬 — 띠 정의(`pin_puller_bands`)를 단일 셸로 꿰맨 것."""
    return S.stepped_plate(pin_puller_bands(port, fit, n_theta),
                           name="pin_puller")


# ═══════════════════════════════ 2분할 실리콘 주형 — 사용자 CAD 치수 그대로
def mold_radii(port: FourierPort, fit: FitSpec, d_outer: float) -> tuple:
    """(코어 최대반경, Ø40, Ø44, Ø54, Ø62 에 해당하는 반경) [mm].

    지관에서 나온다: `r1` = 지관 내경/2, `r2` = 지관 외경/2. 나머지는 그 바깥으로
    `mold_groove_w`, `mold_wall_w` 만큼. 지관을 바꾸면 전부 따라 바뀐다.
    """
    r_core = float(port.max_radius()) * (1.0 + fit.mold_shrink_pct / 100.0)
    r1 = float(d_outer) / 2.0                       # Ø40 — 지관 내경
    r2 = r1 + float(fit.tube_wall)                  # Ø44 — 지관 외경
    r3 = r2 + float(fit.mold_groove_w)              # Ø54
    r4 = r3 + float(fit.mold_wall_w)                # Ø62
    if r_core >= r1 - WALL_MIN_FDM:
        raise ValueError(
            f"코어 최대반경 {r_core:.2f} 가 보어 {r1:.2f} 에 너무 가깝다 — "
            f"포트가 지관에 비해 크다")
    return r_core, r1, r2, r3, r4


def mold_height(length: float, fit: FitSpec = None) -> float:
    """[mm] 실리콘 주형 마스터의 높이 — **그레인 길이에서 나온다.**

    `h = length + fit.mold_head` (기본 +10). 상수가 아니라 길이의 함수인
    이유는 `FitSpec.mold_head` 주석에 적었다.
    """
    fit = fit or FitSpec()
    h = float(length) + float(fit.mold_head)
    need = float(fit.mold_groove_depth) + float(fit.mold_floor)
    if h <= need:
        raise ValueError(
            f"마스터 높이 {h:.1f} 가 너무 낮다 — 홈 깊이 "
            f"{fit.mold_groove_depth:g} + 바닥 {fit.mold_floor:g} = {need:g} 보다 "
            f"커야 별 공동이 생긴다 (그레인 길이 {length:g})")
    return h


def _mold_levels(fit: FitSpec, h: float) -> tuple:
    """(홈 바닥, 보어 바닥, 링 윗면) 의 z — 전부 **위 면에서** 잰다."""
    z_g = float(h) - float(fit.mold_groove_depth)
    z_b = float(h) - float(fit.mold_bore_depth)
    z_r = float(h) - float(fit.mold_ring_recess)
    if not (z_g < z_b < z_r < float(h)):
        raise ValueError(
            f"깊이 순서가 어긋난다 — 홈 {fit.mold_groove_depth:g} > 보어 "
            f"{fit.mold_bore_depth:g} > 링 {fit.mold_ring_recess:g} 이어야 한다")
    return z_g, z_b, z_r


def silicone_mold_plate_bands(port: FourierPort, d_outer: float, length: float,
                              fit: FitSpec = None, n_theta: int = None):
    """**아래 몸체** — 사용자 CAD 의 지름 사다리에 코어만 별 단면으로.

    높이는 `mold_height(length, fit)` = **그레인 길이 + 10**. 띠 (아래 → 위),
    아래는 length = 70 (h = 80) 일 때의 z:

        z 0  → 3    바닥 (살)
        z 3  → 65   **별 단면 공동**  ← 굳으면 코어 핀
        z 65 → 71   공동 + 바깥 홈(Ø44~Ø54)   ← 공동은 여기까지 이어진다
        z 71 → 76   Ø40 보어 + 바깥 홈      ← 사이에 **서 있는 2 mm 링**
        z 76 → 80   Ø54 카운터보어 (링 윗면이 드러난다)

    위쪽 세 단(홈·보어·링)의 깊이는 위 면에서 재므로 길이와 무관하고, 길어진
    만큼은 **전부 별 공동**으로 간다.

    지관은 바깥 홈(Ø44~Ø54)에 벽이 들어가고, 서 있는 링(Ø40~Ø44)이 안쪽을 잡는다.
    """
    fit = fit or FitSpec()
    star, _ = P.polygon(port, fit.chord_tol, n_theta, shrink_pct=fit.mold_shrink_pct)
    r_core, r1, r2, r3, r4 = mold_radii(port, fit, d_outer)
    h = mold_height(length, fit)
    z_g, z_b, z_r = _mold_levels(fit, h)
    z_f = float(fit.mold_floor)
    if not (z_f < z_g):
        raise ValueError("바닥이 홈 바닥보다 위에 있다")
    tol = fit.chord_tol_mold
    c1, c3, c4 = (P.circle(r1, tol), P.circle(r3, tol), P.circle(r4, tol))
    groove = (P.circle(r3, tol), P.circle(r2, tol))
    return [
        (c4, [None],  0.0, z_f, [None]),
        (c4, [star],  z_f, z_g, [None]),
        (c4, [star],  z_g, z_b, [groove]),
        (c4, [c1],    z_b, z_r, [groove]),
        (c4, [c3],    z_r, h,   [None]),
    ]


def silicone_mold_plate(port: FourierPort, d_outer: float, length: float,
                        fit: FitSpec = None, n_theta: int = None):
    return S.stepped_plate(
        silicone_mold_plate_bands(port, d_outer, length, fit, n_theta),
        name="silicone_mold_plate")


def silicone_mold_top_bands(port: FourierPort, d_outer: float,
                            fit: FitSpec = None, n_theta: int = None):
    """**위 뚜껑** — 같은 수치, 코어는 **솟은 기둥**.

    원본 Top 의 Ø20 포스트(테보다 5 돌출)를 별 단면으로 바꾼 것이다. 테 높이는
    `mold_groove_depth` + `mold_top_back` = 18, 기둥은 거기서 `mold_top_post`
    만큼 더 솟는다. 깊이는 전부 **아래 면 기준** (Plate 의 위 면과 마주 본다):
    홈 15 · 보어 9 · 링 4.

    ⚠ 코어가 **기둥**이라 아래 면에서 보면 그 둘레가 **환형 빈 자리**다 —
      단순 구멍이 아니다. `stepped_plate` 의 환형 홈으로 적는다.
    """
    fit = fit or FitSpec()
    star, _ = P.polygon(port, fit.chord_tol, n_theta, shrink_pct=fit.mold_shrink_pct)
    r_core, r1, r2, r3, r4 = mold_radii(port, fit, d_outer)
    # 바깥 벽(Ø62 원통)의 높이. 기둥은 **항상** 이보다 `mold_top_post` 만큼
    # 더 간다 (사용자 2026-10-30). Plate 와 달리 길이를 따라가지 않는다 —
    # 뚜껑이라 그레인 길이와 무관하다.
    h = float(fit.mold_groove_depth) + float(fit.mold_top_back)
    post = float(fit.mold_top_post)
    if post <= 0.0:
        raise ValueError("기둥이 솟지 않는다 — `mold_top_post` 는 양수여야 한다")
    z_g, z_b, z_r = _mold_levels(fit, h)
    tol = fit.chord_tol_mold
    c1, c3, c4 = (P.circle(r1, tol), P.circle(r3, tol), P.circle(r4, tol))
    g_out = (P.circle(r3, tol), P.circle(r2, tol))     # 바깥 홈 (지관 자리)
    return [
        (c4,   [None], 0.0, z_g, [None,  None]),
        (c4,   [None], z_g, z_b, [g_out, None]),
        (c4,   [None], z_b, z_r, [g_out, (c1, star)]),
        (c4,   [None], z_r, h,   [(P.circle(r3, tol), star), None]),
        (star, [None], h,   h + post, [None, None]),
    ]


def silicone_mold_top(port: FourierPort, d_outer: float, fit: FitSpec = None,
                      n_theta: int = None):
    return S.stepped_plate(
        silicone_mold_top_bands(port, d_outer, fit, n_theta),
        name="silicone_mold_top")


def fragment_tray_bands(fit: FitSpec = None):
    """파편 받이 접시의 띠 정의 — 바닥 한 장 + 벽 한 장(안쪽이 구멍)."""
    fit = fit or FitSpec()
    R = float(fit.tray_radius)
    outer = P.circle(R, fit.chord_tol)
    inner = P.circle(R - float(fit.tray_wall), fit.chord_tol)
    w, d = float(fit.tray_wall), float(fit.tray_depth)
    return [(outer, [None], 0.0, w), (outer, [inner], w, w + d)]


def fragment_tray(fit: FitSpec = None):
    """파편 받이 접시 — 탈형 중 떨어지는 가시 파편을 받는다.

    파단과 미충전을 가르는 근거 중 하나가 **파편이 실제로 나오는가**다. 접시가
    없으면 파편이 작업대로 흩어져 "몇 개" 를 말할 수 없다.
    """
    return S.stepped_plate(fragment_tray_bands(fit), name="fragment_tray")


# ════════════════════════════════════════════════════════════════ 일습 출력
def export_set(port: FourierPort, d_outer: float, length: float, outdir: str,
               fit: FitSpec = None, n_theta: int = None, step: bool = True,
               stl: bool = True) -> ExportReport:
    """그레인 + 몰드 일습을 `outdir` 에 쓰고, 충실도 보고서를 돌려준다."""
    fit = fit or FitSpec()
    os.makedirs(outdir, exist_ok=True)
    g, fid = grain(port, d_outer, length, fit, n_theta)
    pin_len = float(length) + float(fit.pin_overhang) + float(fit.pin_overhang_bot)
    builds = [
        ("grain", g),
        ("core_pin", core_pin(port, length, fit, n_theta)),
        ("core_mold", core_mold(port, length, fit, n_theta)),
        ("piston", piston(port, d_outer, fit, n_theta)),
        ("end_plate", end_plate(port, d_outer, fit, n_theta)),
        ("pin_puller", pin_puller(port, fit, n_theta)),
        ("fragment_tray", fragment_tray(fit)),
        ("silicone_mold_plate",
         silicone_mold_plate(port, d_outer, length, fit, n_theta)),
        ("silicone_mold_top", silicone_mold_top(port, d_outer, fit, n_theta)),
    ]
    # ── 추진제 체적과 넣을 질량. 단차가 하드 스톱이므로 길이가 **고정**이고,
    #    체적도 고정이다 — 92 % 는 넣는 질량으로 맞춘다 (사용자 2026-10-29).
    l_prop = float(length) - 2.0 * float(fit.end_inset)
    a_cross = np.pi * (float(d_outer) / 2.0) ** 2 - float(port.area())
    v_prop = a_cross * l_prop                                     # [mm³]
    m_full = float(fit.rho_ideal_g_cc) * v_prop / 1000.0          # [g]
    rep = ExportReport(
        design={"r0": float(port.r0), "n_modes": int(port.n_modes),
                "d_outer": float(d_outer), "length": float(length)},
        fit={**asdict(fit), "mold_shrink_pct": fit.mold_shrink_pct,
             "seg_gap": fit.seg_gap,
             "plug_radius": fit.plug_radius(d_outer),
             "flange_radius": fit.flange_radius(d_outer)},
        fidelity={k: v for k, v in vars(fid).items() if k != "notes"},
        warnings=list(fid.notes),
        propellant={
            "tube_length_mm": float(length),
            "end_inset_mm": float(fit.end_inset),
            "seg_gap_mm": fit.seg_gap,
            "length_mm": l_prop,
            "port_area_mm2": float(port.area()),
            "cross_area_mm2": float(a_cross),
            "volume_mm3": float(v_prop),
            "rho_ideal_g_cc": float(fit.rho_ideal_g_cc),
            "pack_frac": float(fit.pack_frac),
            "mass_full_g": m_full,
            "mass_target_g": m_full * float(fit.pack_frac),
        })
    if l_prop <= 0.0:
        rep.warnings.append(
            f"추진제 길이가 0 이하다 — 지관 {length:g} mm 에 단차 "
            f"2 × {fit.end_inset:g} mm 는 들어가지 않는다")
    if fit.seal_len_eff() < float(fit.seal_len) - 1e-12:
        rep.warnings.append(
            f"씰 띠를 {fit.seal_len:g} → {fit.seal_len_eff():g} mm 로 줄였다 "
            f"(단차 {fit.end_inset:g} mm 의 절반까지) — 씰은 유지되지만 물림이 짧다")
    rep.fit["seal_len_eff"] = fit.seal_len_eff()
    w_seat, w_plug = index_hole_walls(port, fit, d_outer)
    rep.fit["index_hole_wall_seat"] = w_seat
    rep.fit["index_hole_wall_plug"] = w_plug
    if fit.pin_overhang_bot < fit.seat_depth:
        rep.warnings.append(
            f"아래쪽 핀 오버행 {fit.pin_overhang_bot:g} mm < 자리 깊이 "
            f"{fit.seat_depth:g} mm — 핀이 자리에 바닥까지 닿지 않는다")
    need = fit.pin_overhang_needed()
    if fit.pin_overhang < need:
        rep.warnings.append(
            f"핀 오버행 {fit.pin_overhang:g} mm < 필요 {need:g} mm "
            f"(단차 {fit.end_inset:g} + 피스톤 {fit.piston_thick:g} + 공구 물림 "
            f"{need - fit.end_inset - fit.piston_thick:g}) — 뽑기 공구가 핀을 못 문다")
    # 플랜지 반경 = 테두리 반경 + `flange_margin` 이므로 **벽 두께와 무관하게**
    # 항상 그만큼 내민다. 못 덮는 경우는 `flange_margin ≤ 0` 뿐이다.
    # ⚠ 코드가 잡을 수 없는 쪽은 `tube_wall` **자체가 틀린** 경우다 — 실제 벽이
    #   입력보다 두꺼우면 플랜지가 실물 테두리를 덜 덮는다. 그래서 이 값은 실측해야
    #   하고, 지금은 실측값이다 (Ø40 · 벽 2.0, 사용자 2026-10-29).
    r_rim = float(d_outer) / 2.0 + float(fit.tube_wall)
    rep.fit["rim_radius"] = r_rim
    rep.fit["flange_overhang"] = fit.flange_radius(d_outer) - r_rim
    if fit.flange_margin <= 0.0:
        rep.warnings.append(
            f"⛔ **하드 스톱이 없다** — `flange_margin` {fit.flange_margin:g} mm 로는 "
            f"플랜지(R {fit.flange_radius(d_outer):.2f})가 지관 테두리(R "
            f"{r_rim:.2f})를 덮지 못한다. 플러그가 끝까지 들어가 추진제 체적이 "
            f"재현되지 않는다")
    if fit.shrink_pct == 0.0:
        rep.warnings.append(
            "추진제 수축 미보정 (시편 시험 T4 전) — 실리콘 수축 "
            f"{fit.silicone_shrink_pct:.2f} % 만 걸려 있다")
    if fit.silicone_shrink_pct == 0.0:
        rep.warnings.append(
            "실리콘 수축 0 % — 제품 자료의 수축비를 넣을 것 "
            "(Mold Max 30: 0.002 in/in = 0.2 %)")

    from . import step as step_mod
    for name, mesh in builds:
        files = []
        if stl:
            pth = os.path.join(outdir, f"{name}.stl")
            S.write_stl(pth, mesh)
            files.append(os.path.basename(pth))
        ok, diag = mesh.watertight()
        if not ok:
            rep.warnings.append(f"{name}: 수밀 아님 — {diag}")
        rep.parts.append({"name": name, "faces": int(len(mesh.f)),
                          "volume_mm3": mesh.volume, "watertight": bool(ok),
                          "files": ", ".join(files) or "-"})
    if step:
        made, why = step_mod.export_step(port, d_outer, length, outdir, fit, n_theta)
        if made:
            for p in rep.parts:
                if f"{p['name']}.step" in made:
                    p["files"] = (p["files"] + ", " if p["files"] != "-" else "") \
                                 + f"{p['name']}.step"
        else:
            # 낡은 STEP 이 새 STL 옆에 남지 않게 치운다
            stale = [f for f in os.listdir(outdir) if f.endswith(".step")]
            for f in stale:
                os.remove(os.path.join(outdir, f))
            rep.warnings.append(
                f"STEP 미생성 — {why}"
                + (f" (낡은 STEP {len(stale)} 개 삭제)" if stale else ""))

    with open(os.path.join(outdir, "export_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"design": rep.design, "fit": rep.fit, "fidelity": rep.fidelity,
                   "propellant": rep.propellant,
                   "parts": rep.parts, "warnings": rep.warnings}, fh,
                  ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "export_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep.summary() + "\n")
    return rep
