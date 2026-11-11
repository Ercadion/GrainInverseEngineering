#!/usr/bin/env python3
"""
t2_input_web.py
══════════════════════════════════════════════════════════════════════════
V2 그레인 탈형 관찰 — T2 (가시·살 생존) 판독기  v1.0

끝면에서 잰 가시 끝 반경 `r_break` 를 넣으면 판독표로 `t_prop_min` 을 환산하고,
탈형 관찰의 기록·판정·도구 반영 규칙을 그대로 적용해
화면에 보여 준다. 그래프도 같이 그린다.

  ▸ 입력은 **웹에서** 받는다 (rocket_input_builder.py 의 TMS 그래프 웹과 같은 방식)
  ▸ 판독표는 `out/mold_demold_D40/spike_table_D40.csv` 를 읽는다 — 여기서
    형상을 다시 계산하지 않는다. 표를 만든 것은 `experiments/v2_spike_table.py`

사용:
  python t2_input_web.py                 웹 UI 실행 (기본 포트 5091)
  python t2_input_web.py --port 5095     포트 지정
  python t2_input_web.py --no-browser    브라우저 자동 열기 끄기
  python t2_input_web.py --selftest      서버 없이 계산 검증만 (의존성 적음)

⚠ 이 도구는 **재지 않는다.** 재는 것은 사람이고, 이 도구는 잰 값을 규칙대로
  환산할 뿐이다. 그래서 규칙이 "못 쟀다" 라고 말하는 자리에서는 숫자를 내지
  않고 못 쟀다고 말한다 (판정 규칙 미충전·판별불가 갈래).
══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE / "out" / "mold_demold_D40" / "spike_table_D40.csv"

# 포트 최소반경 기본값 — Ø40 · r0 10 · a6 5 형상에서 가시가 온전했을 때의 끝 반경
DEFAULT_R_PORT_MIN = 5.00
DEFAULT_TOL = 0.10          # 사진 ↔ 캘리퍼스 교차검증 차이 (측정 불확실도)
N_LOBE = 6
MAX_SEG = 4                 # 끝면 8 개까지. 그 이상은 색으로 구분이 안 된다
T_PROVISIONAL = 2.0         # 현재 임시값 (constraints.py)

TEX_BREAK = "파단"
TEX_UNFILL = "미충전"
TEX_UNKNOWN = "판별불가"
TEX_INTACT = "온전"
TEXTURES = [TEX_BREAK, TEX_UNFILL, TEX_UNKNOWN, TEX_INTACT]

GRIDS = [("n=121 (운용)", 0.5), ("n=241", 0.25)]

# 끝면 색 — 고정 순서로만 쓴다. 돌려쓰지 않는다 (8 개를 넘으면 입력에서 막는다).
FACE_COLORS = ["#0f62fe", "#ff832b", "#198038", "#a56eff",
               "#da1e28", "#007d79", "#8a3ffc", "#d02670"]
FACE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ══════════════════════════════════════════════════════════════════════════
#  판독표
# ══════════════════════════════════════════════════════════════════════════

def load_table(path) -> list[tuple[float, float]]:
    """CSV → [(t_prop_min, r_break)] — r_break 오름차순, 단조 확인까지."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            rows.append((float(rec["t_prop_min_mm"]), float(rec["r_break_mm"])))
    if len(rows) < 2:
        raise ValueError(f"판독표에 점이 {len(rows)} 개뿐이다: {path}")
    rows.sort(key=lambda p: p[1])
    for (t0, r0), (t1, r1) in zip(rows, rows[1:]):
        if not (r1 > r0 and t1 > t0):
            raise ValueError(
                f"판독표가 단조가 아니다: t {t0}→{t1}, r {r0}→{r1}. "
                f"단조가 아니면 역환산이 한 값으로 정해지지 않는다")
    return rows


def lookup(rows, r: float) -> dict:
    """r_break → t_prop_min. 표 **범위 밖이면 외삽하지 않는다.**

    ⚠ 외삽을 막는 이유: 판독표는 형상에서 나온 곡선이지 직선이 아니다. 표
      바깥에서 직선으로 늘이면 그 값은 형상이 아니라 마지막 두 점의 기울기가
      말하는 값이 된다 — 재지 않은 값을 잰 것처럼 보이게 만든다.
    """
    lo_t, lo_r = rows[0]
    hi_t, hi_r = rows[-1]
    if r < lo_r:
        return {"status": "below", "t": None, "lo": lo_t, "lo_r": lo_r,
                "hi_r": hi_r}
    if r > hi_r:
        return {"status": "above", "t": None, "hi": hi_t, "lo_r": lo_r,
                "hi_r": hi_r}
    for (t0, r0), (t1, r1) in zip(rows, rows[1:]):
        if r0 <= r <= r1:
            t = t0 + (t1 - t0) * (r - r0) / (r1 - r0)
            return {"status": "ok", "t": t, "lo_r": lo_r, "hi_r": hi_r,
                    "seg": (t0, r0, t1, r1)}
    return {"status": "above", "t": None, "hi": hi_t,
            "lo_r": lo_r, "hi_r": hi_r}       # 도달 불가 (방어)


def quantize(t: float, h: float) -> tuple[int, float]:
    """`_thin_area` 구조요소의 셀 올림. k = 2⌈(t/2)/h⌉+1, 실효 = (k−1)·h."""
    k = 2 * math.ceil((t / 2.0) / h) + 1
    return k, (k - 1) * h


# ══════════════════════════════════════════════════════════════════════════
#  판정 — 기록 규칙 · 판정 규칙
# ══════════════════════════════════════════════════════════════════════════

def read_t2(spikes, rows, r_port_min=DEFAULT_R_PORT_MIN, tol=DEFAULT_TOL):
    """spikes = [{seg, face, idx, r, tex}] → 판독·판정 전부.

    기록 규칙: 평균 내지 않는다. 판정에는 **가장 보수적인 값**(가장 큰 r_break).
    판정 규칙  : 그 값의 **질감**이 갈래를 고른다.
    """
    out = {"warnings": [], "notes": []}
    vals = [s["r"] for s in spikes]
    if not vals:
        return {"ok": False, "error": "입력된 가시가 없다"}

    r_max, r_min = max(vals), min(vals)
    srt = sorted(vals)
    n = len(srt)
    median = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    spread = r_max - r_min
    cons = max(spikes, key=lambda s: s["r"])          # 보수값을 낸 가시

    out["summary"] = {
        "n": n, "r_max": r_max, "r_min": r_min, "median": median,
        "spread": spread, "cons_label": _label(cons),
        "cons_tex": cons["tex"],
    }

    # ── 물리적으로 불가능한 입력 먼저 막는다 ────────────────────────────
    if r_min < r_port_min - tol:
        out["warnings"].append(
            f"r_break 최솟값 {r_min:.2f} 가 포트 최소반경 {r_port_min:.2f} 보다 "
            f"작다. 가시는 포트 안쪽으로 더 들어갈 수 없으므로 중심 잡기나 "
            f"스케일이 틀렸을 가능성이 크다 (측정 요령 ⚠ 중심 잡기).")

    # ── 판정 규칙 갈래 고르기 ───────────────────────────────────────────────
    intact_all = all(s["r"] <= r_port_min + tol for s in spikes)
    if intact_all:
        out["verdict"] = {
            "branch": "온전",
            "measured": False,
            "t": None,
            "headline": f"가시가 전부 온전하다 → t_prop_min ≤ 판독표 최솟값 "
                        f"{rows[0][0]:.2f} mm",
            "detail": "6-로브 깊은 형상이 살아난다. 다만 이 시험은 "
                      "상한만 말한다 — 정확한 값은 더 얇은 판독표가 필요하다.",
            "kind": "good",
        }
        out["readout"] = []
        out["quant"] = []
        _pilot_note(out, broke=False)
        return {"ok": True, **out}

    lk = lookup(rows, r_max)
    if lk["status"] == "below":
        out["verdict"] = {
            "branch": "표 범위 아래",
            "measured": False, "t": None,
            "headline": f"보수값 {r_max:.2f} 가 판독표 최솟값 "
                        f"{lk['lo_r']:.4f} 보다 작다 — 환산할 수 없다",
            "detail": "가시가 거의 온전한 쪽이다. 더 얇은 쪽으로 판독표를 "
                      "넓혀 다시 만들어야 한다 (experiments/v2_spike_table.py).",
            "kind": "warn",
        }
        out["readout"] = []
        out["quant"] = []
        return {"ok": True, **out}
    if lk["status"] == "above":
        out["verdict"] = {
            "branch": "표 범위 위",
            "measured": False, "t": None,
            "headline": f"보수값 {r_max:.2f} 가 판독표 최댓값 "
                        f"{lk['hi_r']:.4f} 를 넘는다 — 환산할 수 없다",
            "detail": f"가시가 {lk['hi']:.2f} mm 두께분보다 더 깊이 잘렸다. "
                      f"외삽하지 않는다 — 판독표를 더 두꺼운 쪽으로 넓혀 다시 "
                      f"만들어야 한다.",
            "kind": "err",
        }
        out["readout"] = []
        out["quant"] = []
        return {"ok": True, **out}

    t_cons = lk["t"]

    # 보수값을 낸 가시의 질감이 판정을 지배한다
    if cons["tex"] == TEX_BREAK:
        out["verdict"] = {
            "branch": "파단",
            "measured": True, "t": t_cons,
            "headline": f"t_prop_min = {t_cons:.2f} mm — 이것은 측정값이다",
            "detail": f"보수값을 낸 가시 {_label(cons)} 가 파단으로 판별됐다 "
                      f"(판정 규칙 두 번째 갈래).",
            "kind": "good",
        }
    else:
        why = ("미충전" if cons["tex"] == TEX_UNFILL
               else "판별불가 → 판정 규칙 에 따라 미충전으로 보수적 판정")
        out["verdict"] = {
            "branch": cons["tex"],
            "measured": False, "t": None,
            "headline": "t_prop_min 은 못 쟀다 — 충전 조건을 바꿔 재시도",
            "detail": f"보수값을 낸 가시 {_label(cons)} 가 {why}. 끝이 잘린 "
                      f"이유가 탈형이 아니라 충전이면 그 반경은 강도에 대해 "
                      f"아무 말도 하지 않는다. 임시값 "
                      f"{T_PROVISIONAL:.1f} mm 를 유지한다.",
            "kind": "err",
        }
        out["notes"].append(
            f"참고로 그 반경을 그대로 환산하면 {t_cons:.2f} mm 지만, **쓰면 "
            f"안 되는 숫자다.** 충전 한계를 강도로 잘못 읽는 것이 된다.")

    # ── 판독값들 (참고용 포함) ────────────────────────────────────────
    rd = []
    for lab, r, use in (("보수값 (판정에 쓴다)", r_max, True),
                        ("중앙값 (참고)", median, False),
                        ("최솟값 (참고)", r_min, False)):
        x = lookup(rows, r)
        rd.append({"label": lab, "r": r,
                   "t": x["t"], "status": x["status"], "use": use})
    if not out["verdict"]["measured"]:
        # ⚠ 판정이 "못 쟀다" 면 보수값도 **쓰는 값이 아니다.** 표에서까지
        #   "판정에 쓴다" 로 보이면 경고를 읽기 전에 숫자를 베껴 간다.
        rd[0]["label"] = "보수값 (판정에는 쓰지 않는다)"
        rd[0]["use"] = False
    out["readout"] = rd

    # ── 도구 반영 규칙 격자 양자화 ───────────────────────────────────────────────
    q = []
    for name, h in GRIDS:
        k, eff = quantize(t_cons, h)
        q.append({"grid": name, "h": h, "k": k, "eff": eff,
                  "delta": eff - t_cons,
                  "pct": 100.0 * (eff - t_cons) / t_cons if t_cons else 0.0})
    out["quant"] = q
    out["quant_valid"] = out["verdict"]["measured"]

    # 중앙값을 썼다면 어디로 가는지 — 선택의 폭을 보여 준다
    if rd[1]["t"] is not None:
        tm = rd[1]["t"]
        out["quant_median"] = [
            {"grid": nm, "eff": quantize(tm, h)[1]} for nm, h in GRIDS]
        out["summary"]["t_median"] = tm

    # ── 경고 ────────────────────────────────────────────────────────
    if spread > 0.5:
        out["warnings"].append(
            f"흩어짐 {spread:.2f} mm 가 기록 규칙 보고 문턱 0.5 mm 를 넘는다 — "
            f"결과와 **함께 보고해야 한다.** 측정 불확실도 {tol:.2f} mm 의 "
            f"{spread/tol:.0f} 배라면 가시마다 실제로 다른 반경에서 끝났다는 "
            f"뜻이고, 단일 문턱이 아니라 분포다.")
    bad = [s for s in spikes if s["tex"] in (TEX_UNFILL, TEX_UNKNOWN)]
    if bad and cons["tex"] == TEX_BREAK:
        out["warnings"].append(
            f"보수값은 파단이라 판정은 서지만, {len(bad)} 곳이 "
            f"{'/'.join(sorted({s['tex'] for s in bad}))} 다 "
            f"({', '.join(_label(s) for s in bad)}). 충전 조건을 함께 "
            f"기록할 것 — 이 갈래가 최댓값으로 올라오면 판정이 뒤집힌다.")
    mixed = [s for s in spikes if s["tex"] == TEX_INTACT]
    if mixed and not intact_all:
        out["warnings"].append(
            f"{len(mixed)} 곳만 온전하다 — 온전과 잘림이 섞였다. 국소 차이가 "
            f"큰 것이므로 각도·충전 쪽을 의심할 것 (도구 반영 규칙).")

    _pilot_note(out, broke=(out["verdict"]["branch"] == "파단"))
    return {"ok": True, **out}


def _label(s) -> str:
    return f"S-{s['seg']}{s['face']} #{s['idx']}"


def _pilot_note(out, broke: bool):
    """18 mm 시편 0.26 배 단서 — 결론의 강도가 **방향에 따라 다르다.**"""
    if broke:
        out["notes"].append(
            "18 mm 시편은 70 mm 그레인 대비 가시 뿌리 전단력이 **0.26 배**다. "
            "그 무른 조건에서도 부러졌다면 더 가혹한 실물 길이에서는 확실히 "
            "부러진다 — **이 결과는 그대로 전이된다.**")
    else:
        out["notes"].append(
            "⚠ 18 mm 시편은 70 mm 그레인 대비 가시 뿌리 전단력이 **0.26 배**다. "
            "살아남았다는 결과는 **실물로 전이되지 않는다** — 긴 시편으로 다시 "
            "확인해야 한다.")


# ══════════════════════════════════════════════════════════════════════════
#  그래프
# ══════════════════════════════════════════════════════════════════════════

def _pick_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "Source Han Sans K",
                 "NanumBarunGothic"):
        if cand in have:
            return cand
    return None


def make_charts(spikes, rows, res, r_port_min, n_lobe):
    """2 패널 PNG (base64). 한글 폰트가 없으면 그 사실을 돌려준다."""
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _pick_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    faces = []
    for s in spikes:
        key = (s["seg"], s["face"])
        if key not in faces:
            faces.append(key)
    faces.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=130)

    # ── ① 가시별 r_break ──────────────────────────────────────────────
    for i, (seg, face) in enumerate(faces):
        pts = sorted((s for s in spikes if (s["seg"], s["face"]) == (seg, face)),
                     key=lambda s: s["idx"])
        ax1.plot([p["idx"] for p in pts], [p["r"] for p in pts],
                 color=FACE_COLORS[i % len(FACE_COLORS)],
                 marker=FACE_MARKERS[i % len(FACE_MARKERS)],
                 ms=8, lw=1.8, label=f"S-{seg}{face}")
    r_max = res["summary"]["r_max"]
    ax1.axhline(r_max, ls="--", lw=1.6, color="#a2191f")
    ax1.axhline(r_port_min, ls=":", lw=1.4, color="#8d8d8d")
    lo = min(r_port_min, res["summary"]["r_min"]) - 0.25
    hi = max(r_max, r_port_min) + 0.45
    ax1.set_ylim(lo, hi)
    ax1.set_xlim(0.5, n_lobe + 0.5)
    ax1.set_xticks(range(1, n_lobe + 1))
    used = res["verdict"].get("measured", False)
    ax1.text(0.6, r_max + 0.06,
             f"보수값 {r_max:.2f}" + (" (판정에 쓴다)" if used
                                     else " (판정에는 쓰지 않는다)"),
             color="#a2191f", fontsize=10)
    ax1.text(0.6, r_port_min + 0.06, f"포트 최소반경 {r_port_min:.2f}"
                                     f" — 가시가 온전했다면 여기",
             color="#525252", fontsize=9)
    ax1.set_xlabel("가시 번호 (각도 기준 구멍이 원점)")
    ax1.set_ylabel("r_break [mm]")
    ax1.set_title(f"① 가시 {len(spikes)} 곳의 끝 반경"
                  f" — 흩어짐 {res['summary']['spread']:.2f} mm", fontsize=12)
    ax1.grid(alpha=.25, lw=.6)
    ax1.legend(fontsize=9, loc="lower right", ncol=min(len(faces), 4))

    # ── ② 판독표 환산 ────────────────────────────────────────────────
    tt = [p[0] for p in rows]
    rr = [p[1] for p in rows]
    ax2.plot(rr, tt, color="#393939", lw=2.0, label="판독표", zorder=2)
    ok = [s for s in spikes if rows[0][1] <= s["r"] <= rows[-1][1]]
    if ok:
        ax2.scatter([s["r"] for s in ok],
                    [lookup(rows, s["r"])["t"] for s in ok],
                    s=46, color="#4589ff", zorder=3, label="측정 환산")
    vt = res["verdict"].get("t")
    if vt is not None:
        ax2.scatter([r_max], [vt], marker="D", s=135, color="#da1e28",
                    zorder=4, label=f"보수값 {r_max:.2f} → {vt:.2f} mm")
        for i, q in enumerate(res.get("quant", [])):
            c = ["#6929c4", "#007d79"][i % 2]
            ax2.axhline(q["eff"], ls="--", lw=1.3, color=c)
            ax2.text(rows[0][1] + 0.06, q["eff"] + 0.04,
                     f"{q['grid']} 에서는 {q['eff']:.2f} 로 동작",
                     color=c, fontsize=9)
    ax2.set_xlabel("r_break [mm]")
    ax2.set_ylabel("t_prop_min [mm]")
    ax2.set_title("② 판독표로 환산 — 그리고 도구 격자에 걸리는 자리",
                  fontsize=12)
    ax2.grid(alpha=.25, lw=.6)
    ax2.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return (base64.b64encode(buf.getvalue()).decode(), font)


# ══════════════════════════════════════════════════════════════════════════
#  요청 처리
# ══════════════════════════════════════════════════════════════════════════

def handle_read(payload: dict) -> dict:
    try:
        tbl_path = payload.get("table") or str(DEFAULT_TABLE)
        rows = load_table(tbl_path)
    except FileNotFoundError:
        return {"ok": False, "error":
                f"판독표를 못 찾았다: {tbl_path}\n"
                f"experiments/v2_spike_table.py 로 먼저 만들어야 한다."}
    except Exception as exc:
        return {"ok": False, "error": f"판독표를 읽지 못했다: {exc}"}

    r_port_min = float(payload.get("r_port_min") or DEFAULT_R_PORT_MIN)
    tol = float(payload.get("tol") or DEFAULT_TOL)
    n_lobe = int(payload.get("n_lobe") or N_LOBE)

    spikes, blank = [], 0
    for raw in payload.get("spikes", []):
        txt = str(raw.get("r", "")).strip()
        if txt == "":
            blank += 1
            continue
        try:
            r = float(txt)
        except ValueError:
            return {"ok": False, "error":
                    f"S-{raw.get('seg')}{raw.get('face')} #{raw.get('idx')} 의 "
                    f"r_break 가 숫자가 아니다: {txt!r}"}
        tex = raw.get("tex") or TEX_BREAK
        if tex not in TEXTURES:
            return {"ok": False, "error": f"모르는 질감: {tex!r}"}
        spikes.append({"seg": int(raw["seg"]), "face": raw["face"],
                       "idx": int(raw["idx"]), "r": r, "tex": tex})
    if not spikes:
        return {"ok": False, "error":
                "r_break 를 하나도 넣지 않았다. 적어도 한 곳은 넣어야 한다."}

    res = read_t2(spikes, rows, r_port_min, tol)
    if not res.get("ok"):
        return res

    if blank:
        res.setdefault("warnings", []).insert(0,
            f"{blank} 곳이 비어 있다 — {len(spikes)} 곳만으로 판독했다. "
            f"측정 요령 는 끝면마다 가시 전부를 보라고 한다. 비운 자리가 "
            f"'못 봤다' 인지 '없다' 인지 기록에 남길 것.")

    try:
        png, font = make_charts(spikes, rows, res, r_port_min, n_lobe)
        res["chart"] = png
        if font is None:
            res.setdefault("warnings", []).append(
                "한글 폰트를 못 찾아 그림의 글자가 깨진다. 판독 숫자는 "
                "영향이 없다. (Windows: 맑은 고딕이 있으면 자동으로 쓴다)")
    except Exception as exc:
        res["chart"] = None
        res.setdefault("warnings", []).append(
            f"그림을 그리지 못했다: {exc.__class__.__name__}: {exc}. "
            f"판독 숫자는 그대로다.")

    res["table_info"] = {
        "path": str(tbl_path), "n": len(rows),
        "r_lo": rows[0][1], "r_hi": rows[-1][1],
        "t_lo": rows[0][0], "t_hi": rows[-1][0]}
    res["provisional"] = T_PROVISIONAL
    return res


# ══════════════════════════════════════════════════════════════════════════
#  페이지
# ══════════════════════════════════════════════════════════════════════════

PAGE = r"""<!doctype html>
<html lang="ko" data-theme="g100">
<head>
<meta charset="utf-8">
<title>T2 판독기 — 가시·살 생존</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root, [data-theme="g100"]{
  --bg:#161616; --layer:#262626; --layer2:#393939; --border:#525252;
  --text:#f4f4f4; --text2:#c6c6c6; --accent:#78a9ff; --blue:#0f62fe;
  --warn:#f1c21b; --err:#fa4d56; --ok:#42be65;
}
[data-theme="g10"]{
  --bg:#f4f4f4; --layer:#ffffff; --layer2:#e0e0e0; --border:#8d8d8d;
  --text:#161616; --text2:#525252; --accent:#0f62fe; --ok:#198038;
}
*{box-sizing:border-box; margin:0; padding:0}
body{background:var(--bg); color:var(--text);
     font-family:'IBM Plex Sans','Malgun Gothic',sans-serif; font-size:14px}
header{display:flex; align-items:center; height:48px; padding:0 1rem;
       border-bottom:1px solid var(--layer2); position:sticky; top:0;
       background:var(--bg); z-index:5}
header .name{font-weight:600}
header .name span{font-weight:300; color:var(--accent)}
header .sp{flex:1}
header button{background:none; border:none; color:var(--text);
              cursor:pointer; font-size:16px}
main{max-width:1320px; margin:0 auto; padding:1.5rem 1rem 4rem}
h1{font-size:22px; font-weight:400; margin-bottom:.25rem}
.sub{color:var(--text2); font-size:13px; margin-bottom:1.25rem;
     line-height:1.6; max-width:900px}
.grid{display:grid; grid-template-columns:minmax(360px,440px) minmax(0,1fr);
      gap:1.25rem; align-items:start}
@media(max-width:980px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--layer); border:1px solid var(--layer2);
      padding:1rem; min-width:0; overflow:hidden}
.card h2{font-size:14px; font-weight:600; margin-bottom:.75rem;
         color:var(--accent)}
fieldset{border:none; margin-bottom:1.1rem; min-width:0}
legend{font-size:11px; color:var(--text2); margin-bottom:.5rem;
       text-transform:uppercase; letter-spacing:.05em; font-weight:500}
.row{display:flex; gap:.5rem; margin-bottom:.5rem;
     align-items:center; min-width:0}
.row label{flex:1.5 1 0; min-width:0; font-size:12px; color:var(--text2);
           display:flex; align-items:center; line-height:1.35;
           overflow-wrap:anywhere}
.row input,.row select{flex:1 1 0; min-width:0; width:100%; max-width:100%;
    background:var(--bg); color:var(--text);
    border:none; border-bottom:1px solid var(--border);
    padding:.4rem .5rem; font-size:13px; font-family:inherit}
.row select{appearance:none; -webkit-appearance:none; cursor:pointer}
.row input:focus,.row select:focus{outline:2px solid var(--blue);
    outline-offset:-1px}
.btn{width:100%; background:var(--blue); color:#fff; border:none;
     padding:.8rem; font-size:14px; cursor:pointer; margin-top:.5rem;
     font-family:inherit}
.btn:disabled{background:var(--layer2); color:var(--text2); cursor:wait}
.mini{background:var(--layer2); color:var(--text); border:none;
      padding:.4rem .7rem; font-size:12px; cursor:pointer;
      font-family:inherit; flex:0 0 auto; white-space:nowrap}
.mini:hover{background:var(--border)}
.row.btns{gap:.35rem; flex-wrap:wrap; margin-top:.6rem}
#status{font-size:12px; color:var(--accent); margin-top:.6rem; min-height:16px}
.note{font-size:11px; color:var(--text2); margin-top:.6rem; line-height:1.6}
.sub-block{background:var(--bg); border-left:2px solid var(--layer2);
    padding:.6rem .6rem .35rem; margin:.15rem 0 .7rem}
.sub-hint{font-size:11px; color:var(--text2); line-height:1.5;
    margin:.1rem 0 .45rem}
.face-h{font-size:12px; color:var(--accent); margin:.5rem 0 .35rem;
        font-weight:600}
.sp-row{display:grid; grid-template-columns:36px 1fr 104px; gap:.35rem;
        margin-bottom:.3rem; align-items:center}
.sp-row .no{font-size:11px; color:var(--text2); text-align:right;
            font-family:'IBM Plex Mono',monospace}
.sp-row input,.sp-row select{background:var(--bg); color:var(--text);
    border:none; border-bottom:1px solid var(--border);
    padding:.3rem .4rem; font-size:12px; font-family:inherit;
    min-width:0; width:100%}
.sp-row input:focus,.sp-row select:focus{outline:2px solid var(--blue);
    outline-offset:-1px}
table.sum{width:100%; border-collapse:collapse; font-size:13px}
table.sum td{padding:.35rem .5rem; border-bottom:1px solid var(--layer2)}
table.sum td:first-child{color:var(--text2)}
table.sum td:last-child{text-align:right;
    font-family:'IBM Plex Mono',monospace}
table.iter{width:100%; border-collapse:collapse; font-size:12px;
    margin-top:.5rem; font-family:'IBM Plex Mono',monospace}
table.iter th{padding:.35rem .45rem; text-align:right; color:var(--text2);
    font-weight:500; border-bottom:1px solid var(--layer2); font-size:11px}
table.iter td{padding:.35rem .45rem; text-align:right;
    border-bottom:1px solid var(--layer2)}
table.iter tr.final td{background:rgba(15,98,254,.1); font-weight:600}
table.iter td.l,table.iter th.l{text-align:left}
.verdict{padding:.85rem 1rem; margin-bottom:1rem; font-size:14px;
         line-height:1.6; border-left:4px solid var(--border);
         background:var(--bg)}
.verdict.good{border-left-color:var(--ok)}
.verdict.warn{border-left-color:var(--warn)}
.verdict.err{border-left-color:var(--err)}
.verdict .hl{font-size:16px; font-weight:600; display:block;
             margin-bottom:.3rem}
.verdict .dt{color:var(--text2); font-size:12.5px}
.warn{background:rgba(241,194,27,.08); border-left:3px solid var(--warn);
      padding:.55rem .75rem; font-size:12px; margin-top:.5rem;
      line-height:1.6}
.notebox{background:rgba(120,169,255,.07); border-left:3px solid var(--accent);
      padding:.55rem .75rem; font-size:12px; margin-top:.5rem;
      line-height:1.6}
.errbox{background:rgba(250,77,86,.1); border-left:3px solid var(--err);
      padding:.6rem .8rem; font-size:13px; margin-top:.5rem;
      line-height:1.6; white-space:pre-wrap}
.chart{width:100%; background:#fff; border:1px solid var(--layer2);
       margin-top:1rem}
.empty{color:var(--text2); padding:3rem 1rem; text-align:center;
       border:1px dashed var(--layer2); line-height:1.7}
.cap{font-size:11px; color:var(--text2); margin-top:.6rem; line-height:1.6}
h3.sec{font-size:12px; color:var(--text2); text-transform:uppercase;
       letter-spacing:.05em; margin:1.2rem 0 .4rem; font-weight:500}
</style>
</head>
<body>
<header>
  <div class="name">V2 탈형 관찰 <span>· T2 판독기</span></div>
  <div class="sp"></div>
  <button id="theme" title="테마">◐</button>
</header>
<main>
  <h1>T2 — 가시·살 생존 <span style="color:var(--accent)">판독</span></h1>
  <p class="sub">끝면에서 잰 가시 끝 반경 <code>r_break</code> 를 넣으면
    판독표로 <code>t_prop_min</code> 을 환산하고, 기록(보수값·흩어짐)·
    판정·격자 양자화 규칙을 그대로 적용합니다.
    <strong>이 도구는 재지 않습니다</strong> — 규칙이 "못 쟀다"고 말하는
    자리에서는 숫자를 내지 않습니다.</p>

  <div class="grid">
    <div class="card">
      <h2>입력</h2>
      <fieldset>
        <legend>형상 · 판독표</legend>
        <div class="row"><label>판독표 CSV</label>
          <input id="table" placeholder="(기본값 사용)"></div>
        <div class="row"><label>포트 최소반경 [mm]<br>
          <span style="font-size:10px">가시가 온전했을 때의 끝 반경</span></label>
          <input id="rport" value="5.00"></div>
        <div class="row"><label>측정 불확실도 [mm]<br>
          <span style="font-size:10px">사진 ↔ 캘리퍼스 교차검증 차이</span></label>
          <input id="tol" value="0.10"></div>
        <div class="row"><label>로브 수 (끝면당 가시 수)</label>
          <input id="nlobe" value="6"></div>
        <div class="row"><label>세그먼트 수</label>
          <select id="nseg">
            <option value="1">1 (끝면 2)</option>
            <option value="2">2 (끝면 4)</option>
            <option value="3">3 (끝면 6)</option>
            <option value="4">4 (끝면 8)</option>
          </select></div>
      </fieldset>

      <fieldset>
        <legend>S 열 — 가시별 끝 반경</legend>
        <div class="sub-hint">평균 내지 말고 전부 적습니다. 못 본 자리는
          비워 두면 그 수를 경고로 돌려줍니다.</div>
        <div id="spikes"></div>
      </fieldset>

      <button class="btn" id="go">판독</button>
      <div class="row btns">
        <button type="button" class="mini" id="demo">가정 시나리오 채우기</button>
        <button type="button" class="mini" id="clear">비우기</button>
      </div>
      <div id="status"></div>
      <div class="note">질감 판별이 <b>판정을 지배</b>합니다. 가장 큰
        <code>r_break</code> 를 낸 가시가 미충전/판별불가면 판정 규칙 에 따라
        <code>t_prop_min</code> 은 못 잰 것이 되고, 그때 이 도구는 숫자를
        내지 않습니다.</div>
    </div>

    <div class="card" id="out">
      <h2>판독 결과</h2>
      <div class="empty">왼쪽에 <code>r_break</code> 를 넣고 [판독] 을 누르면
        여기에 판정·환산·격자 양자화와 그림이 나옵니다.<br>
        처음이면 [가정 시나리오 채우기] 로 동작을 먼저 보세요.</div>
    </div>
  </div>
</main>

<script>
const FACES = ['상','하'];
const TEX = ['파단','미충전','판별불가','온전'];
// 가정 시나리오 (실측 아님)
const DEMO = [6.41,6.58,6.33,6.72,6.49,6.55, 6.28,6.47,6.90,6.36,6.61,6.44];
const DEMO_TEX = ['파단','파단','파단','파단','파단','파단',
                  '파단','파단','파단','파단','판별불가','파단'];

function buildRows(){
  const nseg = +document.getElementById('nseg').value;
  const nlobe = Math.max(1, Math.min(12, +document.getElementById('nlobe').value||6));
  const old = {};
  document.querySelectorAll('#spikes .sp-row').forEach(r=>{
    old[r.dataset.key] = [r.querySelector('input').value,
                          r.querySelector('select').value];
  });
  let h = '';
  for(let s=1; s<=nseg; s++){
    for(const f of FACES){
      h += `<div class="face-h">S-${s}${f} — 세그먼트 ${s} · 끝면 ${f}</div>`;
      for(let i=1; i<=nlobe; i++){
        const key = `${s}|${f}|${i}`;
        const prev = old[key] || ['','파단'];
        h += `<div class="sp-row" data-key="${key}" data-seg="${s}"
                   data-face="${f}" data-idx="${i}">
          <span class="no">#${i}</span>
          <input inputmode="decimal" placeholder="r_break [mm]"
                 value="${prev[0]}">
          <select>${TEX.map(t=>`<option${t===prev[1]?' selected':''}>${t}</option>`).join('')}</select>
        </div>`;
      }
    }
  }
  document.getElementById('spikes').innerHTML = h;
}

function collect(){
  const out = [];
  document.querySelectorAll('#spikes .sp-row').forEach(r=>{
    out.push({seg:+r.dataset.seg, face:r.dataset.face, idx:+r.dataset.idx,
              r:r.querySelector('input').value,
              tex:r.querySelector('select').value});
  });
  return out;
}

function esc(s){return String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));}
function md(s){return esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');}
function f2(x){return (x===null||x===undefined)?'—':(+x).toFixed(2);}

function render(d){
  const o = document.getElementById('out');
  if(!d.ok){
    o.innerHTML = '<h2>판독 결과</h2><div class="errbox">'+esc(d.error)+'</div>';
    return;
  }
  const v = d.verdict, s = d.summary, ti = d.table_info;
  let h = '<h2>판독 결과</h2>';
  h += `<div class="verdict ${v.kind}"><span class="hl">${md(v.headline)}</span>
        <span class="dt">판정 규칙 갈래: <b>${esc(v.branch)}</b> · ${md(v.detail)}</span></div>`;

  h += '<h3 class="sec">측정 요약 (기록 규칙 — 평균 내지 않는다)</h3><table class="sum">';
  h += `<tr><td>가시 수</td><td>${s.n} 곳</td></tr>`;
  h += `<tr><td>보수값 (최대 r_break)</td><td>${f2(s.r_max)} mm — ${esc(s.cons_label)} · ${esc(s.cons_tex)}</td></tr>`;
  h += `<tr><td>중앙값</td><td>${f2(s.median)} mm</td></tr>`;
  h += `<tr><td>최솟값</td><td>${f2(s.r_min)} mm</td></tr>`;
  h += `<tr><td>흩어짐</td><td>${f2(s.spread)} mm${s.spread>0.5?' ⚠':''}</td></tr>`;
  h += '</table>';

  if(d.readout && d.readout.length){
    h += '<h3 class="sec">판독표 환산</h3><table class="iter"><tr>'
      + '<th class="l">입력</th><th>r_break [mm]</th><th>t_prop_min [mm]</th></tr>';
    for(const r of d.readout){
      const t = r.status==='ok' ? f2(r.t) : '범위 밖';
      h += `<tr class="${r.use?'final':''}"><td class="l">${esc(r.label)}</td>`
         + `<td>${f2(r.r)}</td><td>${t}</td></tr>`;
    }
    h += '</table>';
  }

  if(d.quant && d.quant.length){
    h += '<h3 class="sec">도구 반영 규칙 격자 양자화 — 실제로 작동하는 값</h3>';
    if(!d.quant_valid){
      h += '<div class="warn">아래 표는 <b>판정이 "못 쟀다" 이므로 쓰면 '
         + '안 되는 값</b>이다. 보수값을 그대로 넣었을 때 어디에 걸리는지만 '
         + '보여 준다.</div>';
    }
    h += '<table class="iter"><tr><th class="l">격자</th><th>h</th><th>k</th>'
      + '<th>실효 t_prop_min</th><th>입력 대비</th></tr>';
    for(const q of d.quant){
      h += `<tr><td class="l">${esc(q.grid)}</td><td>${q.h}</td><td>${q.k}</td>`
         + `<td>${f2(q.eff)} mm</td><td>+${f2(q.delta)} (${q.pct.toFixed(0)}%)</td></tr>`;
    }
    h += '</table>';
    if(d.quant_median){
      const m = d.quant_median.map(q=>`${esc(q.grid)} → ${f2(q.eff)}`).join(' · ');
      h += `<div class="cap">참고 — 중앙값 ${f2(s.t_median)} 를 넣으면: ${m}. `
         + `보수값이냐 중앙값이냐가 격자 선택보다 크게 가를 수 있다.</div>`;
    }
    h += '<div class="cap">k = 2⌈(t/2)/h⌉+1, 실효 = (k−1)·h. '
       + '<code>_thin_area</code> 구조요소가 셀 단위로 올림되기 때문이다.</div>';

    if(d.quant_valid && d.quant.length>=2){
      const a = d.quant[0], b = d.quant[1];
      const dp = (x)=> (x>0?'+':'') + x.toFixed(2);
      h += '<h3 class="sec">도구 반영 규칙 선택지 — 이 수치에서의 비용</h3>'
        + '<table class="iter"><tr><th class="l">선택</th>'
        + '<th class="l">결과</th><th class="l">비용</th></tr>'
        + `<tr><td class="l">보수적으로 올려 쓴다 (${esc(a.grid)})</td>`
        + `<td class="l">실효 ${f2(a.eff)} mm — 임시값 ${d.provisional.toFixed(1)} 대비 ${dp(a.eff-d.provisional)}</td>`
        + '<td class="l">0</td></tr>'
        + `<tr><td class="l">격자를 조인다 (${esc(b.grid)})</td>`
        + `<td class="l">실효 ${f2(b.eff)} mm — 올림 손실 ${f2(b.delta)} mm</td>`
        + '<td class="l">반복당 비용 ↑, n_samples 자동 상향 동반</td></tr>'
        + '<tr><td class="l">판정만 촘촘한 격자로</td>'
        + '<td class="l">최적화는 운용 격자, 최종 판정만 조인 격자</td>'
        + '<td class="l">구현 필요 — 지금은 없다</td></tr></table>'
        + '<div class="cap">이 선택은 <b>값을 본 뒤에</b> 한다. '
        + '미리 정하지 않는다 (도구 반영 규칙).</div>';
    }
  }

  for(const w of (d.warnings||[])) h += '<div class="warn">⚠ '+md(w)+'</div>';
  for(const n of (d.notes||[]))    h += '<div class="notebox">'+md(n)+'</div>';

  if(d.chart) h += `<img class="chart" alt="T2 판독 그래프" src="data:image/png;base64,${d.chart}">`;
  h += `<div class="cap">판독표: ${esc(ti.path)} · ${ti.n} 점 · `
     + `r_break ${ti.r_lo.toFixed(4)}~${ti.r_hi.toFixed(4)} mm `
     + `(t ${ti.t_lo.toFixed(2)}~${ti.t_hi.toFixed(2)} mm). `
     + `표 범위 밖은 <b>외삽하지 않는다.</b> 현재 임시값 ${d.provisional.toFixed(1)} mm.</div>`;
  o.innerHTML = h;
}

document.getElementById('go').onclick = async ()=>{
  const btn = document.getElementById('go'), st = document.getElementById('status');
  btn.disabled = true; st.textContent = '판독 중...';
  try{
    const res = await fetch('/api/t2/read', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        table: document.getElementById('table').value.trim(),
        r_port_min: document.getElementById('rport').value,
        tol: document.getElementById('tol').value,
        n_lobe: document.getElementById('nlobe').value,
        spikes: collect()})});
    const d = await res.json();
    render(d);
    st.textContent = d.ok ? '완료' : '입력을 확인하세요';
  }catch(e){ st.textContent = '요청 실패: ' + e; }
  btn.disabled = false;
};

document.getElementById('demo').onclick = ()=>{
  document.getElementById('nseg').value = '1';
  document.getElementById('nlobe').value = '6';
  buildRows();
  document.querySelectorAll('#spikes .sp-row').forEach((r,i)=>{
    if(i < DEMO.length){
      r.querySelector('input').value = DEMO[i];
      r.querySelector('select').value = DEMO_TEX[i];
    }
  });
  document.getElementById('status').textContent =
    '가정 시나리오를 넣었습니다 — 실측이 아닙니다.';
};
document.getElementById('clear').onclick = ()=>{
  document.querySelectorAll('#spikes .sp-row').forEach(r=>{
    r.querySelector('input').value = '';
    r.querySelector('select').value = '파단';
  });
  document.getElementById('status').textContent = '';
};
document.getElementById('nseg').onchange = buildRows;
document.getElementById('nlobe').onchange = buildRows;
document.getElementById('theme').onclick = ()=>{
  const e = document.documentElement;
  e.dataset.theme = e.dataset.theme === 'g100' ? 'g10' : 'g100';
};
buildRows();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════
#  자체 검증 — 서버 없이도 돈다
# ══════════════════════════════════════════════════════════════════════════

def selftest(table=None) -> bool:
    rows = load_table(table or DEFAULT_TABLE)
    ok = True

    def check(name, got, want, eps=1e-6):
        nonlocal ok
        good = abs(got - want) <= eps if isinstance(want, float) else got == want
        print(f"  {'✓' if good else '✗'} {name}: {got!r}"
              + ("" if good else f"  (기대 {want!r})"))
        ok = ok and good

    print("판독표 보간")
    check("표 위의 점은 그대로", lookup(rows, 6.6411)["t"], 2.000, 1e-3)
    check("6.90 → 2.3696", lookup(rows, 6.90)["t"], 2.3696285506, 1e-6)
    check("6.48 → 1.8512", lookup(rows, 6.48)["t"], 1.8512190617, 1e-6)
    check("표 아래는 외삽 안 함", lookup(rows, 4.0)["status"], "below")
    check("표 위도 외삽 안 함", lookup(rows, 9.0)["status"], "above")

    print("양자화  k = 2⌈(t/2)/h⌉+1")
    check("2.3696 @ h=0.5 → k", quantize(2.3696285506, 0.5)[0], 7)
    check("2.3696 @ h=0.5 → 실효", quantize(2.3696285506, 0.5)[1], 3.0, 1e-9)
    check("2.3696 @ h=0.25 → k", quantize(2.3696285506, 0.25)[0], 11)
    check("2.3696 @ h=0.25 → 실효", quantize(2.3696285506, 0.25)[1], 2.5, 1e-9)
    check("1.8512 @ h=0.5 → 실효", quantize(1.8512190617, 0.5)[1], 2.0, 1e-9)
    check("표 위의 값도 올림된다 (2.0 → 2.0)",
          quantize(2.0, 0.5)[1], 2.0, 1e-9)

    print("판정 판정 규칙")
    demo = [6.41, 6.58, 6.33, 6.72, 6.49, 6.55,
            6.28, 6.47, 6.90, 6.36, 6.61, 6.44]
    tex = ["파단"] * 10 + ["판별불가", "파단"]
    sp = []
    for i, (r, tx) in enumerate(zip(demo, tex)):
        sp.append({"seg": 1, "face": "상" if i < 6 else "하",
                   "idx": i % 6 + 1, "r": r, "tex": tx})
    res = read_t2(sp, rows)
    check("가정 시나리오 → 파단 갈래", res["verdict"]["branch"], "파단")
    check("측정값으로 선다", res["verdict"]["measured"], True)
    check("t = 2.37", round(res["verdict"]["t"], 2), 2.37)
    check("흩어짐 0.62", round(res["summary"]["spread"], 2), 0.62)
    check("흩어짐 경고가 뜬다",
          any("흩어짐" in w for w in res["warnings"]), True)
    check("판별불가 혼재 경고가 뜬다",
          any("판별불가" in w for w in res["warnings"]), True)

    # 최댓값이 판별불가면 판정이 뒤집힌다 — 이게 핵심 갈래다
    sp2 = [dict(s) for s in sp]
    sp2[8]["tex"] = "판별불가"                      # r=6.90 자리
    res2 = read_t2(sp2, rows)
    check("최댓값이 판별불가 → 못 쟀다",
          res2["verdict"]["measured"], False)
    check("그때 t 는 내지 않는다", res2["verdict"]["t"], None)

    sp3 = [dict(s, r=5.02, tex="온전") for s in sp]
    res3 = read_t2(sp3, rows)
    check("전부 온전 → 상한만", res3["verdict"]["branch"], "온전")
    check("온전이면 t 를 내지 않는다", res3["verdict"]["t"], None)
    check("온전이면 전이 경고가 붙는다",
          any("전이되지 않는다" in n for n in res3["notes"]), True)

    sp4 = [dict(s, r=9.5) for s in sp]
    res4 = read_t2(sp4, rows)
    check("표 범위 위 → 환산 거부", res4["verdict"]["branch"], "표 범위 위")

    print("요청 처리 (handle_read)")
    pay = {"spikes": [{"seg": 1, "face": "상", "idx": i + 1,
                       "r": demo[i], "tex": tex[i]} for i in range(12)]}
    hr = handle_read(pay)
    check("ok", hr["ok"], True)
    check("그림이 붙는다", hr.get("chart") is not None, True)
    bad = handle_read({"spikes": [{"seg": 1, "face": "상", "idx": 1,
                                   "r": "abc", "tex": "파단"}]})
    check("숫자 아닌 입력은 거부", bad["ok"], False)
    empty = handle_read({"spikes": []})
    check("빈 입력은 거부", empty["ok"], False)

    print("\n" + ("전부 통과" if ok else "실패 있음"))
    return ok


# ══════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="T2 (가시·살 생존) 판독 웹 UI",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--port", type=int, default=5091, help="포트 (기본 5091)")
    ap.add_argument("--no-browser", action="store_true",
                    help="브라우저 자동 열기 끄기")
    ap.add_argument("--table", default=None, help="판독표 CSV 경로")
    ap.add_argument("--selftest", action="store_true",
                    help="서버 없이 계산 검증만 하고 끝낸다")
    args = ap.parse_args()

    if args.selftest:
        return 0 if selftest(args.table) else 1

    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("\n❌ Flask 가 필요합니다:  pip install flask")
        return 1

    tbl = Path(args.table) if args.table else DEFAULT_TABLE
    if not tbl.exists():
        print(f"\n⚠ 판독표가 없습니다: {tbl}")
        print("  experiments/v2_spike_table.py 로 먼저 만들어야 합니다.")
        print("  (웹은 그대로 뜨지만 [판독] 을 누르면 그 사실을 알려 줍니다)")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/api/t2/read", methods=["POST"])
    def api_read():
        payload = request.get_json(force=True) or {}
        if not payload.get("table") and args.table:
            payload["table"] = args.table
        try:
            return jsonify(handle_read(payload))
        except Exception as exc:                   # pragma: no cover
            return jsonify({"ok": False,
                            "error": f"{exc.__class__.__name__}: {exc}"})

    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")
        ).start()

    print(f"\n🌐 T2 판독기 실행 중  →  http://127.0.0.1:{args.port}/")
    print(f"   판독표: {tbl}")
    print("   종료: Ctrl+C\n")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\n  서버 종료됨.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
