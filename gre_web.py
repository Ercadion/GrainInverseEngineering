#!/usr/bin/env python3
"""
gre_web.py
══════════════════════════════════════════════════════════════════════════
고체 로켓 그레인 형상 역설계 — 웹 입력·실행 UI  v1.0

목표 추력(·압력) 곡선과 **케이스·노즐목**을 넣으면 포트 형상을 역설계하고,
합격 기준(추력·압력 **10 % 이내**)으로 판정한다.

  ▸ 입력은 웹에서 받는다 (rocket_input_builder.py 의 TMS 그래프 웹과 같은 방식:
    Flask + 임베드 HTML + matplotlib base64)
  ▸ 실행 모드를 **화면에서 고른다**
        [단일]      씨앗 하나로 한 번 — 가장 빠르다
        [해 족]     여러 출발점 → 허용 집합 안의 설계들을 표로 비교
        [최소 K]    모드 수 K 를 올려 가며 처음 통과하는 복잡도를 찾는다
  ▸ 계산은 전부 `gre` 패키지를 그대로 호출한다. **이 파일에 물리는 없다.**
    여기서 식을 다시 적으면 UI 와 코어가 조용히 갈라진다 (STEP 과
    STL 이 그렇게 갈라졌다). 기본값도 코드에서 읽어 화면에 채운다.

⚠ **도구는 입력을 바꾸지 않는다.** 노즐목과 케이스는 입력이다. 목표 곡선이
  그 케이스로 실현 불가능하면 조건을 몰래 넓혀 답을 내지 않고 **실현 불가로
  보고한다** (`runner.verdict_text`).

사용:
  python gre_web.py                   웹 UI 실행 (기본 포트 5080)
  python gre_web.py --port 5085
  python gre_web.py --no-browser
  python gre_web.py --selftest        서버 없이 계산 경로 검증
══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

N_TARGET_PTS = 400          # 목표 곡선 리샘플 점 수 (inverse_demo 와 같다)


# ══════════════════════════════════════════════════════════════════════════
#  기본값 — **코드에서 읽는다.** 화면에 손으로 베끼지 않는다.
# ══════════════════════════════════════════════════════════════════════════

def defaults() -> dict:
    from gre.core.burnrate import NAKKA_TABLE
    from gre.optimize.constraints import (D_THROAT_MIN_MM, ManufacturingLimits)
    from gre.optimize.forward import DiffMotor
    from gre.optimize.loss import Tolerances

    mo = DiffMotor(d_outer=60.0)
    lim = ManufacturingLimits()
    tol = Tolerances()
    # ⚠ `d_outer`·`n_grid` 는 **위쪽에 자기 입력칸이 있다.** 고급 패널에도
    #   내보내면 같은 값을 두 곳에서 받게 되고, 한쪽이 조용히 이긴다.
    skip = {"d_outer", "n_grid", "ds", "br", "dtype", "device", "n_samples"}
    motor = {f.name: getattr(mo, f.name) for f in fields(DiffMotor)
             if f.name not in skip and f.init}
    return {
        "propellants": list(NAKKA_TABLE.keys()),
        "d_throat_min_mm": D_THROAT_MIN_MM,
        "motor": {k: (v if not isinstance(v, bool) else bool(v))
                  for k, v in motor.items()},
        "limits": asdict(lim),
        "tol": asdict(tol),
    }


# ══════════════════════════════════════════════════════════════════════════
#  목표 곡선 만들기 — 네 갈래
# ══════════════════════════════════════════════════════════════════════════

def _parse_table(text: str):
    """붙여넣은/올린 CSV·TSV·공백 구분 표 → (t, F, P 또는 None).

    헤더가 있으면 건너뛴다. 열은 t, F, (P) 순으로 본다.
    """
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lstrip().startswith("#"):
            continue
        parts = [p for p in re.split(r"[,\t;\s]+", line) if p != ""]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            if rows:                       # 중간의 글자 줄은 알린다
                raise ValueError(f"숫자가 아닌 줄을 만났다: {line!r}")
            continue                       # 첫 줄은 헤더로 보고 넘긴다
        if len(vals) < 2:
            raise ValueError(f"열이 2 개보다 적다: {line!r}")
        rows.append(vals)
    if len(rows) < 8:
        raise ValueError(f"표의 줄이 {len(rows)} 개뿐이다 — 곡선이라 보기 어렵다")
    ncol = min(len(r) for r in rows)
    t = [r[0] for r in rows]
    f = [r[1] for r in rows]
    p = [r[2] for r in rows] if ncol >= 3 else None
    return t, f, p


def build_target(spec: dict, motor=None):
    """`spec["source"]` 에 따라 `Target` 과 설명을 만든다.

    반환 (target, info). info 는 화면에 그대로 보여 줄 한 줄짜리 요약.
    """
    import numpy as np
    import torch
    from gre.optimize.forward import resample
    from gre.optimize.loss import Target

    src = spec.get("source", "analytic")
    dt = torch.float64

    if src in ("paste", "upload"):
        text = spec.get("text", "")
        if not text.strip():
            raise ValueError("목표 곡선 표가 비어 있다")
        tt, ff, pp = _parse_table(text)
        t = torch.tensor(tt, dtype=dt)
        f = torch.tensor(ff, dtype=dt)
        p = torch.tensor(pp, dtype=dt) if pp is not None else None
        if bool(spec.get("ignore_pressure")):
            p = None
        # 같은 시간축으로 균일 리샘플 — Target 은 t 와 p 의 길이가 같아야 한다
        grid = torch.linspace(float(t[0]), float(t[-1]), N_TARGET_PTS, dtype=dt)
        f = resample(t, f, grid)
        if p is not None:
            p = resample(t, p, grid)
        tgt = Target(t=grid, f=f, p=p)
        info = (f"표 {len(tt)} 행 → {N_TARGET_PTS} 점 리샘플 · "
                f"{'압력 곡선 포함 (사양 5 개)' if p is not None else '추력만 (사양 3 개)'}")
        return tgt, info

    if src == "analytic":
        f_max = float(spec.get("f_max", 200.0))
        t_burn = float(spec.get("t_burn", 2.0))
        t_rise = float(spec.get("t_rise", 0.12))
        t_tail = float(spec.get("t_tail", 0.30))
        if not (t_rise + t_tail < t_burn):
            raise ValueError(
                f"상승 {t_rise} + 하강 {t_tail} 이 연소시간 {t_burn} 이상이다")
        grid = torch.linspace(0.0, t_burn, N_TARGET_PTS, dtype=dt)
        g = grid.numpy()
        f = np.full_like(g, f_max)
        rise = g < t_rise
        f[rise] = f_max * (g[rise] / t_rise)
        tail = g > (t_burn - t_tail)
        f[tail] = f_max * (t_burn - g[tail]) / t_tail
        tgt = Target(t=grid, f=torch.tensor(f, dtype=dt))
        info = (f"사다리꼴 {f_max:.0f} N × {t_burn:.2f} s "
                f"(상승 {t_rise:.2f} / 하강 {t_tail:.2f}) · 추력만 (사양 3 개)")
        return tgt, info

    if src == "known":
        # 알려진 포트 형상에서 **정방향으로** 목표를 만든다 — 복원 시연용.
        # 압력도 같은 모터에서 나오므로 사양 5 개가 된다.
        if motor is None:
            raise ValueError("알려진 형상 모드는 모터가 필요하다")
        from gre.optimize.runner import DesignVars
        r0 = float(spec.get("k_r0", 11.0))
        a = [float(x) for x in spec.get("k_a", [0.0, 0.0, 2.2, 0.0])]
        L = [float(x) for x in spec.get("k_lengths", [95.0])]
        at = float(spec.get("k_at", 78.0))
        dv = DesignVars(r0=torch.tensor(r0, dtype=dt),
                        a=torch.tensor(a, dtype=dt),
                        b=torch.zeros(len(a), dtype=dt),
                        a_throat=torch.tensor(at, dtype=dt), lengths=L)
        tt, tp, tf, td = motor.simulate(dv.r0, dv.a, dv.b, dv.lengths,
                                        dv.a_throat)
        grid = torch.linspace(0.0, float(tt[-1]), N_TARGET_PTS, dtype=dt)
        tgt = Target(t=grid, f=resample(tt, tf, grid).detach(),
                     p=resample(tt, tp, grid).detach())
        info = (f"알려진 형상에서 생성: r0 {r0} · a {a} · L {L} · "
                f"A_t {at:.1f} mm² → F_max {float(tf.max()):.0f} N · "
                f"P_max {float(tp.max()):.2f} MPa · 장약 {td['mass_g']:.0f} g "
                f"· 압력 포함 (사양 5 개)")
        return tgt, info

    raise ValueError(f"모르는 목표 곡선 입력 방식: {src!r}")


# ══════════════════════════════════════════════════════════════════════════
#  입력 → gre 객체
# ══════════════════════════════════════════════════════════════════════════

def build_motor(spec: dict):
    from gre.optimize.forward import DiffMotor
    kw = dict(spec.get("motor") or {})
    kw["d_outer"] = float(spec["d_outer"])
    kw["n_grid"] = int(spec.get("n_grid", 121))
    for k in ("density", "c_star", "c_star_eff", "cf_eff", "gamma",
              "p_ambient", "chamber_margin", "separation_frac", "h_max"):
        if k in kw:
            kw[k] = float(kw[k])
    for k in ("n_w",):
        if k in kw:
            kw[k] = int(kw[k])
    if "ends_burn" in kw:
        kw["ends_burn"] = bool(kw["ends_burn"])
    kw.pop("n_samples", None)
    return DiffMotor(**kw)


def build_throat(spec: dict):
    from gre.optimize.constraints import ThroatSpec
    mode = spec.get("throat_mode", "fixed")
    if mode == "fixed":
        return ThroatSpec.fixed(float(spec["throat_d"]))
    return ThroatSpec.band(float(spec["throat_dmin"]), float(spec["throat_dmax"]))


def build_limits(spec: dict):
    from gre.optimize.constraints import ManufacturingLimits
    kw = {k: float(v) for k, v in (spec.get("limits") or {}).items()}
    return ManufacturingLimits(**kw)


def build_tol(spec: dict):
    from gre.optimize.loss import Tolerances
    kw = {k: float(v) for k, v in (spec.get("tol") or {}).items()}
    return Tolerances(**kw)


def parse_lengths(text) -> list:
    if isinstance(text, (list, tuple)):
        vals = [float(x) for x in text]
    else:
        vals = [float(x) for x in re.split(r"[,\s]+", str(text).strip()) if x]
    if not vals:
        raise ValueError("세그먼트 길이가 비었다")
    if any(v <= 0 for v in vals):
        raise ValueError("세그먼트 길이는 양수여야 한다")
    return vals


# ══════════════════════════════════════════════════════════════════════════
#  작업 관리 — 최적화는 분 단위다. 한 번에 하나만 돌린다.
# ══════════════════════════════════════════════════════════════════════════

_IT_RE = re.compile(
    r"it\s+(\d+)\s+loss\s+(\S+)\s+모양\s+([\d.]+)%\s+I\s+([\d.]+)%"
    r"\s+F\s+([\d.]+)%\s+벌점\s+(\S+)\s+복잡도\s+([\d.]+)\s+장약\s+([\d.]+)g"
    r"\s*(집합 안)?")


class Job:
    """한 번의 역설계 실행. 진행 상황은 **stdout 을 읽어** 만든다.

    ⚠ `runner.optimize` 에 콜백을 넣지 않은 이유: 코어를 고치는 일이라
      먼저 허락을 받아야 한다. stdout 파싱은 알고리즘을 전혀 건드리지
      않으므로, 파싱이 실패해도 **결과는 그대로**고 진행 표시만 거칠어진다.
    """

    def __init__(self, jid: str, spec: dict):
        self.id = jid
        self.spec = spec
        self.status = "대기"
        self.started = time.time()
        self.finished = None
        self.iters_seen = 0
        self.iters_total = int(spec.get("_iters_total") or 0)
        self.last = None            # 마지막 반복 기록
        self.lines = []             # 로그 (최근 400 줄)
        self.result = None
        self.error = None
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    # ── 로그 ──────────────────────────────────────────────────────────
    def log(self, line: str):
        line = line.rstrip()
        if not line:
            return
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > 400:
                del self.lines[:-400]
        m = _IT_RE.search(line)
        if m:
            with self.lock:
                self.iters_seen += 1
                self.last = {
                    "it": int(m.group(1)), "loss": m.group(2),
                    "shape": float(m.group(3)), "impulse": float(m.group(4)),
                    "f_max": float(m.group(5)), "penalty": m.group(6),
                    "complexity": float(m.group(7)), "mass": float(m.group(8)),
                    "in_set": bool(m.group(9)),
                }

    def snapshot(self) -> dict:
        with self.lock:
            frac = (min(1.0, self.iters_seen / self.iters_total)
                    if self.iters_total else None)
            return {
                "id": self.id, "status": self.status,
                "elapsed": time.time() - self.started,
                "iters_seen": self.iters_seen, "iters_total": self.iters_total,
                "frac": frac, "last": self.last,
                "log": self.lines[-14:],
                "done": self.status in ("완료", "실패", "중단"),
                "error": self.error,
            }


class _Tee:
    """stdout 을 가로채 Job 으로 흘린다 (원래 콘솔에도 그대로 보낸다)."""

    def __init__(self, job: Job, orig):
        self.job, self.orig, self.buf = job, orig, ""

    def write(self, s):
        try:
            self.orig.write(s)
        except Exception:
            pass
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.job.log(line)

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass


JOBS: dict[str, Job] = {}
_ACTIVE = threading.Lock()


def estimate_iters(spec: dict) -> int:
    mode = spec.get("mode", "single")
    if mode == "single":
        return int(spec.get("iters", 90))
    if mode == "family":
        return int(spec.get("iters", 90)) * int(spec.get("n_starts", 6))
    # 최소 K — 조기 종료하므로 **상한**이다
    k_max = int(spec.get("k_max", 4))
    per = int(spec.get("iters_per_k", 120))
    seeds = (int(spec.get("n_seeds", 3))
             if spec.get("seeding", "warm") == "independent" else 1)
    return (k_max + 1) * per * seeds


# ══════════════════════════════════════════════════════════════════════════
#  실행
# ══════════════════════════════════════════════════════════════════════════

def run_job(job: Job):
    import torch
    from gre.optimize.loss import LossWeights
    from gre.optimize.runner import (final_report, minimal_complexity_search,
                                     optimize, seed_design, solution_family,
                                     verdict_text)
    from gre.validate.input_check import check_inputs

    spec = job.spec
    orig = sys.stdout
    sys.stdout = _Tee(job, orig)
    try:
        job.status = "모터 준비"
        motor = build_motor(spec)
        limits = build_limits(spec)
        tol = build_tol(spec)
        weights = LossWeights()
        throat = build_throat(spec)
        lengths = parse_lengths(spec.get("lengths", "95"))
        job.log(f"모터: Ø{motor.d_outer} · 격자 {motor.n_grid} · "
                f"표본 {motor.n_samples}"
                + (" (격자에 맞춰 올려졌다)" if motor.n_samples_raised else ""))

        job.status = "목표 곡선"
        target, tinfo = build_target(spec.get("target") or {}, motor)
        job.log(f"목표: {tinfo}")

        # ── 입력 점검 — 못 돌릴 입력이면 **여기서 멈춘다** ─────────────
        job.status = "입력 점검"
        at = (math.pi * float(spec["throat_d"]) ** 2 / 4.0
              if spec.get("throat_mode", "fixed") == "fixed" else None)
        rep_in = check_inputs(target, motor, limits, tuple(lengths), a_throat=at)
        findings = _findings(rep_in)
        job.log(f"입력 점검: 확정 불가 {len(rep_in.fatal)} · "
                f"경고 {len(rep_in.warnings)} · 검사 {len(findings)}")
        if not rep_in.runnable:
            job.result = {"mode": spec.get("mode"), "findings": findings,
                          "stopped_at_check": True,
                          "target_info": tinfo,
                          "verdict": "**이 목표로는 역설계를 시작하지 않습니다.** "
                                     "입력 점검에서 확정 불가가 나왔습니다 — "
                                     "아래 항목을 먼저 고쳐야 합니다.",
                          "charts": charts_target_only(target)}
            job.status = "완료"
            return
        if job.cancel.is_set():
            job.status = "중단"
            return

        mode = spec.get("mode", "single")
        job.status = f"최적화 ({mode})"
        n_modes = int(spec.get("n_modes", 4))
        extra = {}

        if mode == "single":
            dv = seed_design(r0=float(spec.get("r0_init", 12.0)),
                             n_modes=n_modes, lengths=tuple(lengths),
                             seed_amplitude=float(spec.get("seed_amp", 0.4)),
                             seed=int(spec.get("seed", 0)), throat=throat)
            dv0 = (float(dv.r0.detach()), dv.a.detach().clone().numpy().copy(),
                   dv.b.detach().clone().numpy().copy())
            dv, hist = optimize(motor, dv, target, limits, weights,
                                iters=int(spec.get("iters", 90)),
                                log_every=1, tol=tol, throat=throat)
            rep, curves = final_report(motor, dv, target, limits, weights,
                                       tol=tol)
            extra["hist"] = [float(h["loss"]) for h in hist]
            extra["seed_shape"] = dv0

        elif mode == "family":
            fam = solution_family(
                motor, target, n_starts=int(spec.get("n_starts", 6)),
                n_modes=n_modes, lengths=tuple(lengths),
                limits=limits, weights=weights,
                iters=int(spec.get("iters", 90)), tol=tol, throat=throat)
            extra["family"] = fam
            if not fam["accepted"]:
                job.result = family_result(fam, target, tol, findings, tinfo,
                                           motor, limits, None)
                job.status = "완료"
                return
            best = fam["recommended"]
            dv = _dv_from(best, lengths)
            rep, curves = final_report(motor, dv, target, limits, weights,
                                       tol=tol)
            extra["hist"] = None

        elif mode == "mink":
            # ⚠ 반환은 **(설계 또는 None, 보고)** 튜플이다. `feasible` 이 False
            #   면 돌아온 설계는 답이 아니라 **진단용**이므로 쓰면 안 된다.
            dv, krep = minimal_complexity_search(
                motor, target, limits=limits, weights=weights,
                k_max=int(spec.get("k_max", 4)),
                iters_per_k=int(spec.get("iters_per_k", 120)),
                tol=tol, lengths=tuple(lengths),
                r0_init=float(spec.get("r0_init", 12.0)),
                n_seeds=int(spec.get("n_seeds", 3)), throat=throat,
                seeding=spec.get("seeding", "warm"))
            extra["mink"] = {
                "k_chosen": krep.get("k_chosen"),
                "feasible": bool(krep.get("feasible")),
                "note": krep.get("note", ""),
                "k_table": _jsonable(krep.get("k_table") or []),
                "amplitude_floor_mm": krep.get("amplitude_floor_mm"),
                "amplitude_floor_binding": krep.get("amplitude_floor_binding"),
            }
            if dv is None:
                job.result = {
                    "mode": mode, "findings": findings, "target_info": tinfo,
                    "verdict": krep.get("note")
                              or "허용 집합에 드는 K 를 찾지 못했다.",
                    "in_set": False,
                    "mink": extra["mink"], "charts": charts_target_only(target)}
                job.status = "완료"
                return
            rep, curves = final_report(motor, dv, target, limits, weights,
                                       tol=tol)
            extra["hist"] = None
        else:
            raise ValueError(f"모르는 실행 모드: {mode!r}")

        if job.cancel.is_set():
            job.status = "중단"
            return

        job.status = "그림"
        adm = rep["admissibility"]
        job.result = {
            "mode": mode,
            "findings": findings,
            "target_info": tinfo,
            "verdict": verdict_text(adm),
            "in_set": bool(adm["in_set"]),
            "spec": {str(k): {"val": float(v), "tol": float(t),
                               "gap": float(g)}
                     for k, (v, t, g) in adm["spec"].items()},
            "manufacturing": {k: {"ok": bool(ok), "detail": d}
                              for k, (ok, d) in adm["manufacturing"].items()},
            "design": _design_dict(dv, motor, rep),
            "report": _jsonable({k: rep[k] for k in rep
                                 if k not in ("admissibility",)}),
            "charts": make_charts(motor, dv, target, curves, extra, rep, spec),
            "family_table": (extra.get("family", {}) or {}).get("table"),
            "family_rows": _family_rows(extra.get("family")),
            "spread_mm": (extra.get("family", {}) or {}).get("spread_mm"),
            "tol_shape": float(tol.shape),
            "mink": extra.get("mink"),
        }
        job.status = "완료"

    except Exception as exc:
        job.error = f"{exc.__class__.__name__}: {exc}"
        job.log("실패: " + job.error)
        for ln in traceback.format_exc().splitlines()[-6:]:
            job.log("  " + ln)
        job.status = "실패"
    finally:
        sys.stdout = orig
        job.finished = time.time()


def _safe(obj):
    """JSON 으로 내보내기 전 **유한하지 않은 수를 None 으로** 바꾼다.

    ⚠ `jsonify` 는 inf/nan 을 `Infinity`/`NaN` 으로 찍는데 그건 JSON 이
      아니다 — 브라우저의 `JSON.parse` 가 그 자리에서 터진다. 실제로 최소 K
      결과의 `width_mm: Infinity` 가 결과 화면을 통째로 날렸다. 값을 지어내지
      않고 **없음(None)** 으로 내려 화면에서 '—' 로 보이게 한다.
    """
    import math as _m
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, float) and not _m.isfinite(obj):
        return None
    return obj


def _findings(report) -> list:
    """`Finding` → JSON. ⚠ `ok` 는 numpy 비교에서 와 `np.bool_` 일 수 있는데,
    그 타입은 JSON 직렬화가 안 된다 (브라우저에서 TypeError 로 터졌다).
    여기서 한 번에 파이썬 타입으로 내린다."""
    return [{"code": str(f.code), "level": str(f.level), "ok": bool(f.ok),
             "title": str(f.title), "detail": str(f.detail),
             "action": str(f.action)} for f in report.findings]


def _dv_from(rep: dict, lengths):
    import torch
    from gre.optimize.runner import DesignVars
    dt = torch.float64
    return DesignVars(r0=torch.tensor(float(rep["r0"]), dtype=dt),
                      a=torch.tensor(rep["a"], dtype=dt),
                      b=torch.tensor(rep["b"], dtype=dt),
                      a_throat=torch.tensor(float(rep["a_throat"]), dtype=dt),
                      lengths=list(lengths))


def _design_dict(dv, motor, rep) -> dict:
    import numpy as np
    at = float(dv.a_throat.detach())
    return {
        "r0": float(dv.r0.detach()),
        "a": [float(x) for x in dv.a.detach().numpy()],
        "b": [float(x) for x in dv.b.detach().numpy()],
        "a_throat_mm2": at,
        "d_throat_mm": 2.0 * math.sqrt(at / math.pi),
        "lengths": list(dv.lengths),
        "d_outer": float(motor.d_outer),
        "mass_g": float(rep["mass_g"]),
        "web_mm": float(rep["web_mm"]),
        "r_tip_mm": float(rep["r_tip_mm"]),
        "P_max_sim": float(rep["P_max_sim"]),
        "components": int(rep.get("components", 1)),
        "complexity": float(rep["complexity"]),
    }


def _family_rows(fam):
    if not fam:
        return None
    out = []
    for r in fam["family"]:
        out.append({
            # ⚠ 계수를 같이 싣는다. 초판은 요약 숫자만 실어서, 사용자가 표를
            #   보고 **추천이 아닌 씨앗**을 고르면 그 설계를 다시 만들 방법이
            #   없었다 (씨앗을 통째로 재현해야 했다).
            "a": [float(x) for x in r["a"]],
            "b": [float(x) for x in r["b"]],
            "seed": int(r["seed"]), "r0": float(r["r0"]),
            "shape": float(r["rms_shape_pct"]), "impulse": float(r["I_err_pct"]),
            "f_max": float(r["F_err_pct"]),
            # ⚠ 압력 두 항을 **반드시 같이** 싣는다. 초판은 p_max 를 담고도
            #   화면에 안 그렸고 p_shape 는 담지도 않았다. 그래서 해 족 표만
            #   보면 사양 5 개 중 3 개만 보였고, 실측에서 최악 여유가
            #   0.02 %p 인 설계가 1.35 %p 로 보였다 (압력 모양이 숨어서).
            "p_max": float(r.get("P_err_pct", float("nan"))),
            "p_shape": float(r.get("P_shape_pct", float("nan"))),
            "complexity": float(r["complexity"]),
            "sliver": 100.0 * float(r["sliver_frac"]),
            "mass": float(r["mass_g"]),
            "d_throat": 2.0 * math.sqrt(float(r["a_throat"]) / math.pi),
            "ok": bool(r["within_tol"] and r["feasible"]),
        })
    return out


def _jsonable(obj):
    import numpy as np
    import torch
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()
                if not isinstance(v, (torch.Tensor,))
                or v.numel() <= 64}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [float(x) for x in obj.ravel()[:64]]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return [float(x) for x in obj.detach().reshape(-1)[:64]]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (str, int, float)):
        return obj
    return str(obj)


def family_result(fam, target, tol, findings, tinfo, motor, limits, dv):
    return {
        "mode": "family", "findings": findings, "target_info": tinfo,
        "verdict": ("**허용 집합 안에 든 설계가 하나도 없습니다.** "
                    "씨앗을 모두 돌렸지만 사양·제작 제약을 함께 만족하는 해가 "
                    "없었습니다. 목표 곡선이나 입력 조건(케이스·노즐목·장약 "
                    "상한)을 넓혀야 합니다."),
        "in_set": False,
        "family_table": fam.get("table"),
        "family_rows": _family_rows(fam),
        "charts": charts_target_only(target),
    }


# ══════════════════════════════════════════════════════════════════════════
#  CAD 내보내기 — 역설계 결과 → 몰드 일습 (STEP + STL)
# ══════════════════════════════════════════════════════════════════════════

def export_cad_for(job: "Job", body: dict = None) -> dict:
    """끝난 작업의 **복원된 포트**로 `export_set` 을 돌린다.

    사용자가 2026-10-30 에 못박은 세 규칙은 전부 `FitSpec` 쪽에 들어가 있다 —
    여기서 다시 적지 않는다 (적으면 두 곳이 갈라진다):

        마스터 높이 = 그레인 길이 + `mold_head`(10)   `parts.mold_height`
        지관 벽 두께 = `tube_wall` **2.0 고정**        `FitSpec.tube_wall`
        Top 기둥 = 바깥 벽 + `mold_top_post`(5) 항상   `silicone_mold_top_bands`
    """
    from gre.export import FitSpec, export_set
    from gre.export import parts as PT
    from gre.geometry.port import FourierPort

    body = body or {}
    if not job.result or not job.result.get("design"):
        raise ValueError("내보낼 설계가 없다 — 역설계를 먼저 끝내야 한다")
    d = job.result["design"]
    if job.result.get("in_set") is False:
        raise ValueError(
            "허용 집합 밖 설계는 내보내지 않는다. 그 설계는 답이 아니라 "
            "**진단용**이다 — 목표나 입력 조건을 고친 뒤 다시 "
            "돌리십시오.")

    lengths = [float(x) for x in d["lengths"]]
    prop_len = lengths[0]
    warn = []
    if len({round(x, 6) for x in lengths}) > 1:
        warn.append(f"세그먼트 길이가 서로 다르다 {lengths} — 첫 값 "
                    f"{prop_len:g} mm 로만 몰드를 만들었다. 길이가 다르면 "
                    f"세그먼트마다 따로 뽑아야 한다.")

    outdir = str(body.get("outdir") or "").strip() or os.path.join(
        "out", f"cad_{job.id}")
    outdir = os.path.abspath(outdir)
    port = FourierPort(r0=d["r0"], a=d["a"], b=d["b"])
    fit = FitSpec()

    # ⚠⚠ **같은 이름의 다른 길이다.** `DiffMotor.lengths` 는 **추진제**가 타는
    #   길이이고, `export_set(length=...)` 은 **지관** 길이다 (몰드가
    #   `length − 2·end_inset` 을 추진제로 삼는다). 초판은 추진제 길이를 지관
    #   길이로 그대로 넘겨서, 설계가 74 mm 를 태우기로 해 놓고 몰드는 66 mm
    #   짜리를 찍고 있었다 — 장약이 395.6 g 이어야 하는데 353.0 g 이었다.
    #   아래 질량 대조가 그것을 잡는다.
    tube_len = prop_len + 2.0 * float(fit.end_inset)
    rep = export_set(port, float(d["d_outer"]), tube_len, outdir, fit=fit,
                     step=bool(body.get("step", True)))

    # ── 자가 대조: 몰드가 만드는 장약 == 설계가 태우기로 한 장약 ───────────
    n_seg = len(lengths)
    m_mold = float(rep.propellant["mass_target_g"]) * n_seg
    m_design = float(d.get("mass_g") or 0.0)
    if m_design > 0 and abs(m_mold - m_design) / m_design > 0.01:
        warn.append(
            f"⚠ 몰드가 만드는 장약 {m_mold:.1f} g 이 설계가 태우기로 한 "
            f"{m_design:.1f} g 과 {100*(m_mold-m_design)/m_design:+.1f} % "
            f"다르다. 길이 해석이 어긋났을 수 있다 — 쓰기 전에 확인할 것.")

    h_plate = PT.mold_height(tube_len, fit)
    h_wall_top = float(fit.mold_groove_depth) + float(fit.mold_top_back)
    files = sorted(os.listdir(outdir))
    return {
        "ok": True,
        "outdir": outdir,
        "files": files,
        "n_files": len(files),
        "parts": [{"name": q["name"], "volume_mm3": q["volume_mm3"],
                   "watertight": bool(q["watertight"]), "files": q["files"]}
                  for q in rep.parts],
        "rules": {
            "추진제 길이 (설계)": f"{prop_len:g} mm × {n_seg} 세그먼트",
            "지관 길이": f"{tube_len:g} mm = 추진제 {prop_len:g} + "
                         f"2 × 단차 {fit.end_inset:g}",
            "마스터 높이": f"{h_plate:.1f} mm = 지관 {tube_len:g} + "
                           f"{fit.mold_head:g}",
            "지관 벽 두께": f"{fit.tube_wall:.1f} mm (고정)",
            "Top 기둥 꼭대기": f"{h_wall_top + fit.mold_top_post:.1f} mm = "
                               f"바깥 벽 {h_wall_top:.1f} + "
                               f"{fit.mold_top_post:g}",
            "장약 대조": f"몰드 {m_mold:.1f} g vs 설계 {m_design:.1f} g",
        },
        "propellant": _safe(rep.propellant),
        "fidelity": _safe(rep.fidelity),
        "warnings": warn + list(rep.warnings),
    }


# ══════════════════════════════════════════════════════════════════════════
#  그림
# ══════════════════════════════════════════════════════════════════════════

def _pick_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("Malgun Gothic", "AppleGothic", "NanumGothic",
              "Noto Sans CJK KR", "Noto Sans KR", "NanumBarunGothic"):
        if c in have:
            return c
    return None


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fnt = _pick_font()
    if fnt:
        plt.rcParams["font.family"] = fnt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def charts_target_only(target):
    """실행 전/실패 시 — 목표 곡선만이라도 보여 준다."""
    plt = _setup_mpl()
    n = 2 if target.p is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.0))
    axes = [axes] if n == 1 else list(axes)
    axes[0].plot(target.t.numpy(), target.f.numpy(), color="#161616", lw=2.0)
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("추력 [N]")
    axes[0].set_title("목표 추력 곡선", fontsize=11)
    axes[0].grid(alpha=.25)
    if n == 2:
        axes[1].plot(target.t.numpy(), target.p.numpy(), color="#da1e28", lw=2.0)
        axes[1].set_xlabel("t [s]")
        axes[1].set_ylabel("압력 [MPa]")
        axes[1].set_title("목표 압력 곡선", fontsize=11)
        axes[1].grid(alpha=.25)
    fig.tight_layout()
    return [{"title": "목표 곡선", "png": _b64(fig)}]


def _port_xy(r0, a, b, n=512):
    import numpy as np
    th = np.linspace(0, 2 * np.pi, n)
    k = np.arange(1, len(a) + 1)[:, None]
    r = float(r0) + (np.asarray(a)[:, None] * np.cos(k * th)
                     + np.asarray(b)[:, None] * np.sin(k * th)).sum(0)
    return r * np.cos(th), r * np.sin(th)


def make_charts(motor, dv, target, curves, extra, rep, spec):
    """수렴 · 추력 · 압력 · 포트 형상 (+ 모드별 추가)."""
    import numpy as np
    plt = _setup_mpl()
    t, p, f, diag, omega = curves
    tn = t.detach().numpy()
    out = []

    # ── 곡선 비교 ────────────────────────────────────────────────────
    has_p = target.p is not None
    ncol = 3 if has_p else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.0))
    ax = list(np.atleast_1d(ax))

    ax[0].plot(target.t.numpy(), target.f.numpy(), color="#8d8d8d", lw=3.4,
               alpha=.65, label="목표")
    ax[0].plot(tn, f.detach().numpy(), color="#0f62fe", lw=1.8, label="복원")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("추력 [N]")
    ax[0].set_title(f"추력 — 모양 {rep['rms_shape_pct']:.2f}% · "
                    f"I {rep['I_err_pct']:.2f}% · F {rep['F_err_pct']:.2f}%",
                    fontsize=10)
    ax[0].grid(alpha=.25)
    ax[0].legend(fontsize=9)

    i = 1
    if has_p:
        ax[i].plot(target.t.numpy(), target.p.numpy(), color="#8d8d8d", lw=3.4,
                   alpha=.65, label="목표")
        ax[i].plot(tn, p.detach().numpy(), color="#da1e28", lw=1.8, label="복원")
        ax[i].axhline(float(spec.get("limits", {}).get("p_case_allow", 11.0)),
                      ls="--", lw=1.2, color="#a2191f")
        ax[i].set_xlabel("t [s]")
        ax[i].set_ylabel("압력 [MPa]")
        ttl = f"압력 — 최대압 {rep['P_err_pct']:.2f}%"
        if "P_shape_pct" in rep:
            ttl += f" · 모양 {rep['P_shape_pct']:.2f}%"
        ax[i].set_title(ttl, fontsize=10)
        ax[i].grid(alpha=.25)
        ax[i].legend(fontsize=9)
        i += 1

    r_out = float(motor.d_outer) / 2.0
    x, y = _port_xy(float(dv.r0.detach()), dv.a.detach().numpy(),
                    dv.b.detach().numpy())
    seed = extra.get("seed_shape")
    if seed is not None:
        sx, sy = _port_xy(seed[0], seed[1], seed[2])
        ax[i].plot(sx, sy, color="#c6c6c6", ls=":", lw=1.3, label="출발")
    ax[i].plot(x, y, color="#0f62fe", lw=2.2, label="복원 포트")
    ax[i].add_patch(plt.Circle((0, 0), r_out, fill=False, ec="#161616",
                               lw=1.4, ls="--"))
    ax[i].set_aspect("equal")
    lim = r_out * 1.12
    ax[i].set_xlim(-lim, lim)
    ax[i].set_ylim(-lim, lim)
    ax[i].set_title(f"포트 형상 — 웹 {rep['web_mm']:.2f} · "
                    f"꼭짓점 {rep['r_tip_mm']:.2f} mm", fontsize=10)
    ax[i].set_xlabel("x [mm]")
    ax[i].legend(fontsize=9)
    fig.tight_layout()
    out.append({"title": "곡선과 형상", "png": _b64(fig)})

    # ── 수렴 ─────────────────────────────────────────────────────────
    hist = extra.get("hist")
    # ⚠ 점이 하나면 그리지 않는다. `stop_in_set` 으로 반복 0 에서 끝나는 일이
    #   흔한데, 그때 빈 축만 큼직하게 나와 "수렴 그림이 비었다" 로 읽힌다.
    if hist and len(hist) >= 2:
        fig2, a2 = plt.subplots(figsize=(6.2, 3.6))
        a2.plot(hist, color="#0f62fe", lw=1.6)
        # 집합 안에서 사양 힌지는 정확히 0 이라 손실이 0 이 될 수 있다 —
        # 그러면 로그 축이 그릴 것이 없다. 0 이 섞이면 선형으로 둔다.
        if min(hist) > 0.0:
            a2.set_yscale("log")
        a2.set_xlabel("반복")
        a2.set_ylabel("손실")
        a2.set_title("Adam 수렴 — 손실은 반복 간 비교가 안 된다 "
                     "(희소성 램프)", fontsize=10)
        a2.grid(alpha=.25)
        fig2.tight_layout()
        out.append({"title": "수렴", "png": _b64(fig2)})

    # ── 해 족 산점도 ─────────────────────────────────────────────────
    fam = extra.get("family")
    if fam:
        fig3, a3 = plt.subplots(figsize=(6.2, 4.2))
        for r in fam["family"]:
            ok = r["within_tol"] and r["feasible"]
            a3.scatter(r["complexity"], r["rms_shape_pct"],
                       s=90, marker="o" if ok else "x",
                       color="#0f62fe" if ok else "#da1e28",
                       label=None, zorder=3)
            a3.annotate(f"s{r['seed']}", (r["complexity"], r["rms_shape_pct"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=8,
                        color="#525252")
        a3.axhline(100 * float(spec.get("tol", {}).get("shape", 0.10)),
                   ls="--", lw=1.3, color="#8d8d8d")
        a3.set_xlabel("복잡도 Σ|c_k| [mm]")
        a3.set_ylabel("모양 오차 [%]")
        a3.set_title("해 족 — 파랑 합격 / 빨강 탈락, 점선은 모양 허용 오차",
                     fontsize=10)
        a3.grid(alpha=.25)
        fig3.tight_layout()
        out.append({"title": "해 족", "png": _b64(fig3)})

    return out


# ══════════════════════════════════════════════════════════════════════════
#  자체 검증
# ══════════════════════════════════════════════════════════════════════════

def selftest() -> bool:
    ok = True

    def chk(name, got, want=True):
        nonlocal ok
        good = (got == want)
        print(f"  {'✓' if good else '✗'} {name}: {got!r}"
              + ("" if good else f"  (기대 {want!r})"))
        ok = ok and good

    print("기본값을 코드에서 읽는가")
    d = defaults()
    chk("추진제 목록", d["propellants"], ["KNSB", "KNDX", "KNSU"])
    chk("제작 제약 항목", sorted(d["limits"]),
        sorted(["w_web_min", "t_prop_min", "t_void_min", "r_tip_min",
                "slenderness_max", "mass_max", "p_case_allow"]))
    chk("허용 오차 항목", sorted(d["tol"]),
        sorted(["shape", "impulse", "f_max", "p_shape", "p_max"]))
    chk("허용 오차 기본 10 %", d["tol"]["shape"], 0.10)
    chk("노즐목 하한", d["d_throat_min_mm"], 2.52)

    print("표 파싱")
    t, f, p = _parse_table("t,F\n0,0\n0.1,10\n0.2,20\n0.3,30\n0.4,40\n"
                           "0.5,50\n0.6,60\n0.7,70\n")
    chk("헤더를 건너뛴다", len(t), 8)
    chk("압력 열이 없으면 None", p, None)
    _, _, p2 = _parse_table("\n".join(f"{i*0.1},{i*10},{1+i*0.1}"
                                      for i in range(10)))
    chk("세 번째 열은 압력", len(p2), 10)
    try:
        _parse_table("0,1\n1,2\n")
        chk("너무 짧은 표는 거부", False)
    except ValueError:
        chk("너무 짧은 표는 거부", True)

    print("목표 곡선 — 사다리꼴")
    tgt, info = build_target({"source": "analytic", "f_max": 200.0,
                              "t_burn": 2.0, "t_rise": 0.1, "t_tail": 0.3})
    chk("점 수", len(tgt.t), N_TARGET_PTS)
    chk("최대추력", round(float(tgt.f.max())), 200)
    chk("압력이 없으면 사양 3 개", tgt.pressure_is_spec, False)
    try:
        build_target({"source": "analytic", "f_max": 100.0, "t_burn": 0.3,
                      "t_rise": 0.2, "t_tail": 0.2})
        chk("상승+하강 ≥ 연소시간은 거부", False)
    except ValueError:
        chk("상승+하강 ≥ 연소시간은 거부", True)

    print("노즐목은 입력이다")
    from gre.optimize.constraints import ThroatSpec
    th = build_throat({"throat_mode": "fixed", "throat_d": 10.0})
    chk("고정 모드", th.d_mm, 10.0)
    th2 = build_throat({"throat_mode": "band", "throat_dmin": 8.0,
                        "throat_dmax": 12.0})
    chk("범위 모드", (th2.d_min_mm, th2.d_max_mm), (8.0, 12.0))
    try:
        ThroatSpec()
        chk("기본 생성은 거부된다", False)
    except ValueError:
        chk("기본 생성은 거부된다", True)

    print("JSON 안전성")
    bad = {"a": float("inf"), "b": [float("nan"), 1.0], "c": {"d": -float("inf")}}
    sf = _safe(bad)
    chk("inf → None", sf["a"], None)
    chk("nan → None", sf["b"][0], None)
    chk("중첩도 처리", sf["c"]["d"], None)
    chk("멀쩡한 값은 그대로", sf["b"][1], 1.0)
    json.dumps(sf, allow_nan=False)          # 터지면 테스트가 실패한다
    chk("엄격 JSON 으로 직렬화된다", True)

    print("진행 파싱 (stdout → 진행률)")
    j = Job("t", {})
    j.log("  it   12  loss 1.0623e-03  모양   6.80%  I   9.93%  F   0.04%"
          "  벌점 0.00e+00  복잡도  1.58  장약  369.4g  집합 안")
    chk("반복 번호", j.last["it"], 12)
    chk("모양 오차", j.last["shape"], 6.80)
    chk("집합 안 표시", j.last["in_set"], True)
    j.log("아무 줄이나")
    chk("무관한 줄은 진행을 올리지 않는다", j.iters_seen, 1)

    print("반복 수 추정")
    chk("단일", estimate_iters({"mode": "single", "iters": 90}), 90)
    chk("해 족", estimate_iters({"mode": "family", "iters": 90,
                                 "n_starts": 6}), 540)
    chk("최소 K (warm)", estimate_iters({"mode": "mink", "k_max": 4,
                                         "iters_per_k": 120,
                                         "seeding": "warm"}), 600)

    print("짧은 실제 실행 — 알려진 형상에서 생성 → 단일 모드 8 반복")
    spec = {
        "mode": "single", "d_outer": 60.0, "n_grid": 121, "iters": 8,
        "lengths": "95", "n_modes": 4, "r0_init": 13.0, "seed": 1,
        "seed_amp": 0.4, "throat_mode": "fixed", "throat_d": 9.966,
        "motor": {"n_w": 220, "h_max": 5e-3},
        "target": {"source": "known", "k_r0": 11.0, "k_a": [0, 0, 2.2, 0],
                   "k_lengths": [95.0], "k_at": 78.0},
        "limits": {}, "tol": {},
    }
    job = Job("st", spec)
    t0 = time.perf_counter()
    run_job(job)
    print(f"    ({time.perf_counter()-t0:.0f} s, 상태 {job.status})")
    chk("실행 상태", job.status, "완료")
    if job.result:
        r = job.result
        chk("판정 문장이 있다", bool(r.get("verdict")), True)
        chk("압력이 사양에 들어간다", "최대압" in (r.get("spec") or {}), True)
        chk("그림이 나온다", len(r.get("charts") or []) >= 1, True)
        chk("설계 변수가 나온다", "d_throat_mm" in (r.get("design") or {}), True)
        d = r["design"]
        chk("노즐목이 입력 그대로다",
            abs(d["d_throat_mm"] - 9.966) < 0.02, True)
        chk("진행률이 올라갔다", job.iters_seen >= 1, True)
    else:
        chk("결과가 있다", False)

    print("CAD 내보내기 — 사용자 세 규칙")
    if job.result and job.result.get("in_set"):
        import shutil
        out = "/tmp/gre_web_selftest_cad"
        shutil.rmtree(out, ignore_errors=True)
        cad = export_cad_for(job, {"outdir": out, "step": False})
        chk("내보내기 성공", cad["ok"], True)
        chk("부품 9 종", len(cad["parts"]), 9)
        chk("전부 수밀", all(q["watertight"] for q in cad["parts"]), True)
        chk("실리콘 주형 2 종이 들어 있다",
            sum(q["name"].startswith("silicone_mold") for q in cad["parts"]), 2)
        from gre.export import FitSpec
        from gre.export import parts as PT
        from gre.geometry.port import FourierPort
        d = job.result["design"]
        fit = FitSpec()
        L = float(d["lengths"][0])
        port = FourierPort(r0=d["r0"], a=d["a"], b=d["b"])
        pl = PT.silicone_mold_plate(port, float(d["d_outer"]),
                                    L + 2.0 * fit.end_inset, fit).bounds()
        tp = PT.silicone_mold_top(port, float(d["d_outer"]), fit).bounds()
        tube = L + 2.0 * fit.end_inset
        chk("① 마스터 높이 = 지관 + 10",
            round(pl[1][2] - pl[0][2], 6), round(tube + 10.0, 6))
        chk("② 지관 벽 2.0 고정", fit.tube_wall, 2.0)
        chk("③ Top 기둥 = 바깥 벽 + 5",
            round(tp[1][2] - tp[0][2], 6),
            round(fit.mold_groove_depth + fit.mold_top_back
                  + fit.mold_top_post, 6))
        # ⚠ 이게 핵심이다 — 몰드가 만드는 장약이 설계가 태우기로 한 장약과
        #   같아야 한다. 초판은 추진제 길이를 지관 길이로 넘겨 11 % 모자랐다.
        n_seg = len(d["lengths"])
        m_mold = cad["propellant"]["mass_target_g"] * n_seg
        chk("④ 몰드 장약 == 설계 장약 (1 % 안)",
            abs(m_mold - d["mass_g"]) / d["mass_g"] < 0.01, True)
        chk("   장약 경고가 안 뜬다",
            not any("장약" in w for w in cad["warnings"]), True)
        shutil.rmtree(out, ignore_errors=True)
    else:
        chk("CAD 검증을 돌릴 집합-안 결과가 있다", False)

    print("\n" + ("전부 통과" if ok else "실패 있음"))
    return ok


# ══════════════════════════════════════════════════════════════════════════
#  페이지
# ══════════════════════════════════════════════════════════════════════════

PAGE = r"""<!doctype html>
<html lang="ko" data-theme="g100">
<head>
<meta charset="utf-8">
<title>그레인 형상 역설계</title>
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
main{max-width:1420px; margin:0 auto; padding:1.5rem 1rem 4rem}
h1{font-size:22px; font-weight:400; margin-bottom:.25rem}
.sub{color:var(--text2); font-size:13px; margin-bottom:1.25rem;
     line-height:1.6; max-width:980px}
.grid{display:grid; grid-template-columns:minmax(380px,460px) minmax(0,1fr);
      gap:1.25rem; align-items:start}
@media(max-width:1020px){.grid{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--layer); border:1px solid var(--layer2);
      padding:1rem; min-width:0; overflow:hidden}
.card h2{font-size:14px; font-weight:600; margin-bottom:.75rem;
         color:var(--accent)}
fieldset{border:none; margin-bottom:1.1rem; min-width:0}
legend{font-size:11px; color:var(--text2); margin-bottom:.5rem;
       letter-spacing:.05em; font-weight:500}
.row{display:flex; gap:.5rem; margin-bottom:.5rem;
     align-items:center; min-width:0}
.row label{flex:1.4 1 0; min-width:0; font-size:12px; color:var(--text2);
           display:flex; align-items:center; line-height:1.35;
           flex-wrap:wrap; overflow-wrap:anywhere}
/* 보조 설명은 **줄을 바꾼다.** flex 안에서는 <br> 이 안 먹어서 라벨에
   붙어 버린다 ("세그먼트 길이 [mm]쉼표로 여러 개" 처럼). */
.row label span{flex:1 0 100%; display:block; font-size:10px; opacity:.85}
.row label br{display:none}
.row input,.row select,.row textarea{flex:1 1 0; min-width:0; width:100%;
    max-width:100%; background:var(--bg); color:var(--text);
    border:none; border-bottom:1px solid var(--border);
    padding:.4rem .5rem; font-size:13px; font-family:inherit}
.row textarea{font-family:'IBM Plex Mono',monospace; font-size:12px;
    min-height:120px; border:1px solid var(--border)}
.row select{appearance:none; -webkit-appearance:none; cursor:pointer}
.row input:focus,.row select:focus,.row textarea:focus{
    outline:2px solid var(--blue); outline-offset:-1px}
.row.wide{flex-direction:column; align-items:stretch; gap:.3rem}
.row.wide label{flex:none}
.btn{width:100%; background:var(--blue); color:#fff; border:none;
     padding:.8rem; font-size:14px; cursor:pointer; margin-top:.5rem;
     font-family:inherit}
.btn:disabled{background:var(--layer2); color:var(--text2); cursor:not-allowed}
.btn.ghost{background:var(--layer2); color:var(--text)}
.mini{background:var(--layer2); color:var(--text); border:none;
      padding:.4rem .7rem; font-size:12px; cursor:pointer;
      font-family:inherit; flex:0 0 auto; white-space:nowrap}
.mini:hover{background:var(--border)}
.row.btns{gap:.35rem; flex-wrap:wrap; margin-top:.6rem}
#status{font-size:12px; color:var(--accent); margin-top:.6rem; min-height:16px}
.note{font-size:11px; color:var(--text2); margin-top:.6rem; line-height:1.6}
details{border-left:2px solid var(--layer2); padding-left:.7rem;
        margin-bottom:.9rem}
details>summary{font-size:12px; color:var(--accent); cursor:pointer;
        padding:.25rem 0; list-style:none; user-select:none}
details>summary::before{content:'▸ '; }
details[open]>summary::before{content:'▾ '; }
.modehelp{font-size:11px; color:var(--text2); line-height:1.6;
          background:var(--bg); padding:.5rem .6rem; margin:.1rem 0 .7rem;
          border-left:2px solid var(--accent)}
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
table.iter td.l,table.iter th.l{text-align:left}
table.iter tr.bad td{background:rgba(250,77,86,.09)}
table.iter tr.good td{background:rgba(66,190,101,.09)}
table.iter tr.final td{background:rgba(15,98,254,.12); font-weight:600}
.verdict{padding:.85rem 1rem; margin-bottom:1rem; font-size:14px;
         line-height:1.65; border-left:4px solid var(--border);
         background:var(--bg)}
.verdict.good{border-left-color:var(--ok)}
.verdict.err{border-left-color:var(--err)}
.warn{background:rgba(241,194,27,.08); border-left:3px solid var(--warn);
      padding:.55rem .75rem; font-size:12px; margin-top:.4rem;
      line-height:1.6}
.bad{background:rgba(250,77,86,.1); border-left:3px solid var(--err);
      padding:.55rem .75rem; font-size:12px; margin-top:.4rem;
      line-height:1.6}
.good{background:rgba(66,190,101,.07); border-left:3px solid var(--ok);
      padding:.5rem .75rem; font-size:12px; margin-top:.4rem;
      line-height:1.6}
.errbox{background:rgba(250,77,86,.1); border-left:3px solid var(--err);
      padding:.6rem .8rem; font-size:13px; margin-top:.5rem;
      line-height:1.6; white-space:pre-wrap;
      font-family:'IBM Plex Mono',monospace}
.chart{width:100%; max-width:920px; background:#fff;
       border:1px solid var(--layer2); margin-top:.8rem; display:block}
.empty{color:var(--text2); padding:3rem 1rem; text-align:center;
       border:1px dashed var(--layer2); line-height:1.8}
.cap{font-size:11px; color:var(--text2); margin-top:.5rem; line-height:1.6}
h3.sec{font-size:12px; color:var(--text2); letter-spacing:.05em;
       margin:1.2rem 0 .4rem; font-weight:500; text-transform:uppercase}
.bar{height:6px; background:var(--layer2); margin:.5rem 0 .35rem;
     overflow:hidden}
.bar i{display:block; height:100%; background:var(--blue); width:0;
       transition:width .35s}
.bar.indet i{width:35%; animation:slide 1.4s infinite linear}
@keyframes slide{from{margin-left:-35%}to{margin-left:100%}}
pre.log{background:var(--bg); border:1px solid var(--layer2); padding:.6rem;
    font-family:'IBM Plex Mono',monospace; font-size:11px; line-height:1.55;
    max-height:220px; overflow:auto; white-space:pre-wrap; margin-top:.5rem;
    color:var(--text2)}
pre.table{background:var(--bg); border:1px solid var(--layer2);
    padding:.6rem; font-family:'IBM Plex Mono',monospace; font-size:11.5px;
    line-height:1.6; overflow:auto; white-space:pre; margin-top:.5rem}
.pill{display:inline-block; font-size:11px; padding:.1rem .45rem;
      border:1px solid var(--border); color:var(--text2); margin-left:.35rem}
</style>
</head>
<body>
<header>
  <div class="name">그레인 형상 <span>역설계</span></div>
  <div class="sp"></div>
  <button id="theme" title="테마">◐</button>
</header>
<main>
  <h1>목표 곡선 → <span style="color:var(--accent)">포트 형상</span></h1>
  <p class="sub">목표 추력(·압력) 곡선과 <b>케이스·노즐목</b>을 넣으면 포트
    형상을 역설계하고 <b>추력·압력 10 % 이내</b>로 판정합니다.
    노즐목과 케이스는 <b>입력</b>이라 도구가 바꾸지 않습니다 — 그 조건으로
    목표가 실현 불가능하면 억지로 답을 내지 않고 <b>실현 불가로
    보고</b>합니다.</p>

  <div class="grid">
    <div class="card">
      <h2>입력</h2>

      <fieldset>
        <legend>실행 모드</legend>
        <div class="row"><label>모드</label>
          <select id="mode">
            <option value="single">단일 실행 — 씨앗 하나</option>
            <option value="family">해 족 — 여러 출발점 비교</option>
            <option value="mink">최소 복잡도 K 탐색</option>
          </select></div>
        <div class="modehelp" id="modehelp"></div>
        <div id="modeParams"></div>
      </fieldset>

      <fieldset>
        <legend>목표 곡선</legend>
        <div class="row"><label>입력 방식</label>
          <select id="tsrc">
            <option value="known">알려진 형상에서 생성 (복원 시연)</option>
            <option value="analytic">사다리꼴 생성</option>
            <option value="paste">표 붙여넣기 (t, F[, P])</option>
            <option value="upload">CSV 파일</option>
          </select></div>
        <div id="tgtParams"></div>
      </fieldset>

      <fieldset>
        <legend>케이스 · 노즐목 — 입력이다 (도구가 바꾸지 않는다)</legend>
        <div class="row"><label>케이스 내경 Ø [mm]</label>
          <input id="d_outer" value="60"></div>
        <div class="row"><label>세그먼트 길이 [mm]<br>
          <span style="font-size:10px">쉼표로 여러 개</span></label>
          <input id="lengths" value="95"></div>
        <div class="row"><label>노즐목</label>
          <select id="throat_mode">
            <option value="fixed">고정 직경 — 보유 노즐</option>
            <option value="band">범위 — 이 안에서 고른다</option>
          </select></div>
        <div id="throatParams"></div>
      </fieldset>

      <fieldset>
        <legend>예산 · 격자</legend>
        <div class="row"><label>격자 n_grid</label>
          <select id="n_grid">
            <option value="121">121 (운용 · h 0.50 mm)</option>
            <option value="241">241 (조임 · h 0.25 mm · 느리다)</option>
          </select></div>
        <div class="row"><label>모드 수 K</label><input id="n_modes" value="4"></div>
        <div class="row"><label>초기 r0 [mm]</label><input id="r0_init" value="12"></div>
      </fieldset>

      <details>
        <summary>제작 제약 (ManufacturingLimits)</summary>
        <div id="limBox"></div>
        <div class="note">⚠ <code>t_prop_min</code> 은 격자에 양자화된다 —
          n=121 에서 분해능 1.0 mm. T2 판독기(<code>t2_input_web.py</code>)가
          내는 값을 여기에 넣는다.</div>
      </details>

      <details>
        <summary>허용 오차 (합격 기준)</summary>
        <div id="tolBox"></div>
        <div class="note">전부 0.10 = 10 %. 압력 두 항은 <b>압력 곡선이
          입력으로 들어왔을 때만</b> 사양이 된다.</div>
      </details>

      <details>
        <summary>추진제 · 모델 계수</summary>
        <div id="motorBox"></div>
      </details>

      <button class="btn" id="go">역설계 실행</button>
      <div class="row btns">
        <button type="button" class="mini" id="check">입력만 점검</button>
        <button type="button" class="mini" id="stop" disabled>중단</button>
        <button type="button" class="mini" id="reset">기본값으로</button>
      </div>
      <div id="status"></div>
      <div class="bar" id="bar" style="display:none"><i></i></div>
      <pre class="log" id="log" style="display:none"></pre>
    </div>

    <div class="card" id="out">
      <h2>결과</h2>
      <div class="empty">왼쪽에서 모드와 목표 곡선을 정하고
        [역설계 실행] 을 누르세요.<br>
        처음이면 <b>알려진 형상에서 생성 + 단일 실행</b> 이 가장 빠릅니다 —
        목표를 만든 형상이 복원되는지 바로 보입니다.</div>
    </div>
  </div>
</main>

<script>
let META=null, JOB=null, POLL=null;

const MODE_HELP = {
  single: '씨앗 하나로 한 번 돌립니다. 가장 빠릅니다 — 격자 121 에서 반복당 약 2 초.',
  family: '출발점을 여러 개 돌려 허용 집합 안에 든 설계를 전부 모읍니다. ' +
          '같은 곡선에 다른 형상이 여럿 나오는 것을 보여 주는 화면입니다. ' +
          '시간이 씨앗 수만큼 배로 듭니다.',
  mink:   'K=0(원형)부터 올려 가며 처음 허용 집합에 드는 모드 수를 찾습니다. ' +
          '간단한 형상으로 되면 복잡하게 갈 이유가 없다는 원칙입니다. 가장 깁니다.'
};
const MODE_PARAMS = {
  single: [['iters','반복 수',90]],
  family: [['iters','씨앗당 반복 수',90],['n_starts','씨앗 수',6]],
  mink:   [['k_max','최대 K',4],['iters_per_k','K 당 반복 수',120],
           ['n_seeds','씨앗 수 (독립 씨앗일 때만)',3]]
};
const TGT_PARAMS = {
  known: [['k_r0','목표 형상 r0 [mm]','11'],
          ['k_a','목표 a 계수 (쉼표)','0, 0, 2.2, 0'],
          ['k_lengths','목표 세그먼트 길이 [mm]','95'],
          ['k_at','목표 노즐목 면적 [mm²]','78']],
  analytic: [['f_max','최대 추력 [N]','200'],['t_burn','연소 시간 [s]','2.0'],
             ['t_rise','상승 시간 [s]','0.12'],['t_tail','하강 시간 [s]','0.30']],
};
const THROAT_PARAMS = {
  fixed: [['throat_d','목 직경 Ø [mm]','10.0']],
  band:  [['throat_dmin','목 직경 하한 Ø [mm]','8.0'],
          ['throat_dmax','목 직경 상한 Ø [mm]','12.0']],
};

function esc(s){return String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));}
function md(s){return esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')
                            .replace(/`(.+?)`/g,'<code>$1</code>');}
function f2(x){return (x===null||x===undefined||!isFinite(x))?'—':(+x).toFixed(2);}
function el(id){return document.getElementById(id);}

function rows(spec, prefix){
  return spec.map(([k,lab,def])=>
    `<div class="row"><label>${esc(lab)}</label>
     <input id="${prefix}${k}" value="${esc(def)}"></div>`).join('');
}

function renderMode(){
  const m = el('mode').value;
  el('modehelp').textContent = MODE_HELP[m];
  el('modeParams').innerHTML = rows(MODE_PARAMS[m], 'm_');
  if(m==='mink'){
    el('modeParams').innerHTML +=
      `<div class="row"><label>씨앗 전략</label><select id="m_seeding">
        <option value="warm">웜 스타트 — 직전 K 해를 물려받는다</option>
        <option value="independent">독립 씨앗 — K 마다 새로</option>
       </select></div>`;
  }
}
function renderTgt(){
  const s = el('tsrc').value, box = el('tgtParams');
  if(s==='paste'){
    box.innerHTML = `<div class="row wide"><label>t, F[, P] — 한 줄에 하나</label>
      <textarea id="t_text" placeholder="t,F,P&#10;0.00,0,1.0&#10;0.05,120,2.1&#10;..."></textarea></div>
      <div class="row"><label>압력 열을 무시</label>
      <select id="t_ignore_p"><option value="0">아니오 (있으면 사양으로)</option>
      <option value="1">예 (추력만)</option></select></div>`;
  }else if(s==='upload'){
    box.innerHTML = `<div class="row"><label>CSV 파일</label>
      <input type="file" id="t_file" accept=".csv,.txt,.tsv"></div>
      <div class="row"><label>압력 열을 무시</label>
      <select id="t_ignore_p"><option value="0">아니오</option>
      <option value="1">예 (추력만)</option></select></div>
      <div class="note" id="t_fileinfo"></div>`;
    el('t_file').onchange = async (e)=>{
      const f = e.target.files[0]; if(!f) return;
      window.__csv = await f.text();
      el('t_fileinfo').textContent =
        `${f.name} — ${window.__csv.split('\n').length} 줄 읽음`;
    };
  }else{
    box.innerHTML = rows(TGT_PARAMS[s], 't_');
  }
}
function renderThroat(){
  el('throatParams').innerHTML = rows(THROAT_PARAMS[el('throat_mode').value], '');
}
function fillDefaults(){
  const mk=(box,obj,prefix)=>{
    el(box).innerHTML = Object.entries(obj).map(([k,v])=>{
      if(typeof v === 'boolean')
        return `<div class="row"><label>${esc(k)}</label><select id="${prefix}${k}">
          <option value="1"${v?' selected':''}>예</option>
          <option value="0"${v?'':' selected'}>아니오</option></select></div>`;
      if(k==='propellant')
        return `<div class="row"><label>${esc(k)}</label><select id="${prefix}${k}">`
          + META.propellants.map(p=>`<option${p===v?' selected':''}>${p}</option>`).join('')
          + `</select></div>`;
      return `<div class="row"><label>${esc(k)}</label>
        <input id="${prefix}${k}" value="${esc(v)}"></div>`;
    }).join('');
  };
  mk('limBox', META.limits, 'lim_');
  mk('tolBox', META.tol, 'tol_');
  mk('motorBox', META.motor, 'mot_');
}

function collect(){
  const m = el('mode').value, s = el('tsrc').value;
  const num = id => el(id) ? el(id).value : undefined;
  const spec = {
    mode: m, d_outer: num('d_outer'), lengths: num('lengths'),
    n_grid: +num('n_grid'), n_modes: +num('n_modes'), r0_init: num('r0_init'),
    throat_mode: el('throat_mode').value,
    limits: {}, tol: {}, motor: {},
  };
  for(const [k] of MODE_PARAMS[m]) spec[k] = num('m_'+k);
  if(m==='mink') spec.seeding = num('m_seeding');
  for(const [k] of (THROAT_PARAMS[spec.throat_mode])) spec[k] = num(k);
  for(const k of Object.keys(META.limits)) spec.limits[k] = num('lim_'+k);
  for(const k of Object.keys(META.tol))    spec.tol[k]    = num('tol_'+k);
  for(const k of Object.keys(META.motor)){
    const v = num('mot_'+k);
    spec.motor[k] = (typeof META.motor[k] === 'boolean') ? (v==='1') : v;
  }
  const tgt = {source:s};
  if(s==='paste'){ tgt.text = num('t_text');
                   tgt.ignore_pressure = num('t_ignore_p')==='1'; }
  else if(s==='upload'){ tgt.text = window.__csv || '';
                   tgt.ignore_pressure = num('t_ignore_p')==='1'; }
  else for(const [k] of TGT_PARAMS[s]){
    const v = num('t_'+k);
    tgt[k] = (k==='k_a'||k==='k_lengths')
      ? v.split(/[,\s]+/).filter(x=>x!=='').map(Number) : v;
  }
  spec.target = tgt;
  return spec;
}

async function post(url, body){
  const r = await fetch(url,{method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  return r.json();
}

el('go').onclick = async ()=>{
  el('go').disabled = true; el('stop').disabled = false;
  el('bar').style.display='block'; el('log').style.display='block';
  el('bar').classList.add('indet'); el('bar').querySelector('i').style.width='';
  el('status').textContent = '시작하는 중...';
  const d = await post('/api/gre/run', collect());
  if(!d.ok){ el('status').textContent = '거부됨'; showErr(d.error);
             el('go').disabled=false; el('stop').disabled=true; return; }
  JOB = d.id; POLL = setInterval(poll, 1000); poll();
};
el('stop').onclick = async ()=>{ if(JOB) await post('/api/gre/cancel/'+JOB,{}); };
el('check').onclick = async ()=>{
  el('status').textContent = '입력 점검 중...';
  const d = await post('/api/gre/check', collect());
  el('status').textContent = d.ok ? '점검 완료' : '점검 실패';
  if(!d.ok){ showErr(d.error); return; }
  renderFindings(d);
};
el('reset').onclick = ()=>{ fillDefaults(); el('status').textContent='기본값으로 되돌렸습니다.'; };

async function poll(){
  const r = await fetch('/api/gre/status/'+JOB); const s = await r.json();
  el('log').textContent = (s.log||[]).join('\n');
  el('log').scrollTop = el('log').scrollHeight;
  let txt = s.status;
  if(s.last) txt += ` · 반복 ${s.last.it} · 모양 ${s.last.shape.toFixed(2)}%`
                 + ` I ${s.last.impulse.toFixed(2)}% F ${s.last.f_max.toFixed(2)}%`
                 + (s.last.in_set ? ' · 집합 안' : '');
  txt += ` · ${s.elapsed.toFixed(0)} s`;
  el('status').textContent = txt;
  if(s.frac !== null && s.frac !== undefined){
    el('bar').classList.remove('indet');
    el('bar').querySelector('i').style.width = (100*s.frac).toFixed(1)+'%';
  }
  if(s.done){
    clearInterval(POLL); el('go').disabled=false; el('stop').disabled=true;
    el('bar').classList.remove('indet');
    if(s.status==='실패'){ showErr(s.error||'알 수 없는 실패'); return; }
    if(s.status==='중단'){ el('status').textContent='중단됨'; return; }
    const rr = await fetch('/api/gre/result/'+JOB); render(await rr.json());
  }
}

function showErr(msg){
  el('out').innerHTML = '<h2>결과</h2><div class="errbox">'+esc(msg)+'</div>';
}

function findingsHtml(fs){
  if(!fs || !fs.length) return '';
  let h = '<h3 class="sec">입력 점검</h3>';
  const bad = fs.filter(f=>!f.ok);
  if(!bad.length) h += '<div class="good">검사 '+fs.length+' 항목 전부 통과.</div>';
  for(const f of bad){
    const cls = f.level==='FATAL' ? 'bad' : 'warn';
    h += `<div class="${cls}"><b>[${esc(f.code)}] ${esc(f.title)}</b>`
       + (f.detail?'<br>'+esc(f.detail):'')
       + (f.action?'<br>조치: '+esc(f.action):'') + '</div>';
  }
  return h;
}
function renderFindings(d){
  el('out').innerHTML = '<h2>결과</h2>'
    + (d.target_info?`<div class="cap">목표: ${esc(d.target_info)}</div>`:'')
    + findingsHtml(d.findings)
    + (d.runnable===false
        ? '<div class="bad"><b>이 목표로는 역설계를 시작하지 않습니다.</b></div>'
        : '<div class="good">역설계를 시작할 수 있습니다.</div>')
    + (d.charts||[]).map(c=>`<img class="chart" alt="${esc(c.title)}"
         src="data:image/png;base64,${c.png}">`).join('');
}

function render(d){
  if(d.error){ showErr(d.error); return; }
  let h = '<h2>결과</h2>';
  h += `<div class="verdict ${d.in_set?'good':'err'}">${md(d.verdict||'')}</div>`;
  if(d.target_info) h += `<div class="cap">목표: ${esc(d.target_info)}</div>`;

  if(d.spec){
    h += '<h3 class="sec">사양 — 합격 기준</h3><table class="iter"><tr>'
      + '<th class="l">항목</th><th>오차 [%]</th><th>허용 [%]</th><th>초과 [%p]</th></tr>';
    for(const [k,v] of Object.entries(d.spec)){
      h += `<tr class="${v.gap>0?'bad':'good'}"><td class="l">${esc(k)}</td>`
         + `<td>${f2(v.val)}</td><td>${f2(v.tol)}</td>`
         + `<td>${v.gap>0?f2(v.gap):'—'}</td></tr>`;
    }
    h += '</table>';
  }
  if(d.manufacturing){
    h += '<h3 class="sec">제작 제약</h3><table class="iter"><tr>'
      + '<th class="l">항목</th><th class="l">값 / 한계</th><th>판정</th></tr>';
    for(const [k,v] of Object.entries(d.manufacturing)){
      h += `<tr class="${v.ok?'good':'bad'}"><td class="l">${esc(k)}</td>`
         + `<td class="l">${esc(v.detail)}</td>`
         + `<td>${v.ok?'통과':'위반'}</td></tr>`;
    }
    h += '</table>';
  }
  if(d.design){
    const g = d.design;
    h += '<h3 class="sec">설계 변수</h3><table class="sum">';
    h += `<tr><td>r0</td><td>${f2(g.r0)} mm</td></tr>`;
    h += `<tr><td>a 계수</td><td>${g.a.map(f2).join(', ')}</td></tr>`;
    h += `<tr><td>b 계수</td><td>${g.b.map(f2).join(', ')}</td></tr>`;
    h += `<tr><td>노즐목</td><td>Ø${f2(g.d_throat_mm)} mm (${f2(g.a_throat_mm2)} mm²) <span class="pill">입력</span></td></tr>`;
    h += `<tr><td>케이스 내경 / 세그먼트</td><td>Ø${f2(g.d_outer)} · ${g.lengths.map(f2).join(', ')} mm</td></tr>`;
    h += `<tr><td>장약 질량</td><td>${f2(g.mass_g)} g</td></tr>`;
    h += `<tr><td>웹 / 꼭짓점</td><td>${f2(g.web_mm)} / ${f2(g.r_tip_mm)} mm</td></tr>`;
    h += `<tr><td>최대압 (해석)</td><td>${f2(g.P_max_sim)} MPa</td></tr>`;
    h += `<tr><td>복잡도 Σ|c_k| · 연결성분</td><td>${f2(g.complexity)} · ${g.components}</td></tr>`;
    h += '</table>';
  }
  if(d.family_rows){
    const hasP = d.family_rows.some(r=>isFinite(r.p_shape)||isFinite(r.p_max));
    h += '<h3 class="sec">해 족 — 씨앗별</h3><table class="iter"><tr>'
      + '<th>씨앗</th><th>r0</th><th>모양%</th><th>I%</th><th>F%</th>'
      + (hasP ? '<th>압력모양%</th><th>최대압%</th>' : '')
      + '<th>최악여유%p</th><th>복잡도</th><th>슬리버%</th><th>장약 g</th>'
      + '<th>목 Ø</th><th>판정</th></tr>';
    const TOL = 100*(+(d.tol_shape ?? 0.10));
    for(const r of d.family_rows){
      const errs = [r.shape, r.impulse, r.f_max, r.p_shape, r.p_max]
                     .filter(x=>isFinite(x));
      const worst = TOL - Math.max(...errs);
      h += `<tr class="${r.ok?'good':'bad'}"><td>${r.seed}</td><td>${f2(r.r0)}</td>`
         + `<td>${f2(r.shape)}</td><td>${f2(r.impulse)}</td><td>${f2(r.f_max)}</td>`
         + (hasP ? `<td>${f2(r.p_shape)}</td><td>${f2(r.p_max)}</td>` : '')
         + `<td><b>${f2(worst)}</b></td>`
         + `<td>${f2(r.complexity)}</td><td>${f2(r.sliver)}</td><td>${f2(r.mass)}</td>`
         + `<td>${f2(r.d_throat)}</td><td>${r.ok?'합격':'탈락'}</td></tr>`;
    }
    h += '<div class="cap">최악여유 = 허용 오차 − 가장 큰 사양 오차. '
       + '<b>사양 5 개를 전부 보고 고른다</b> — 압력 두 항이 빠지면 여유가 '
       + '실제보다 넓어 보인다.</div>';
    h += '</table>';
    if(d.spread_mm) h += `<div class="cap">합격한 설계들 사이의 형상 거리 `
      + `(회전 정렬 후): 최소 ${f2(d.spread_mm.min)} · 중앙 ${f2(d.spread_mm.median)}`
      + ` · 최대 ${f2(d.spread_mm.max)} mm. <b>해는 하나가 아니다</b> — 이 표에서 고른다.</div>`;
  }
  if(d.family_table) h += '<pre class="table">'+esc(d.family_table)+'</pre>';
  if(d.mink){
    const k = d.mink;
    h += '<h3 class="sec">K 탐색 — 가장 작은 모드 수</h3>';
    h += `<div class="${k.feasible?'good':'bad'}">`
       + (k.feasible ? `<b>K = ${k.k_chosen}</b> 에서 통과. ` : '')
       + md(k.note||'') + '</div>';
    if(k.k_table && k.k_table.length){
      h += '<table class="iter"><tr><th>K</th><th>모양%</th><th>최대 초과 %p</th>'
        + '<th class="l">걸린 항목</th><th>반복(누적)</th><th class="l">종료 사유</th></tr>';
      for(const r of k.k_table){
        const sel = (k.feasible && r.K===k.k_chosen && !r.recheck);
        h += `<tr class="${sel?'final':(r.worst_gap_pp>0?'':'good')}">`
           + `<td>${r.K}${r.recheck?' ↺':''}</td><td>${f2(r.rms_shape_pct)}</td>`
           + `<td>${r.worst_gap_pp>0?f2(r.worst_gap_pp):'—'}</td>`
           + `<td class="l">${esc(r.binding||'—')}</td>`
           + `<td>${r.iters_used??'—'}${r.iters_cum!==undefined?' ('+r.iters_cum+')':''}</td>`
           + `<td class="l">${esc(r.stop_reason||'')}</td></tr>`;
      }
      h += '</table><div class="cap">↺ 는 예산 재확인 행입니다 — K 가 아니라 '
        + '<b>반복 예산</b>이 통과를 만든 것은 아닌지 K=0 에 같은 예산을 주고 '
        + '다시 본 결과입니다.</div>';
    }
  }

  if(d.design && d.in_set && JOB){
    h += '<h3 class="sec">CAD 내보내기 — 몰드 일습 9 종</h3>'
      + '<div class="cap">복원된 포트로 그레인·코어핀·피스톤·엔드플레이트·'
      + '공구·접시와 <b>2분할 실리콘 주형 마스터</b>(plate + top)를 STEP·STL 로 '
      + '씁니다. 고정 규칙: 마스터 높이 = <b>그레인 길이 + 10</b> · 지관 벽 '
      + '<b>2.0 고정</b> · Top 기둥 = <b>바깥 벽 + 5</b>.</div>'
      + '<div class="row"><label>출력 폴더</label>'
      + '<input id="cadout" placeholder="(기본값 out/cad_' + esc(JOB) + ')"></div>'
      + '<div class="row"><label>STEP 도 함께</label><select id="cadstep">'
      + '<option value="1">예 (약 20 초 더 걸립니다)</option>'
      + '<option value="0">아니오 — STL 만</option></select></div>'
      + '<button class="btn" id="cadgo">내보내기</button>'
      + '<div id="cadstatus" style="font-size:12px;color:var(--accent);'
      + 'margin-top:.5rem;min-height:16px"></div><div id="cadout_box"></div>';
  }
  h += findingsHtml(d.findings);
  for(const c of (d.charts||[]))
    h += `<img class="chart" alt="${esc(c.title)}" src="data:image/png;base64,${c.png}">`;
  el('out').innerHTML = h;
  if(el('cadgo')) el('cadgo').onclick = runExport;
}

async function runExport(){
  const b = el('cadgo'), st = el('cadstatus');
  b.disabled = true; st.textContent = 'CAD 생성 중... (STEP 포함이면 20 초쯤)';
  try{
    const d = await post('/api/gre/export', {
      job: JOB, outdir: el('cadout').value, step: el('cadstep').value === '1'});
    if(!d.ok){ st.textContent = '실패';
      el('cadout_box').innerHTML = '<div class="errbox">'+esc(d.error)+'</div>';
      b.disabled = false; return; }
    st.textContent = `완료 — ${d.n_files} 개 파일`;
    let h = '<table class="sum">';
    for(const [k,v] of Object.entries(d.rules))
      h += `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`;
    h += `<tr><td>출력 폴더</td><td>${esc(d.outdir)}</td></tr></table>`;
    h += '<table class="iter"><tr><th class="l">부품</th><th>체적 mm³</th>'
       + '<th>수밀</th><th class="l">파일</th></tr>';
    for(const q of d.parts)
      h += `<tr class="${q.watertight?'good':'bad'}"><td class="l">${esc(q.name)}</td>`
         + `<td>${q.volume_mm3.toFixed(0)}</td><td>${q.watertight?'✔':'✘'}</td>`
         + `<td class="l">${esc(q.files)}</td></tr>`;
    h += '</table>';
    if(d.propellant)
      h += `<div class="cap">추진제 길이 ${f2(d.propellant.length_mm)} mm · `
         + `체적 ${d.propellant.volume_mm3.toFixed(0)} mm³ · `
         + `넣을 질량 <b>${f2(d.propellant.mass_target_g)} g</b> `
         + `(${(100*d.propellant.pack_frac).toFixed(0)} % 충전)</div>`;
    for(const w of (d.warnings||[])) h += '<div class="warn">⚠ '+esc(w)+'</div>';
    el('cadout_box').innerHTML = h;
  }catch(e){ st.textContent = '요청 실패: ' + e; }
  b.disabled = false;
}

el('mode').onchange = renderMode;
el('tsrc').onchange = renderTgt;
el('throat_mode').onchange = renderThroat;
el('theme').onclick = ()=>{const e=document.documentElement;
  e.dataset.theme = e.dataset.theme==='g100'?'g10':'g100';};

(async ()=>{
  META = await (await fetch('/api/gre/meta')).json();
  renderMode(); renderTgt(); renderThroat(); fillDefaults();
})();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="그레인 형상 역설계 웹 UI",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--port", type=int, default=5080)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="서버 없이 계산 경로를 검증하고 끝낸다")
    args = ap.parse_args()

    if args.selftest:
        return 0 if selftest() else 1

    try:
        from flask import Flask, jsonify, request
    except ImportError:
        print("\n❌ Flask 가 필요합니다:  pip install flask")
        return 1

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/api/gre/meta")
    def meta():
        return jsonify(defaults())

    @app.route("/api/gre/check", methods=["POST"])
    def check():
        spec = request.get_json(force=True) or {}
        try:
            from gre.validate.input_check import check_inputs
            motor = build_motor(spec)
            limits = build_limits(spec)
            lengths = parse_lengths(spec.get("lengths", "95"))
            target, tinfo = build_target(spec.get("target") or {}, motor)
            at = (math.pi * float(spec["throat_d"]) ** 2 / 4.0
                  if spec.get("throat_mode", "fixed") == "fixed" else None)
            rep = check_inputs(target, motor, limits, tuple(lengths),
                               a_throat=at)
            return jsonify(_safe({
                "ok": True, "runnable": bool(rep.runnable), "target_info": tinfo,
                "findings": _findings(rep),
                "charts": charts_target_only(target)}))
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"{exc.__class__.__name__}: {exc}"})

    @app.route("/api/gre/run", methods=["POST"])
    def run():
        spec = request.get_json(force=True) or {}
        live = [j for j in JOBS.values() if not j.snapshot()["done"]]
        if live:
            return jsonify({"ok": False, "error":
                            f"이미 실행 중인 작업이 있다 ({live[0].id}). "
                            f"끝나거나 중단될 때까지 기다리십시오 — "
                            f"최적화는 CPU 를 다 쓰므로 동시에 돌리면 "
                            f"둘 다 느려집니다."})
        try:
            spec["_iters_total"] = estimate_iters(spec)
            jid = f"job{len(JOBS)+1}-{int(time.time())%100000}"
            job = Job(jid, spec)
            JOBS[jid] = job
            threading.Thread(target=run_job, args=(job,), daemon=True).start()
            return jsonify({"ok": True, "id": jid,
                            "iters_total": spec["_iters_total"]})
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"{exc.__class__.__name__}: {exc}"})

    @app.route("/api/gre/status/<jid>")
    def status(jid):
        job = JOBS.get(jid)
        if not job:
            return jsonify({"status": "없음", "done": True,
                            "error": f"그런 작업이 없다: {jid}"})
        return jsonify(job.snapshot())

    @app.route("/api/gre/result/<jid>")
    def result(jid):
        job = JOBS.get(jid)
        if not job:
            return jsonify({"error": f"그런 작업이 없다: {jid}"})
        if job.error:
            return jsonify({"error": job.error})
        return jsonify(_safe(job.result or {"error": "결과가 아직 없다"}))

    @app.route("/api/gre/export", methods=["POST"])
    def export_cad():
        body = request.get_json(force=True) or {}
        job = JOBS.get(str(body.get("job") or ""))
        if not job:
            return jsonify({"ok": False, "error":
                            "그 작업을 못 찾았다 — 역설계를 먼저 끝내십시오."})
        try:
            return jsonify(_safe(export_cad_for(job, body)))
        except Exception as exc:
            return jsonify({"ok": False,
                            "error": f"{exc.__class__.__name__}: {exc}"})

    @app.route("/api/gre/cancel/<jid>", methods=["POST"])
    def cancel(jid):
        job = JOBS.get(jid)
        if job:
            job.cancel.set()
        return jsonify({"ok": True})

    if not args.no_browser:
        import webbrowser
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")
        ).start()

    print(f"\n🌐 그레인 역설계 UI 실행 중  →  http://127.0.0.1:{args.port}/")
    print("   종료: Ctrl+C\n")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n  서버 종료됨.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
