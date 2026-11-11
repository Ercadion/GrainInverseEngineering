"""다각형(+구멍) → **수밀(watertight) 프리즘 메쉬**, 그리고 STL 쓰기.

의존성을 최소로 둔다 — 삼각분할만 `mapbox_earcut` 을 쓰고 나머지는 numpy 다.
STEP 은 `gre.export.step` 에서 `cadquery` 가 있을 때만 다룬다.

⚠ **수밀성과 체적을 스스로 검사한다.** 슬라이서는 수밀이 아닌 STL 에서 조용히
   이상한 것을 출력한다. 내보낸 메쉬가
     (a) 모든 모서리를 정확히 두 번 쓰는가 (수밀)
     (b) 발산 정리로 잰 체적이 해석 체적과 맞는가
   를 확인하지 않으면, 출력물이 설계와 다른 것을 **인쇄가 끝난 뒤에** 알게 된다.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

try:                                        # 구멍 있는 다각형 삼각분할
    import mapbox_earcut as _earcut
    _HAVE_EARCUT = True
except Exception:                           # pragma: no cover
    _HAVE_EARCUT = False

try:                                        # 제약 들로네 (예비 경로)
    import triangle as _tri_cdt
    _HAVE_CDT = True
except Exception:                           # pragma: no cover
    _HAVE_CDT = False


class TriangulationError(RuntimeError):
    """삼각분할이 입력 윤곽을 닫지 못했다 — 조용히 넘기면 안 되는 상황."""


@dataclass
class Mesh:
    """삼각형 메쉬. `v` (V,3) [mm], `f` (F,3) 정수 인덱스, 바깥 방향 법선."""

    v: np.ndarray
    f: np.ndarray
    name: str = "part"

    # ------------------------------------------------------------- 자가 검사
    @property
    def volume(self) -> float:
        """부호 있는 체적 [mm³] — 발산 정리 (∑ (a×b)·c / 6)."""
        a, b, c = (self.v[self.f[:, 0]], self.v[self.f[:, 1]], self.v[self.f[:, 2]])
        return float(np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0)

    @property
    def area(self) -> float:
        """표면적 [mm²]."""
        a, b, c = (self.v[self.f[:, 0]], self.v[self.f[:, 1]], self.v[self.f[:, 2]])
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    def watertight(self, tol: float = 1e-6) -> tuple:
        """(수밀인가, 진단 문자열). 모든 모서리가 **정확히 두 번** 쓰여야 한다.

        ⚠⚠ **꼭짓점은 인덱스가 아니라 좌표로 병합해서 본다.**

          종전 판은 `self.f` 의 인덱스만 봤다. 그러면 `union_z` 로 쌓은 두 껍질이
          **통과한다** — 접촉면에서 좌표는 같아도 인덱스가 달라 각각 독립적으로
          닫혀 있기 때문이다. 그런데 그건 솔리드 하나가 아니라 **내부에 면이 있는
          껍질 두 개**다. 실제로 `core_mold` 가 그 상태로 통과하고 있었고, 쓰인
          STL 을 다시 읽어 좌표로 병합해 보고서야 블록 외곽 모서리가 **네 번**
          쓰인 것이 드러났다. 그래서 검사를 좌표 기준으로 올린다.

        `tol` 은 병합 격자 크기 [mm] 다 (좌표를 그 눈금으로 반올림해 묶는다).
        """
        q = np.round(np.asarray(self.v, dtype=float) / float(tol)).astype(np.int64)
        uq, inv = np.unique(q, axis=0, return_inverse=True)
        f = np.asarray(inv).ravel()[self.f]
        deg = int((f[:, 0] == f[:, 1]).sum() + (f[:, 1] == f[:, 2]).sum()
                  + (f[:, 2] == f[:, 0]).sum())
        e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        key = np.sort(e, axis=1)
        _, cnt = np.unique(key, axis=0, return_counts=True)
        bad = int((cnt != 2).sum())
        # 방향 일관성: 각 모서리가 양방향으로 한 번씩 쓰였는가
        _, ucnt = np.unique(e, axis=0, return_counts=True)
        dup = int((ucnt != 1).sum())
        ok = bad == 0 and dup == 0 and deg == 0
        return ok, (f"꼭짓점 {len(self.v)} → 좌표 병합 {len(uq)} | 모서리 "
                    f"{len(key)} 개, 두 번 안 쓰인 모서리 {bad} 개, 방향 중복 "
                    f"{dup} 개, 퇴화 삼각형 {deg} 개")

    def bounds(self) -> np.ndarray:
        return np.array([self.v.min(axis=0), self.v.max(axis=0)])


def _ring_edge_set(rings) -> set:
    """링들의 모서리를 (정렬된 인덱스 쌍) 집합으로."""
    out, base = set(), 0
    for r in rings:
        n = len(r)
        for k in range(n):
            i, j = base + k, base + (k + 1) % n
            out.add((i, j) if i < j else (j, i))
        base += n
    return out


def _tri_closes(tri: np.ndarray, rings) -> bool:
    """삼각분할이 **입력 윤곽만을** 경계로 갖는가.

    한 번만 쓰인 모서리의 집합이 링 모서리 집합과 정확히 같아야 한다. 이게 성립
    하지 않으면 뚜껑에 틈이나 겹침이 있고, 프리즘은 수밀이 되지 않는다. earcut 은
    구멍이 **여러 개이고 서로 일직선으로 늘어선** 배치(= 시편 빗 모양)에서 이걸
    깨뜨린다 — 그래서 믿지 않고 매번 검사한다.
    """
    if len(tri) == 0:
        return False
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(e, axis=1)
    u, cnt = np.unique(key, axis=0, return_counts=True)
    if (cnt > 2).any():
        return False
    once = set(map(tuple, u[cnt == 1]))
    return once == _ring_edge_set(rings)


def _point_in_ring(ring, pt) -> bool:
    """`pt` 가 닫힌 고리 안에 있는가 — 광선 교차(짝/홀) 판정."""
    p = np.asarray(ring, dtype=float)
    x, y = float(pt[0]), float(pt[1])
    x0, y0 = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    straddle = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return bool((straddle & (x < xin)).sum() % 2 == 1)


def _hole_point(ring: np.ndarray) -> np.ndarray:
    """링 **내부**의 한 점 (CDT 가 구멍을 지우는 데 쓴다)."""
    try:                                    # 오목한 링에서도 안전한 경로
        from shapely.geometry import Polygon
        p = Polygon(np.asarray(ring, dtype=float)).representative_point()
        return np.array([p.x, p.y])
    except Exception:                       # pragma: no cover
        return np.asarray(ring, dtype=float).mean(axis=0)


def _triangulate_cdt(rings) -> np.ndarray:
    """제약 들로네 삼각분할 — earcut 이 윤곽을 못 닫을 때의 예비 경로."""
    if not _HAVE_CDT:                        # pragma: no cover
        raise TriangulationError(
            "earcut 이 윤곽을 닫지 못했고 예비 경로가 없다: pip install triangle")
    verts = np.vstack(rings)
    segs, base = [], 0
    for r in rings:
        n = len(r)
        i = base + np.arange(n)
        segs.append(np.column_stack([i, base + (np.arange(n) + 1) % n]))
        base += n
    arg = {"vertices": verts, "segments": np.vstack(segs)}
    if len(rings) > 1:
        arg["holes"] = np.array([_hole_point(r) for r in rings[1:]])
    res = _tri_cdt.triangulate(arg, "p")     # 'p' = PSLG, 점 추가 없음
    vo = np.asarray(res["vertices"], dtype=float)
    if len(vo) != len(verts) or not np.allclose(vo, verts):   # pragma: no cover
        raise TriangulationError(
            f"CDT 가 점을 바꿨다 (입력 {len(verts)} → 출력 {len(vo)}) — "
            f"윤곽이 자기교차하는지 확인할 것")
    return np.asarray(res["triangles"], dtype=np.int64).reshape(-1, 3)


def _triangulate(outer: np.ndarray, holes=()) -> np.ndarray:
    """구멍 있는 단면 → 삼각형 인덱스. **결과를 스스로 검사한다.**

    earcut 을 먼저 쓰고, 한 번만 쓰인 모서리가 입력 링과 다르면 CDT 로 다시 한다.
    둘 다 실패하면 예외를 던진다 — 틈 있는 뚜껑을 조용히 내보내는 것보다 낫다.
    """
    rings = [np.asarray(outer, dtype=np.float64)]
    rings += [np.asarray(h, dtype=np.float64) for h in holes]
    verts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    if _HAVE_EARCUT:
        tri = np.asarray(_earcut.triangulate_float64(verts, ends),
                         dtype=np.int64).reshape(-1, 3)
        if _tri_closes(tri, rings):
            return tri
    elif not _HAVE_CDT:                      # pragma: no cover
        raise ImportError("mapbox_earcut 또는 triangle 이 필요하다")
    tri = _triangulate_cdt(rings)
    if not _tri_closes(tri, rings):          # pragma: no cover
        raise TriangulationError(
            f"링 {len(rings)} 개 · 점 {len(verts)} 개 단면을 닫지 못했다")
    return tri


def _orient_up(tri: np.ndarray, flat2d: np.ndarray) -> np.ndarray:
    """모든 삼각형의 법선을 +z 로 맞춘다 (2차원 부호 넓이로 판정)."""
    a, b, c = flat2d[tri[:, 0]], flat2d[tri[:, 1]], flat2d[tri[:, 2]]
    s = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
         - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    out = tri.copy()
    out[s < 0] = out[s < 0][:, ::-1]
    return out


def _wall(i_bot: int, i_top: int, n: int) -> np.ndarray:
    """링 하나의 옆면 — 아래 인덱스 기점 `i_bot`, 위 기점 `i_top`.

    법선 방향은 링의 회전 방향이 정한다 (t̂ × ẑ): 바깥 외곽은 CCW, 구멍·포켓은 CW
    로 넣으면 둘 다 **고체 바깥쪽**을 향한다.
    """
    i = np.arange(n)
    j = (i + 1) % n
    a, b = i_bot + i, i_bot + j
    c, d = i_top + i, i_top + j
    return np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])


def _ring_ccw(pts: np.ndarray) -> np.ndarray:
    """반시계 방향으로 맞춘다 (신발끈 넓이의 부호)."""
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    a2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return p if a2 > 0 else p[::-1]


def prism(outer: np.ndarray, z0: float, z1: float, holes=(),
          name: str = "part") -> Mesh:
    """단면을 z 방향으로 뽑은 수밀 프리즘.

    `outer` 는 바깥 윤곽, `holes` 는 구멍들 (각각 (N,2) [mm]). 방향은 알아서
    맞춘다 — 바깥은 CCW, 구멍은 CW 여야 삼각분할과 벽면 법선이 바깥을 향한다.
    """
    zb, zt = (float(z0), float(z1)) if z1 >= z0 else (float(z1), float(z0))
    out = _ring_ccw(outer)
    hs = [_ring_ccw(h)[::-1] for h in holes]          # 구멍은 CW

    tri = _triangulate(out, hs)
    rings = [out] + hs
    flat = np.vstack(rings)
    n2d = len(flat)
    tri = _orient_up(tri, flat)

    v = np.vstack([np.column_stack([flat, np.full(n2d, zb)]),
                   np.column_stack([flat, np.full(n2d, zt)])])
    faces = [tri[:, ::-1]]                            # 바닥: 법선 −z → 뒤집는다
    faces.append(tri + n2d)                           # 천장: 법선 +z

    base = 0
    for ring in rings:                                # 옆면
        faces.append(_wall(base, base + n2d, len(ring)))
        base += len(ring)
    return Mesh(v=v, f=np.vstack(faces).astype(np.int64), name=name)


@dataclass
class Pocket:
    """위에서 파낸 홈. `depth` 는 **자기 부모 바닥면 기준** 깊이 [mm].

    `children` 로 홈 안의 홈을 중첩할 수 있다 (예: 둘레 벽 안의 슬롯들). 자식은
    부모 윤곽 안에 완전히 들어 있어야 하고 형제끼리 겹치지 않아야 한다 —
    `pocket_solid` 는 불리언을 하지 않고 면을 직접 꿰매므로, 겹치면 수밀 검사가
    잡아낸다.
    """

    poly: np.ndarray
    depth: float
    children: list = field(default_factory=list)
    name: str = ""


def pocket_solid(outer: np.ndarray, thickness: float, pockets=(),
                 name: str = "part") -> Mesh:
    """위면에서 홈을 파낸 **단일 수밀 셸**.

    `prism` 을 z 로 여러 장 쌓아 합치면 맞닿는 면이 안쪽에 그대로 남아 (각 장은
    닫혀 있어도) 슬라이서가 보기엔 겹친 껍질 여러 개가 된다. 여기서는 필요한
    면만 꿰맨다: 바닥 한 장, 둘레 한 겹, 위면(홈 자리는 뚫림), 그리고 홈마다
    옆면 한 겹 + 바닥 한 장. 내부 면이 없으므로 셸이 하나다.
    """
    V, F = [], []

    def add_ring(pts, z):
        i0 = len(V)
        V.extend((float(p[0]), float(p[1]), float(z)) for p in pts)
        return i0

    def cap_up(ring, kids, i_ring, i_kids, z):
        """`ring` 위에 `kids` 를 구멍으로 두고 +z 법선 뚜껑 한 장."""
        tri = _triangulate(ring, kids)
        flat = np.vstack([ring] + kids) if kids else np.asarray(ring, float)
        tri = _orient_up(tri, flat)
        remap = np.concatenate(
            [np.arange(len(ring)) + i_ring]
            + [np.arange(len(k)) + ik for k, ik in zip(kids, i_kids)]
        ).astype(np.int64)
        return remap[tri]

    def emit(ring_cw, depth, children, z_surf, i_surf):
        z_f = z_surf - float(depth)
        i_floor = add_ring(ring_cw, z_f)
        F.append(_wall(i_floor, i_surf, len(ring_cw)))
        kids = [_ring_ccw(c.poly)[::-1] for c in children]
        i_kids = [add_ring(k, z_f) for k in kids]
        F.append(cap_up(ring_cw, kids, i_floor, i_kids, z_f))
        for c, k, ik in zip(children, kids, i_kids):
            emit(k, c.depth, c.children, z_f, ik)

    out = _ring_ccw(outer)
    t = float(thickness)
    i_bot = add_ring(out, 0.0)
    i_top = add_ring(out, t)
    F.append(_wall(i_bot, i_top, len(out)))
    F.append(cap_up(out, [], i_bot, [], 0.0)[:, ::-1])       # 바닥 (−z)

    tops = [_ring_ccw(p.poly)[::-1] for p in pockets]
    i_tops = [add_ring(r, t) for r in tops]
    F.append(cap_up(out, tops, i_top, i_tops, t))
    for p, r, ir in zip(pockets, tops, i_tops):
        emit(r, p.depth, p.children, t, ir)

    return Mesh(v=np.array(V, dtype=float),
                f=np.vstack(F).astype(np.int64), name=name)


def _ring_area(pts) -> float:
    """닫힌 고리의 **부호 없는** 면적 [mm²] — 어느 쪽이 큰 고리인지 판정용."""
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def stepped_plate(bands, name: str = "part") -> Mesh:
    """z 로 쌓인 층들을 **하나의 수밀 셸**로 꿰맨다 — 단차 있는 판/슬리브.

    `bands` 는 z 오름차순으로 `(outer, holes, z0, z1)` 들이다. `holes` 는 층마다
    **같은 개수**여야 하고, 같은 자리의 구멍은 층 사이에서 바뀌어도 된다 (씰 띠 ↔
    헐거운 띠). 없는 구멍은 `None` 으로 적는다 — 그 경계에서 구멍이 막힌다.

    ⚠ **`prism` 을 `union_z` 로 쌓는 것과 다르다.** 쌓으면 맞닿는 면이 안쪽에
      그대로 남아, 각 장은 닫혀 있어도 전체는 **솔리드의 경계면이 아니다** (내부에
      면이 있다). 여기서는 경계에서 바뀐 만큼만 수평 고리로 덮으므로 셸이 하나다.
      같은 자리의 윤곽이 두 층에서 **똑같으면 꼭짓점 고리를 공유**한다 — 복제하면
      그 고리의 모서리가 한 번만 쓰여 `watertight()` 가 떨어진다.

    ⚠⚠ **`watertight()` 로는 쌓기와 구별되지 않는다.**
      그 검사는 "모든 모서리가 정확히 두 번" 이라는 **국소** 검사인데, 쌓은 두 장은
      꼭짓점 집합이 서로 달라 각각 독립적으로 닫혀 있다 — 그래서 **둘 다 통과한다.**
      차이는 **표면적**에 나타난다: 쌓기는 가려진 접촉면을 두 번 포함하므로
      `면적(쌓기) − 면적(단일 셸) = 2 × 접촉 면적` 이다. 시험은 그것을 본다.

    전제: 한 경계에서 바뀌는 두 윤곽은 **포함 관계**여야 한다 (한쪽이 다른 쪽을
    완전히 품는다). 오프셋으로 만든 띠들은 모두 그렇다. 부분적으로 겹치면
    여기서는 잡지 않고 `watertight()` 가 잡는다.
    """
    nb = len(bands)
    if nb == 0:
        raise ValueError("bands 가 비었다")
    B = []
    for band in bands:
        outer, holes, z0, z1 = band[:4]
        grv = list(band[4]) if len(band) > 4 else []
        o = _ring_ccw(outer)
        hs = [None if h is None else _ring_ccw(h)[::-1] for h in holes]
        # 환형 홈: 바깥 고리는 구멍처럼 CW, 안쪽 고리는 **섬**이라 바깥처럼 CCW
        gs = [None if g is None else (_ring_ccw(g[0])[::-1], _ring_ccw(g[1]))
              for g in grv]
        if float(z1) <= float(z0):
            raise ValueError(f"층 두께가 0 이하다 ({z0} → {z1})")
        B.append((o, hs, float(z0), float(z1), gs))
    nh = len(B[0][1])
    ng = len(B[0][4])
    if any(len(b[1]) != nh for b in B):
        raise ValueError("층마다 구멍 개수가 같아야 한다 (없으면 None)")
    if any(len(b[4]) != ng for b in B):
        raise ValueError("층마다 환형 홈 개수가 같아야 한다 (없으면 None)")
    for i in range(nb - 1):
        if abs(B[i][3] - B[i + 1][2]) > 1e-9:
            raise ValueError(f"층 {i} 과 {i+1} 이 z 로 붙어 있지 않다")

    V, F = [], []

    def add(pts, z):
        i0 = len(V)
        V.extend((float(p[0]), float(p[1]), float(z)) for p in pts)
        return i0

    def col(b, c):
        return b[0] if c == 0 else b[1][c - 1]

    def same(a, b):
        return (a is not None and b is not None and len(a) == len(b)
                and np.allclose(a, b))

    def gsame(a, b):
        return (a is not None and b is not None
                and same(a[0], b[0]) and same(a[1], b[1]))

    # ══════════════════════════════════════════════════════════════════
    # 꼭짓점 고리 — **z 레벨마다, 윤곽마다 하나씩**
    #
    # ⚠⚠ 종전에는 "같은 **열**의 윤곽이 같으면 공유" 였다. 그래서 열이 달라지면
    #   (홈의 바깥 고리 ↔ 위층의 구멍, 홈의 안쪽 섬 ↔ 위층의 바깥 윤곽) 같은 자리를
    #   고리 **둘**이 따로 쓰고, 그 이음매로 수밀이 깨졌다. 열을 보지 않고
    #   **기하로** 묶는다.
    # ══════════════════════════════════════════════════════════════════
    def rings_of(b):
        """그 층이 쓰는 고리들 — 저장된 방향 그대로 (바깥/섬 CCW, 구멍/홈 CW)."""
        out = [b[0]]
        out += [h for h in b[1] if h is not None]
        for g in b[4]:
            if g is not None:
                out += [g[0], g[1]]
        return out

    def key(pts):
        p = np.asarray(pts, dtype=float)
        x, y = p[:, 0], p[:, 1]
        cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = 0.5 * float(cr.sum())
        if abs(a) < 1e-12:
            raise ValueError("면적 0 인 고리")
        cx = float(((x + np.roll(x, -1)) * cr).sum()) / (6.0 * a)
        cy = float(((y + np.roll(y, -1)) * cr).sum()) / (6.0 * a)
        return (len(p), round(abs(a), 7), round(cx, 7), round(cy, 7))

    nz = nb + 1
    zs = [B[0][2]] + [b[3] for b in B]
    lvl = [dict() for _ in range(nz)]          # 레벨 → {key: (pts, i0)}
    for i, b in enumerate(B):
        for r in rings_of(b):
            k = key(r)
            for L in (i, i + 1):
                if k not in lvl[L]:
                    lvl[L][k] = (r, add(r, zs[L]))
                elif len(lvl[L][k][0]) != len(r):     # pragma: no cover
                    raise ValueError("같은 키인데 점 수가 다르다")

    # ── 옆면: 층마다 자기 고리를 아래 레벨 → 위 레벨로 잇는다
    for i, b in enumerate(B):
        for r in rings_of(b):
            k = key(r)
            F.append(_wall(lvl[i][k][1], lvl[i + 1][k][1], len(r)))

    def fwd(pts, i):
        p = np.asarray(pts, dtype=float)
        return p, np.arange(len(p)) + int(i)

    def rev(pts, i):
        p = np.asarray(pts, dtype=float)
        return p[::-1], (len(p) - 1 - np.arange(len(p))) + int(i)

    def _signed(pts):
        p = np.asarray(pts, dtype=float)
        return float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], -1)))

    def oriented(L, ring_pts, want_ccw):
        """레벨 `L` 의 그 고리를 원하는 회전 방향으로 (점, 인덱스) 반환."""
        k = key(ring_pts)
        if k not in lvl[L]:                           # pragma: no cover
            raise ValueError("경계 면이 알 수 없는 고리를 썼다")
        pts, i0 = lvl[L][k]
        is_ccw = _signed(pts) > 0
        return fwd(pts, i0) if is_ccw == want_ccw else rev(pts, i0)

    def cap(entries, upward):
        """entries[0] = 바깥(CCW), 나머지 = 구멍(CW). 수평 뚜껑 한 장."""
        tri = _triangulate(entries[0][0], [e[0] for e in entries[1:]])
        flat = np.vstack([e[0] for e in entries])
        tri = _orient_up(tri, flat)
        remap = np.concatenate([e[1] for e in entries]).astype(np.int64)
        t = remap[tri]
        F.append(t if upward else t[:, ::-1])

    # ══════════════════════════════════════════════════════════════════
    # 수평 면 — **2D 불리언으로 한 번에** 구한다
    #
    # ⚠⚠ 종전에는 열마다 "구멍이 커졌나/홈이 사라졌나" 를 따로 보고 면을 깔았다.
    #   그 방식은 **빈 자리끼리 합쳐질 때** 반드시 틀린다 — 특례를 세 번
    #   붙였고(삼켜지는 링, 모양 바뀌는 홈, 솟는 기둥) 그때마다 다른 데가 깨졌다.
    #   경계에서 드러나는 면은 정의상 **아래 재료 ∖ 위 재료**(위로)와 그 반대
    #   (아래로)다. 그걸 그대로 계산한다 — 특례가 없다.
    # ══════════════════════════════════════════════════════════════════
    try:
        from shapely.geometry import Polygon as _SPoly
    except Exception as exc:                          # pragma: no cover
        raise RuntimeError(
            "stepped_plate 의 경계 면 계산에 shapely 가 필요하다 — "
            "pip install shapely") from exc

    def region(b):
        """그 층의 **재료** 2D 영역."""
        g = _SPoly(b[0], [h for h in b[1] if h is not None])
        for gr in b[4]:
            if gr is not None:
                g = g.difference(_SPoly(gr[0], [gr[1]]))
        return g

    regs = [region(b) for b in B]

    def emit(geom, L, upward):
        if geom.is_empty:
            return
        for poly in getattr(geom, "geoms", [geom]):
            if poly.is_empty or poly.area <= 1e-12:
                continue
            ext = oriented(L, np.asarray(poly.exterior.coords[:-1]), True)
            ins = [oriented(L, np.asarray(r.coords[:-1]), False)
                   for r in poly.interiors]
            cap([ext] + ins, upward)

    emit(regs[0], 0, False)                            # 맨 아래 바닥 (−z)
    emit(regs[-1], nb, True)                           # 맨 위 천장 (+z)
    for i in range(nb - 1):
        emit(regs[i].difference(regs[i + 1]), i + 1, True)
        emit(regs[i + 1].difference(regs[i]), i + 1, False)

    return Mesh(v=np.array(V, dtype=float),
                f=np.vstack(F).astype(np.int64), name=name)


def revolve(poly_rz: np.ndarray, n_theta: int = 180, name: str = "part") -> Mesh:
    """축 대칭 회전체 — (r, z) 닫힌 다각형을 z 축 둘레로 돌린다.

    `poly_rz` 는 자오면(meridional) 단면의 **닫힌** 윤곽이고 모든 `r > 0` 이어야
    한다 (축에 닿으면 극점에서 축퇴 삼각형이 생긴다). 노즐처럼 속이 빈 회전체가
    여기 해당한다 — 결과는 수밀이고 위상적으로 도넛(genus 1)이다.

    ⚠ **면 개수가 아니라 체적으로 검산한다.** 정확한 체적은 파푸스 정리로
       `V = 2π·r̄·A` (r̄ 은 단면 도심 반경). 분할된 메쉬는 그보다 조금 작으며,
       그 차이가 곧 θ 방향 테셀레이션 오차다 — `pappus_volume` 으로 같이 보고한다.
    """
    p = np.asarray(poly_rz, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("poly_rz 는 (N, 2) 여야 한다")
    if float(p[:, 0].min()) <= 0.0:
        raise ValueError(f"모든 r 이 0 보다 커야 한다 (최소 {p[:, 0].min():.4g})")
    n = len(p)
    m = int(n_theta)
    th = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    # 자오선 방향이 체적 부호를 정한다 — 양수가 되도록 맞춘다
    x, y = p[:, 0], p[:, 1]
    if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        p = p[::-1]
    r, z = p[:, 0], p[:, 1]
    v = np.empty((m * n, 3), dtype=float)
    v[:, 0] = np.outer(np.cos(th), r).ravel()
    v[:, 1] = np.outer(np.sin(th), r).ravel()
    v[:, 2] = np.tile(z, m)

    i = np.arange(n)
    j = (i + 1) % n
    k = np.arange(m)[:, None] * n
    l = ((np.arange(m) + 1) % m)[:, None] * n
    a, b = (k + i).ravel(), (k + j).ravel()      # 같은 θ
    c, d = (l + i).ravel(), (l + j).ravel()      # 다음 θ
    f = np.vstack([np.column_stack([a, d, b]), np.column_stack([a, c, d])])
    return Mesh(v=v, f=f.astype(np.int64), name=name)


def pappus_volume(poly_rz: np.ndarray) -> float:
    """파푸스 정리로 잰 회전체의 **정확한** 체적 [mm³] — `revolve` 의 검산 기준."""
    p = np.asarray(poly_rz, dtype=float)
    r, z = p[:, 0], p[:, 1]
    rn, zn = np.roll(r, -1), np.roll(z, -1)
    cross = r * zn - rn * z
    area = 0.5 * float(cross.sum())
    if area == 0.0:
        return 0.0
    r_bar = float((((r + rn) * cross).sum()) / 6.0) / area
    return float(2.0 * np.pi * r_bar * abs(area))      # 회전 방향과 무관


def union_z(parts) -> Mesh:
    """z 로 겹치지 않는 조각들을 **한 메쉬로 합친다** (불리언 아님 — 단순 병합).

    서로 떨어져 있거나 면끼리 맞닿는 조각들을 한 파일로 내보낼 때 쓴다.
    맞닿는 면은 그대로 남으므로 수밀 검사는 조각별로 하는 것이 맞다.
    """
    vs, fs, off = [], [], 0
    for m in parts:
        vs.append(m.v)
        fs.append(m.f + off)
        off += len(m.v)
    return Mesh(v=np.vstack(vs), f=np.vstack(fs), name="assembly")


def write_stl(path: str, mesh: Mesh, binary: bool = True) -> str:
    """이진(기본) 또는 ASCII STL 로 쓴다. 반환: 쓴 경로."""
    a, b, c = mesh.v[mesh.f[:, 0]], mesh.v[mesh.f[:, 1]], mesh.v[mesh.f[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 1e-20)
    if binary:
        with open(path, "wb") as fh:
            fh.write(f"{mesh.name} — gre.export".ljust(80)[:80].encode("ascii",
                                                                       "replace"))
            fh.write(struct.pack("<I", len(mesh.f)))
            rec = np.zeros((len(mesh.f), 12), dtype=np.float32)
            rec[:, 0:3], rec[:, 3:6], rec[:, 6:9], rec[:, 9:12] = nrm, a, b, c
            buf = bytearray()
            for row in rec:
                buf += struct.pack("<12fH", *row, 0)
            fh.write(bytes(buf))
    else:
        with open(path, "w", encoding="ascii", errors="replace") as fh:
            fh.write(f"solid {mesh.name}\n")
            for k in range(len(mesh.f)):
                fh.write(f"  facet normal {nrm[k,0]:.6e} {nrm[k,1]:.6e} {nrm[k,2]:.6e}\n")
                fh.write("    outer loop\n")
                for p in (a[k], b[k], c[k]):
                    fh.write(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
                fh.write("    endloop\n  endfacet\n")
            fh.write(f"endsolid {mesh.name}\n")
    return path
