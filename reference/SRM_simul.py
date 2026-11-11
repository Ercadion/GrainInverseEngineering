import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from skimage import measure
from scipy.ndimage import distance_transform_edt

# SAFE eval
SAFE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sqrt": np.sqrt, "abs": np.abs,
    "exp": np.exp, "log": np.log,
    "pi": np.pi, "e": np.e,
    "min": np.minimum,
    "max": np.maximum,
}

# 원 방정식 함수
def circle(x, y, cx=0.0, cy=0.0, r=1.0):
    """
    원 방정식 함수
    ================================
    cx, cy      : 중심
    r           : 반지름
    """
    return np.sqrt((x - cx)**2 + (y - cy)**2) - r

# 직사각형 방정식 함수
def box(x, y, cx=0.0, cy=0.0, hx=1.0, hy=1.0):
    """
    직사각형 방정식 함수
    ================================
    cx, cy      : 중심
    hx          : x방향 길이
    hy          : y방향 길이
    """
    return np.maximum(np.abs(x - cx) - hx, np.abs(y - cy) - hy)

# 별 방정식 함수
def star(x, y, cx=0.0, cy=0.0, r_inner=0.5, r_outer=1.0, k=5, sharp=8.0, angle=0.0):
    """
    별 방정식 함수
    ================================
    cx, cy     : 중심
    r_inner    : 골 반지름
    r_outer    : 꼭지 반지름
    k          : 별 꼭지 개수
    sharp      : 뾰족함
    angle      : 회전 각도 (rad)
    """
    X = x - cx
    Y = y - cy
    rho = np.sqrt(X*X + Y*Y)
    theta = np.arctan2(Y, X) - angle

    # cos(k*theta) 기반 별 꼭지 만들기
    w = 0.5 * (1.0 + np.cos(k * theta))       # [0,1]
    w = w ** sharp                             # 뾰족함 조절
    r = r_inner + (r_outer - r_inner) * w
    return rho - r

# 집합 연산 함수
# 합집합
def U(a, b): return np.minimum(a, b)
# 교집합
def I(a, b): return np.maximum(a, b)
# 차집합
def D(a, b): return np.maximum(a, -b)

# 입력 -> 함수 매핑
EXTRA = {"circle": circle, "box": box, "star": star, "U": U, "I": I, "D": D}

def eval_expr(expr: str, X, Y):
    expr = expr.replace("^", "**")
    scope = dict(SAFE)
    scope.update(EXTRA)
    scope.update({"x": X, "y": Y})
    return eval(expr, {"__builtins__": {}}, scope)

# F에서 contour를 통해 서브픽셀 둘레/면적 계산
def contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale):
    """
    컨투어 좌표를 물리좌표로 변환
    ==========================================
    c           : (N,2) = (row, col) index 좌표
    x0          : x 좌표의 시작점 (좌표계 단위)
    y0          : y 좌표의 시작점 (좌표계 단위)
    dx_unit     : x 방향 단위 길이 (좌표계 단위)
    dy_unit     : y 방향 단위 길이 (좌표계 단위)
    scale       : 스케일 (m/coord-unit)
    """
    rows = c[:, 0]
    cols = c[:, 1]
    x_unit = x0 + cols * dx_unit
    y_unit = y0 + rows * dy_unit
    x_m = x_unit * scale
    y_m = y_unit * scale
    return x_m, y_m

def rebuild_SDF_from_mask(port_mask: np.ndarray, dx_mm: float, dy_mm: float) -> np.ndarray:
    dist_out = distance_transform_edt(~port_mask, sampling=(dy_mm, dx_mm))
    dist_in = distance_transform_edt(port_mask, sampling=(dy_mm, dx_mm))

    F_sdf = dist_out - dist_in
    return F_sdf

def perimeter_from_F(F_eff, x0, y0, dx_unit, dy_unit, scale):
    """
    0-레벨 컨투어(다각형)로 둘레(포트 단면 둘레) 계산(서브픽셀)
    포트(음수 영역)가 여러 개면 둘레 합산
    ==========================================
    F_eff: 2D array
    x0: x 좌표의 시작점 (좌표계 단위)
    y0: y 좌표의 시작점 (좌표계 단위)
    dx_unit: x 방향 단위 길이 (좌표계 단위)
    dy_unit: y 방향 단위 길이 (좌표계 단위)
    scale: 스케일 (m/coord-unit)
    """
    contours = measure.find_contours(F_eff, 0.0)
    P = 0.0
    for c in contours:
        x, y = contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale)
        if len(x) < 2:
            continue
        pts = np.column_stack([x, y])
        
        # 페곡선으로 가정해 마지막-처음 연결
        pts2 = np.vstack([pts, pts[0]])
        d = np.diff(pts2, axis=0)
        P += np.sum(np.sqrt((d*d).sum(axis=1)))
    return P

def area_from_F(F_eff, x0, y0, dx_unit, dy_unit, scale):
    """
    0-레벨 컨투어(다각형)로 면적(포트 단면적)을 계산(서브픽셀)
    포트(음수 영역)가 여러 개면 면적 합산
    ==========================================
    F_eff: 2D array
    x0: x 좌표의 시작점 (좌표계 단위)
    y0: y 좌표의 시작점 (좌표계 단위)
    dx_unit: x 방향 단위 길이 (좌표계 단위)
    dy_unit: y 방향 단위 길이 (좌표계 단위)
    scale: 스케일 (m/coord-unit)
    """
    contours = measure.find_contours(F_eff, 0.0)
    A = 0.0
    for c in contours:
        x, y = contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale)
        if len(x) < 3:
            continue
        # 다각형 면적 계산 (신발끈 공식)
        A_poly = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        A += A_poly
    return A

def grain_chamber_setting(D_c, L_c, D_g, D_gc, L_gs, N, Den_ideal, Den_ratio, T_ideal, eff_combustion):
    """
    그레인 계산
    ================================
    D_c: 챔버 직경(mm)
    L_c: 챔버 길이(mm)
    D_g: 그레인 직경(mm)
    D_gc: 코어 직경(mm)
    L_gs: 세그먼트 길이(mm)
    N: 세그먼트 개수
    Den_ratio: 연료 밀도 비율(0~1)
    """
    V_c = np.pi/4 * D_c**2 * L_c                    # 챔버 부피, mm^3
    L_g = L_gs * N                                  # 그레인 총 길이, mm
    V_g = np.pi/4 * (D_g**2 - D_gc**2) * L_g                    # 그레인 총 부피, mm^3
    V_load = V_g/V_c                                # 챔버 대비 그레인 부피 비율
    Den_actual = Den_ideal * Den_ratio              # 그레인 밀도, g/cc
    m_grain = V_g * Den_actual * 1e-6               # 그레인 질량, kg
    A_be = N * 2 * np.pi/4 * (D_g**2 - D_gc**2)   # 그레인 끝면 연소면적, mm^2
    A_bc = N * np.pi * D_gc * L_gs                # 그레인 코어 연소면적, mm^2
    A_bg = A_be + A_bc                              # 그레인 총 연소면적, mm^2
    T_actual = T_ideal * eff_combustion             # 연소실 온도, K
    return V_c, L_g, V_g, V_load, Den_actual, m_grain, A_be, A_bc, A_bg, T_actual

def D_gc_calculation(R_dot, t_interval, D_gc):
    """
    그레인 코어 직경 계산
    ================================
    R_dot: 연소 속도(mm/s)
    t_interval: 시간 간격(s)
    D_gc: 코어 직경(mm)
    """
    D_gc_new = D_gc + 2 * R_dot * t_interval
    return D_gc_new

def L_g_calculation(R_dot, t_interval, L_g, N):
    """
    그레인 길이 계산
    ================================
    R_dot: 연소 속도(mm/s)
    t_interval: 시간 간격(s)
    L_g: 그레인 길이(mm)
    N: 세그먼트 개수
    """
    L_g_new = L_g - 2 * R_dot * t_interval * N
    return L_g_new

def V_g_calculation(D_g, D_gc, L_g):
    """
    그레인 부피 계산
    ================================
    D_g: 그레인 직경(mm)
    D_gc: 코어 직경(mm)
    L_g: 그레인 길이(mm)
    """
    V_g = np.pi/4 * (D_g**2 - D_gc**2) * L_g
    return V_g

def m_g_calculation(V_g, Den_actual):
    """
    그레인 질량 계산
    ================================
    V_g: 그레인 부피(mm^3)
    Den_actual: 그레인 밀도(g/cc)
    """
    m_grain = V_g * Den_actual * 1e-6        # kg
    return m_grain

def A_be_calculation(N, D_g, D_gc):
    """
    그레인 끝면 연소면적 계산
    ================================
    N: 세그먼트 개수
    D_g: 그레인 직경(mm)
    D_gc: 코어 직경(mm)
    """
    A_be = N * 2 * np.pi/4 * (D_g**2 - D_gc**2)
    return A_be

def A_bc_calculation(D_gc, L_g):
    """
    그레인 코어 연소면적 계산
    ================================
    D_gc: 코어 직경(mm)
    L_g: 그레인 길이(mm)
    """
    A_bc = np.pi * D_gc * L_g
    return A_bc

def A_bg_calculation(A_be, A_bc):
    """
    그레인 총 연소면적 계산
    ================================
    A_be: 그레인 끝면 연소면적(mm^2)
    A_bc: 그레인 코어 연소면적(mm^2)
    """
    A_bg = A_be + A_bc
    return A_bg

def Update_grain(r_dot_test, t_interval, D_gc, L_g, N):
    """
    최대 연소면적 찾기
    ================================
    r_dot_test: 테스트용 후퇴율(mm/s)
    t_interval: 시간 간격(s)
    D_gc: 코어 직경(mm)
    L_g: 그레인 길이(mm)
    N: 세그먼트 개수
    D_g: 그레인 직경(mm)
    """
    D_gc_new = D_gc_calculation(r_dot_test, t_interval, D_gc)
    L_g_new = L_g_calculation(r_dot_test, t_interval, L_g, N)
    return D_gc_new, L_g_new

def m_dot_GEN_calculation(D_g, D_gc, D_gc_new, L_g, L_g_new, Den_actual, t_interval):
    """
    질유량 계산
    ================================
    D_g: 그레인 직경(mm)
    D_gc_new: 새로운 코어 직경(mm)
    L_g_new: 새로운 그레인 길이(mm)
    R_dot: 연소 속도(mm/s)
    Den_actual: 그레인 밀도(g/cc)
    t_interval: 시간 간격(s)
    """
    V_g = V_g_calculation(D_g, D_gc, L_g)
    V_g_new = V_g_calculation(D_g, D_gc_new, L_g_new)
    V_free = V_g - V_g_new
    m_GEN = m_g_calculation(V_free, Den_actual)
    m_dot_GEN = m_GEN / t_interval
    return m_dot_GEN, D_gc_new, L_g_new, V_g_new

def m_dot_NOZ_calculation(P_c, A_NOZ_t, C_star):
    """
    노즐 질유량 계산
    ================================
    P_c: 연소실 압력(MPa)
    P_atm: 대기압(MPa)
    A_NOZ_t: 노즐 단면적(mm^2)
    C_star: 특성 속도(m/s)
    """
    P_c_Pa = P_c * 1e6
    A_NOZ_t_m2 = A_NOZ_t * 1e-6
    m_dot_NOZ = P_c_Pa * A_NOZ_t_m2 / C_star
    return m_dot_NOZ

def m_dot_REMAIN_calculation(m_dot_GEN, m_dot_NOZ):
    """
    남은 질유량 계산
    ================================
    m_dot_GEN: 생성 질유량(kg/s)
    m_dot_NOZ: 노즐 질유량(kg/s)
    """
    m_dot_REMAIN = m_dot_GEN - m_dot_NOZ
    return m_dot_REMAIN

def m_REMAIN_calculation(m_dot_REMAIN, t_interval):
    """
    챔버 잔류 추진제 질량 계산
    ================================
    m_dot_REMAIN: 남은 질유량(kg/s)
    t_interval: 시간 간격(s)
    """
    m_REMAIN = m_dot_REMAIN * t_interval
    return m_REMAIN

def Den_REMAIN_calculation(V_c, V_g, m_REMAIN):
    """
    챔버 잔류 추진제 밀도 계산
    ================================
    V_c: 챔버 부피(mm^3)
    V_g: 그레인 부피(mm^3)
    m_dot_REMAIN: 남은 질유량(kg/s)
    t_interval: 시간 간격(s)
    """
    V_free = V_c - V_g

    Den_REMAIN = m_REMAIN / V_free * 1e9 # kg/mm^3 -> g/cc or kg/m^3
    return Den_REMAIN

def R_dot_calculation(a, n, P_chamber):
    """
    후퇴율 계산
    ================================
    a: 후퇴율 상수 a(mm/s)
    n: 후퇴율 상수 n(MPa)
    P_chamber: 연소실 압력(MPa)
    """
    if P_chamber >= 0.101 and P_chamber <= 0.807:
        a = 10.708
        n = 0.625
    elif P_chamber > 0.807 and P_chamber <= 1.503:
        a = 8.763
        n = -0.314
    elif P_chamber > 1.503 and P_chamber <= 3.792:
        a = 7.852
        n = -0.013
    elif P_chamber > 3.792 and P_chamber <= 7.033:
        a = 3.907
        n = 0.535
    elif P_chamber > 7.033 and P_chamber <= 10.67:
        a = 9.653
        n = 0.064
    else:
        print(f"경고: P_chamber={P_chamber:.3f} MPa는 후퇴율 계산 범위를 벗어납니다.")
    R_dot = a * (P_chamber ** n)
    return R_dot

def P_chamber_calculation(Den_REMAIN, R_specific, T_actual, P_atm):
    """
    연소실 압력 계산
    ================================
    Den_REMAIN: 챔버 잔류 추진제 밀도(g/cc)
    R_specific: 기체상수(J/(kg*K))
    T_actual: 연소실 온도(K)
    P_atm: 대기압(MPa)
    """
    P_chamber = Den_REMAIN * R_specific * T_actual * 1e-6 + P_atm  #MPa
    return P_chamber

# 그레인 외벽은 항상 케이싱/인히비터에 접해 있어 연소하지 않는다는
# 전제로 계산한다 (D_g 는 상수). 이 불변식 덕분에 SDF 격자에서 그레인
# 바깥 영역을 매 스텝 다시 클램프할 필요가 없다 — simulate() 참조.
OUTSIDE_SDF = 1.0e9      # 그레인 외부를 나타내는 SDF 상수 [coord]


def throat_area_eroded(A_NOZ_t0, e_NOZ_mm, progression):
    """
    노즐목 삭마를 반영한 목 면적 [mm²].

        d_t(x) = d_t0 + e_NOZ · x        (x = 연소 진행률 0→1)

    나카 SRM / jsrm(erate) 과 동일한 선형 삭마 모델이다. 진행률은
    소비된 추진제 질량분율로 정의하여 임의 포트 형상·다중 세그먼트에도
    쓸 수 있게 했다 (웹 후퇴율 기준과 번아웃 시점에서 일치).

    e_NOZ_mm = 0 이면 A_NOZ_t0 를 그대로 돌려주므로 기존 결과 불변.
    """
    if e_NOZ_mm is None or e_NOZ_mm <= 0.0:
        return A_NOZ_t0
    x = float(progression)
    if not np.isfinite(x):        # NaN/inf 가 목 면적으로 전파되는 것을 차단
        x = 0.0
    x = min(max(x, 0.0), 1.0)
    D0 = (4.0 * A_NOZ_t0 / np.pi) ** 0.5
    D = D0 + e_NOZ_mm * x
    return np.pi / 4.0 * D * D


def Kn_calculation(A_bg, A_NOZ_t):
    """
    Knudsen 수 계산
    ================================
    A_bg: 그레인 총 연소면적(mm^2)
    A_NOZ_t: 노즐 단면적(mm^2)
    """
    Kn = A_bg / A_NOZ_t
    return Kn

def D_gc_transfer(A_port):
    """
    포트 단면적에서 코어 직경 계산
    ================================
    A_port: 포트 단면적(mm^2)
    """
    D_gc = (4.0 * A_port / np.pi) ** 0.5
    return D_gc


def exit_mach_calculation(gamma, ER):
    """
    노즐 팽창비(ER=Ae/At)로부터 출구 마하수 계산 (초음속 해, Newton-Raphson)
    A/A* = (1/Me)*[(2/(γ+1))*(1+(γ-1)/2*Me²)]^((γ+1)/(2*(γ-1)))

    Parameters
    ----------
    gamma: 비열비
    ER: 팽창비 Ae/At

    Returns
    -------
    Me: 출구 마하수
    """
    if ER <= 1.0:
        return 1.0

    # Newton-Raphson 초음속 해
    Me = 2.0
    for _ in range(50):
        term = (2.0 / (gamma + 1)) * (1.0 + (gamma - 1) / 2.0 * Me**2)
        f = (1.0 / Me) * term ** ((gamma + 1) / (2 * (gamma - 1))) - ER

        # df/dMe
        dfdMe = -f / Me + (1.0 / Me) * ((gamma + 1) / (2 * (gamma - 1))) * term ** ((gamma + 1) / (2 * (gamma - 1)) - 1) * (gamma - 1) / 2.0 * 2 * Me

        if abs(dfdMe) < 1e-12:
            break
        Me_new = Me - f / dfdMe
        if abs(Me_new - Me) < 1e-8:
            break
        Me = Me_new

    return max(Me, 1.0)


def Cf_calculation(gamma, P_c_MPa, P_atm_MPa, ER):
    """
    이상 노즐 추력계수 계산
    Cf = sqrt[2γ²/(γ-1) * (2/(γ+1))^((γ+1)/(γ-1)) * (1-(Pe/Pc)^((γ-1)/γ))] + (Pe-Pa)/Pc*ER

    Parameters
    ----------
    gamma: 비열비
    P_c_MPa: 연소실 압력 [MPa]
    P_atm_MPa: 대기압 [MPa]
    ER: 팽창비 Ae/At

    Returns
    -------
    Cf: 추력계수
    Me: 출구 마하수
    Pe_MPa: 출구 압력 [MPa]
    """
    if P_c_MPa <= P_atm_MPa:
        return 0.0, 1.0, P_atm_MPa


    # 추력계수: 이상 노즐 (가우스 항 + 압력항)
    term1 = 2 * gamma**2 / (gamma - 1)
    term2 = (2.0 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
    term3 = 1.0 - (P_atm_MPa / P_c_MPa) ** ((gamma - 1) / gamma) if P_c_MPa > 0 else 1.0

    Cf = np.sqrt(term1 * term2 * term3)


    return Cf


def F_calculation(Cf, P_c_MPa, A_NOZ_t_mm2, NOZ_eff):
    """
    추력 계산 [N]
    F = Cf * Pc[Pa] * At[m²]

    Parameters
    ----------
    Cf: 추력계수
    P_c_MPa: 연소실 압력 [MPa]
    A_NOZ_t_mm2: 노즐 목 단면적 [mm²]

    Returns
    -------
    F: 추력 [N]
    """
    P_c_Pa = P_c_MPa * 1e6
    A_NOZ_t_m2 = A_NOZ_t_mm2 * 1e-6
    F = NOZ_eff * Cf * P_c_Pa * A_NOZ_t_m2
    return F


def Isp_calculation(F, m_dot_NOZ, g0=9.80665):
    """
    비추력 계산 [s]
    Isp = F / (m_dot * g0)

    Parameters
    ----------
    F: 추력 [N]
    m_dot_NOZ: 질유량 [kg/s]
    g0: 표준 중력가속도 [m/s²]

    Returns
    -------
    Isp: 비추력 [s]
    """
    if m_dot_NOZ < 1e-12:
        return 0.0
    Isp = F / (m_dot_NOZ * g0)
    return Isp


# ──────────────────────────────────────────────────────────────
# NAKKA Kn vs 압력 다항식 계수  (P 단위: MPa)
# Kn = a + b*P + c*P² + d*P³ + e*P⁴ + f*P⁵ + g*P⁶
# 출처: Richard Nakka's Experimental Rocketry (KN-series propellant data)
# ──────────────────────────────────────────────────────────────
_KN_NAKKA_POLY = {
    # KNDX: 압력 구간별 계수  [(P_upper_MPa, [계수 리스트]), ...]
    "KNDX": [
        (2.758, [43.500,  0.24168,  50.484,  -9.9115]),              # 150-400 psi
        (5.861, [163.80,  22.224,  -0.2240]),                        # 450-850 psi
        (8.963, [5095.0, -2460.4,  453.48,  -35.532,  1.0175]),      # 900-1300 psi
    ],
    # 단일 범위 추진제 (0-1300 psi)
    "KNSB":        [34.500, -10.975,  69.667, -19.664,  2.1478, -8.146e-2],  # fine, order 5
    "KNSB_COARSE": [38.000,  92.391,  -2.0438],                               # coarse, order 2
    "KNSU":        [32.954,  44.108,  -1.1025],                               # order 2
    "KNER":        [50.410, 211.194, -47.7976, 8.779018, -0.830282, 0.030519],# order 5
    "KNMN":        [38.000,  92.391,  -2.0438],                               # coarse, order 2
    "KNPSB":       [54.530,  33.652,  -3.4185,  1.55e-1],                     # order 3
    "KNFR":        [39.748,  51.618,  -0.9264],                               # order 2
}


def kn_nakka(propellant_type, P_MPa):
    """
    NAKKA 다항식으로 목표 압력에서의 권장 Kn 계산
    Kn = a + b*P + c*P² + ...  (P in MPa)

    Parameters
    ----------
    propellant_type : str   PROPELLANT 딕셔너리의 키
    P_MPa          : float  목표 연소실 압력 [MPa]

    Returns
    -------
    Kn : float
    """
    data = _KN_NAKKA_POLY.get(propellant_type)
    if data is None:
        print(f"  ※ {propellant_type}: NAKKA Kn 데이터 없음 → 기본값 250 사용")
        return 250.0

    # KNDX: 구간별 계수 리스트
    if propellant_type == "KNDX":
        coeffs = data[-1][1]          # 기본: 마지막(최고압) 구간
        for (p_upper, c) in data:
            if P_MPa <= p_upper:
                coeffs = c
                break
    else:
        coeffs = data

    Kn = sum(c * P_MPa ** i for i, c in enumerate(coeffs))
    return max(float(Kn), 50.0)       # 50 미만은 물리적으로 부적절


# ══════════════════════════════════════════════════════════════════════════
#  목표 압력 자동 보정 (노즐목 역산)  v1.0
# ══════════════════════════════════════════════════════════════════════════
#
# 왜 목표 압력이 그대로 안 나오나:
#   P_target 은 물리 루프에 한 번도 들어가지 않는다. 하는 일은
#       P_target → Kn_max = kn_nakka(...) → A_t = A_bg_max / Kn_max
#   뿐이고, 그 뒤 시뮬레이션은 A_t 만 받아 자기 물리(자기 밀도·c*·연소속도
#   법칙)로 압력을 푼다. 즉 설계식과 시뮬레이션이 서로 다른 두 모델이다.
#   Nakka 의 Kn–P 다항식은 그의 조건에서 뽑힌 곡선이라, 밀도비나 연소속도
#   표가 다르면 결과가 목표를 빗나간다. 평형압이 P ∝ ρ^(1/(1−n)) 이므로
#   예컨대 Den_ratio 0.92 만으로도 n≈0 일 때 약 −8% 가 난다.
#
# 해법:
#   평형 조건  P = Kn·ρ·a·Pⁿ·c*  에서  P ∝ A_t^(−1/(1−n))  이므로 목을
#   해석적으로 역산할 수 있다 (뉴턴 반복이 아니라 멱법칙의 역함수다).
#   단일 구간 안에서는 1회로 0.1% 안에 들지만, 구간별 연소속도 표의
#   경계를 넘는 목표에서는 오버슛하며 진동한다. 그래서 부호가 반대인
#   두 점이 잡히면(= 해가 사이에 갇히면) 아래 bracket_throat 로 갈아탄다.
#   ⚠ 이것은 실측을 보고 맞추는 캘리브레이션이 아니라, 설계식을 자기
#     모델과 self-consistent 하게 푸는 것이다 (측정 데이터를 쓰지 않음).

def throat_for_target_pressure(A_NOZ_t, P_peak, P_target, n_eff,
                               max_step=0.35):
    """
    A_t_new = A_t · (P_peak / P_target)^(1 − n_eff)

    A_NOZ_t : 현재 노즐목 면적 [mm²]
    P_peak  : 그 목으로 돌린 시뮬레이션의 최대 챔버압 [MPa]
    P_target: 목표 압력 [MPa]
    n_eff   : P_peak 부근에서 유효한 압력지수 (구간별 표면 도달 압력이
              속한 구간의 n)
    max_step: 한 번에 허용할 면적 변화 비율 상한 — 발산 방지

    반환: (A_t_new [mm²], reason) — reason 은 "" / "step" / "n" / "step+n".
          두 사유를 합치면 정상적인 n 에 "n 이 비정상" 이라는 헛경고가 나가므로
          분리해서 돌려준다.

    ⚠ n_eff ≥ 1 이면 지수 (1−n) 의 부호가 뒤집혀 보정이 반대 방향으로
      간다. 이 영역에서도 평형 자체는 존재하지만(P^(1−n) = Kn·ρ·a·c*)
      **불안정**해서 교란이 감쇠 대신 증폭되고, Kn 이 커질수록 평형압이
      오히려 내려간다. 정상 연소가 성립하지 않으므로 [-0.5, 0.9] 로
      클램프하고 호출부에서 별도 경고를 낸다
      (rocket_input_builder.run_tms_sim_series 의 "압력지수" 경고).
    """
    if not (A_NOZ_t > 0 and P_peak > 0 and P_target > 0):
        return float(A_NOZ_t), ""
    n_use = min(max(float(n_eff), -0.5), 0.9)
    ratio = (float(P_peak) / float(P_target)) ** (1.0 - n_use)
    lo, hi = 1.0 - float(max_step), 1.0 + float(max_step)
    why = []
    if not (lo <= ratio <= hi):
        why.append("step")
    if n_use != float(n_eff):
        why.append("n")
    ratio = min(max(ratio, lo), hi)
    return float(A_NOZ_t) * ratio, "+".join(why)


def bracket_throat(A_lo, err_lo, A_hi, err_hi, inset=0.05):
    """
    해가 갇힌 구간 [A_lo, A_hi] 안에서 다음 시도할 노즐목을 고른다.

    P 는 A_t 에 대해 단조 감소하므로
        A_lo : 오차 > 0 (압력이 목표보다 높음 → 목을 키워야 함)
        A_hi : 오차 < 0 (압력이 목표보다 낮음 → 목을 줄여야 함)
    log A – log(1+err) 평면에서 false position(할선)을 쓴다. 평형이
    멱법칙이라 이 좌표에서 거의 직선이어서 보통 1~2회에 수렴한다.

    inset : 새 점을 구간 양끝에서 이 비율만큼 안쪽으로 밀어 넣는다.
            할선이 끝점에 달라붙어 구간이 안 줄어드는 정체를 막는다
            (매 반복 최소 inset 만큼 구간이 줄어드는 것을 보장).
    """
    a1, a2 = float(A_lo), float(A_hi)
    if not (a1 > 0 and a2 > a1):
        return 0.5 * (a1 + a2)
    # err = -1 은 압력 0(빈 시계열)에 대응한다. log1p(-1) = -inf 이므로
    # 방어하지 않으면 ValueError 로 1회차 성공 결과까지 통째로 날아간다.
    e1, e2 = float(err_lo), float(err_hi)
    if not (e1 == e1 and e2 == e2) or e1 <= -1.0 or e2 <= -1.0:
        return math.sqrt(a1 * a2)
    f1, f2 = math.log1p(e1), math.log1p(e2)
    if f1 == f2:
        a = math.sqrt(a1 * a2)
    else:
        x = math.log(a1) - f1 * (math.log(a2) - math.log(a1)) / (f2 - f1)
        a = math.exp(x)
    pad = float(inset) * (a2 - a1)
    return min(max(a, a1 + pad), a2 - pad)


def equilibrium_pressure_roots(r_of_P, Kn, Den_gcc, C_star,
                               P_lo=0.15, P_hi=12.0, n_scan=8000):
    """
    평형 챔버압을 찾는다:  g(P) = Kn·ρ[g/cc]·r(P)[mm/s]·c*[m/s]·1e-6 − P = 0

    구간별 연소속도 표는 구간 경계에서 r 이 불연속이라, 어떤 Kn 대역에서는
    평형해가 2~3 개 생긴다. 그 대역에서는 어느 압력에 안착하는지가 점화
    과도에 좌우된다.
      예) KNSB 실측표는 P=3.792 MPa 경계에서 r 이 뛰므로, ρ·c* 에 따라
          Kn 이 어떤 좁은 창(폭 10 남짓) 안에 들면 3중해가 된다.
          ⚠ 그 창의 절대 위치는 ρ·c* 에 반비례해 움직인다. 특정 숫자를
            외우지 말고 이 함수로 확인할 것.

    반환: (roots, jumps)
      roots : 실제 평형해 [MPa]  (g 가 연속적으로 0 을 통과)
      jumps : r 의 불연속 때문에 g 의 부호만 건너뛴 지점 [MPa].
              해가 아니다 — 원인을 알려주는 용도다.
    부호 전환 구간의 |g| 가 격자 한 칸의 전형적 변화량보다 훨씬 크면
    그것은 교차가 아니라 점프로 분류한다.
    """
    Ps = np.linspace(float(P_lo), float(P_hi), int(n_scan))
    vals = np.empty(len(Ps), dtype=float)
    for i, p in enumerate(Ps):
        try:
            vals[i] = Kn * Den_gcc * float(r_of_P(float(p))) * C_star * 1e-6 - p
        except Exception:
            vals[i] = np.nan
    ok = np.isfinite(vals)
    if ok.sum() < 3:
        return [], []
    Ps, g = Ps[ok], vals[ok]
    dg = np.abs(np.diff(g))
    scale = 3.0 * float(np.median(dg)) if len(dg) else 0.0
    roots, jumps = [], []
    span = float(Ps[-1] - Ps[0]) / max(len(Ps) - 1, 1)
    last_P = None
    for i in np.where(g[:-1] * g[1:] <= 0.0)[0]:
        if g[i] == 0.0 and g[i + 1] == 0.0:
            continue
        # NaN 마스킹으로 인덱스가 압축될 수 있으므로 격자 칸수가 아니라
        # 실제 압력 거리로 중복을 판정한다.
        if last_P is not None and (Ps[i] - last_P) <= 2.0 * span:
            continue
        last_P = float(Ps[i])
        Pm = float(0.5 * (Ps[i] + Ps[i + 1]))
        if max(abs(g[i]), abs(g[i + 1])) <= max(scale, 1e-12):
            roots.append(Pm)
        else:
            jumps.append(Pm)
    return roots, jumps


def simulate_NOZ_t(
    D_gc, D_g,
    L_g,
    N,
    t_interval,
    Kn_max,
    r_dot_test   
):
    """
    노즐 단면적 계산
    ================================
    D_gc: 코어 직경(mm)
    D_g: 그레인 직경(mm)
    L_g: 그레인 길이(mm)
    t_interval: 시간 간격(s)
    Kn_max: 최대 Knudsen 수(무차원)
    r_dot_test: 테스트용 후퇴율(mm/s)
    """
    # 최대 연소면적은 "물리적 web 범위(버나웃 전)" 에서만 탐색해야 한다.
    # 과거 코드는 버나웃 정지 조건이 없어, 코어 직경을 그레인 외벽 너머까지
    # 비물리적으로 키우며 가짜 최대 연소면적을 찾았고 (끝면적이 음수가 되는
    # 영역), 그 결과 노즐목이 보어보다 커지는 과대 설계가 나왔다.
    # 개선: D_gc < D_g AND L_g > 0 인 범위에서만 탐색, 초기점도 포함.
    A_bg_list = [A_bg_calculation(A_be_calculation(N, D_g, D_gc),
                                   A_bc_calculation(D_gc, L_g))]
    D_gc_new, L_g_new = Update_grain(r_dot_test, t_interval, D_gc, L_g, N)
    while D_gc_new < D_g and L_g_new > 0.0:
        A_be_new = A_be_calculation(N, D_g, D_gc_new)
        A_bc_new = A_bc_calculation(D_gc_new, L_g_new)
        A_bg_list.append(A_bg_calculation(A_be_new, A_bc_new))
        D_gc_new, L_g_new = Update_grain(r_dot_test, t_interval, D_gc_new, L_g_new, N)
    A_bg_max = max(A_bg_list)
    A_NOZ_t = A_bg_max / Kn_max
    print(f"최대 연소면적 A_bg_max: {A_bg_max:.2f} mm^2")

    # 설계 타당성: 노즐목 지름이 그레인/보어 지름 이상이면 비현실적
    D_throat = (4.0 * A_NOZ_t / np.pi) ** 0.5
    if D_throat >= D_g:
        print(f"⚠️  경고: 계산된 노즐목 지름 {D_throat:.2f} mm 가 "
              f"그레인 지름 {D_g:.2f} mm 이상입니다. 연소면적이 과대 "
              f"(긴 그레인 등) 하여 비현실적 설계입니다 — 그레인 형상/Kn "
              f"재검토 필요.")

    return A_NOZ_t
    
    
def simulate(
    exprs,                  # 포트 방정식(들)
    xlim, ylim,             # 좌표계 단위 범위
    V_c,                    # 챔버 부피 [mm^3]
    D_g, D_gc,              # 그레인 직경, 코어 직경 [mm]
    N,                      # 세그먼트 개수
    L_g,                    # 그레인 길이 [mm]
    V_g,                    # 그레인 부피 [mm^3]
    den_actual,             # 그레인 밀도 [g/cc]
    m_g,                    # 그레인 질량 [kg]
    A_NOZ_t,                # 노즐 단면적 [mm^2]
    NOZ_eff,                # 노즐 효율 (0~1)
    R_specific,             # 기체상수 [J/(kg*K)]
    gamma,                  # 비열비
    C_star,                 # 특성 속도 [m/s]
    T_actual,               # 연소실 온도 [K]
    P_target,               # 목표 압력 [MPa]
    P_atm,                  # 초기 압력(대기압) Pc [MPa]
    e_NOZ,                  # 노즐 삭마 정도
    a, n,                   # 후퇴율 상수 a, n [mm/s], [MPa]
    t_end, t_interval,      # 시간 간격[s]
    grid,                   # 그리드 크기
    snapshot_dt,            # 스냅샷 간격[s]
    ER,                     # 노즐 팽창비 Ae/At
    t_ignition=0.0          # 점화 램프 시간 [s] (0이면 비활성화)
):
    # 격자(단위 = coord)
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xs, ys)

    dx_unit = xs[1] - xs[0]
    dy_unit = ys[1] - ys[0]

    # x 범위(지름)가 곧 실제 그레인 직경이므로 스케일 계산
    # scale = (mm / coord-unit)
    scale = D_g / (xlim[1] - xlim[0])

     # 그레인 외벽: (0,0) 중심, coord 단위 반지름
    Rcoord = 0.5 * (xlim[1] - xlim[0])
    grain_sdf = circle(X, Y, 0.0, 0.0, Rcoord)
    grain_mask = (grain_sdf <= 0)

    # # 포트/가스 도형의 SDF F (식이 여러 개이면 union=min)
    F = np.full_like(X, 1e9, dtype=float)
    for e in exprs:
        F = np.minimum(F, eval_expr(e, X, Y))

    # mm per pixel
    dx_mm = dx_unit * scale
    dy_mm = dy_unit * scale

    # 재생성 관련 변수
    rebuild_interval = 45
    rebuild_counter = 0
    dr_accum = 0.0
    dr_threshold = dx_mm * 3.0

    # 그레인 밖은 큰 양수로 고정한다.
    #
    # ── 외벽 비연소 불변식 ────────────────────────────────────────────
    # 그레인 외벽은 케이싱/인히비터에 접해 연소하지 않으므로 D_g 는
    # 상수이고, grain_mask 도 전 시간에 걸쳐 불변이다. 따라서 바깥
    # 영역을 OUTSIDE_SDF 로 한 번만 고정해두면, 이후 매 스텝의
    # F -= dr 연산으로도 (OUTSIDE_SDF − Σdr) ≫ 0 이 유지된다
    # (Σdr ≤ D_g/2, 즉 최대 수백 mm 대 1e9). 예전 코드가 매 스텝
    # 두 번씩 수행하던 np.where 전체 격자 클램프는 중복이므로 제거했다.
    # SDF 재생성 직후에만 다시 고정하면 된다.
    outside = ~grain_mask
    F[outside] = OUTSIDE_SDF

    # 스냅샷 snapshot_dt 간격으로 저장
    if snapshot_dt is None or snapshot_dt <= 0:
        snap_times = np.array([0.0, t_end], dtype=float)
    else:
        snap_times = np.arange(0.0, t_end + 1e-12, snapshot_dt, dtype=float)
        if snap_times[-1] < t_end - 1e-12:
            snap_times = np.append(snap_times, t_end)
    snap_idx = 0
    tol = 0.5 * t_interval

    # 결과 저장용 리스트 생성
    t_list = []
    D_gc_list, L_g_list = [], []
    A_be_list, A_bc_list, A_bg_list = [], [], []
    m_g_list, m_dot_GEN_list, m_dot_NOZ_list, m_dot_REMAIN_list = [], [], [], []
    den_REMAIN_list, P_c_list, R_dot_list = [], [], []
    Kn_list = []
    F_list, Cf_list, Isp_list = [], [], []
    snapshot_list = []

    t = 0.0
    P_c = P_atm
    R_dot = R_dot_calculation(a, n, P_c)
    m_REMAIN = 0.0

    m_g0 = m_g if m_g > 0 else None     # 삭마 진행률 기준 (초기 그레인 질량)

    while t <= t_end + 1e-12:
        F_eff = F                        # 외벽 비연소 → 이미 클램프된 상태

        # 노즐목 삭마: 소비 질량분율을 진행률로 사용
        A_t_now = throat_area_eroded(
            A_NOZ_t, e_NOZ, 0.0 if m_g0 is None else 1.0 - m_g / m_g0)

        # 포트 단면적(서브픽셀, mm^2)
        A_port = area_from_F(F_eff, xlim[0], ylim[0], dx_unit, dy_unit, scale)
        if A_port <= 0:
            break
        if t < 1e-12:
            D_port = D_gc_transfer(A_port)
            print("인식된 코어 직경 D_gc =", D_gc, "mm")
            print("오차율:", abs(D_port - D_gc) / D_gc * 100, "%")

        D_gc_new, L_g_new = Update_grain(R_dot, t_interval, D_gc, L_g, N)

        m_dot_GEN, D_gc, L_g, V_g =  m_dot_GEN_calculation(D_g, D_gc, D_gc_new, L_g, L_g_new, den_actual, t_interval)

        # 점화 램프 함수: t < t_ignition 구간에서 m_dot_GEN을 0→1 선형 스케일
        # 실제 점화 시 화염이 그레인 전면에 퍼지는 과도 현상을 모사
        if t_ignition > 0.0 and t < t_ignition:
            ignition_factor = t / t_ignition   # 0.0 → 1.0 선형 증가
            m_dot_GEN *= ignition_factor

        D_gc_list.append(D_gc)
        L_g_list.append(L_g)
        
        A_be = A_be_calculation(N, D_g, D_gc)
        A_bc = A_bc_calculation(D_gc, L_g)
        A_bg = A_bg_calculation(A_be, A_bc)
        Kn = Kn_calculation(A_bg, A_t_now)
        A_be_list.append(A_be)
        A_bc_list.append(A_bc)
        A_bg_list.append(A_bg)
        Kn_list.append(Kn)

        m_g -= m_dot_GEN * t_interval
        m_g_list.append(m_g)

        # ── 챔버 질량수지 & 압력 (암시적 backward-Euler, 개선) ─────
        # 기존 코드는 명시적 1-step + `if m_dot_REMAIN<0: 0` 클램프였는데,
        #   (1) 클램프가 챔버 배출(저장 질량 감소)을 막아 압력/추력/임펄스를
        #       비물리적으로 부풀렸고 (질량 비보존),
        #   (2) 명시적 갱신은 dt 가 크면 압력이 0↔고압으로 진동 (불안정).
        # 노즐 배출은 m_dot_NOZ = P_c·(A_NOZ_t/C_star) 로 P_c 에 "선형"이므로
        # 다음 두 식을 연립하면 P_c 에 대한 1차식 → 폐형 암시해가 존재:
        #   m_REMAIN_new = m_REMAIN + (m_dot_GEN - P_c·At/c*)·dt
        #   P_c          = (1e3·R·T / V_free)·m_REMAIN_new + P_atm
        V_free_mm3 = V_c - V_g
        if V_free_mm3 <= 0.0:
            print(f"형상 비현실: 자유체적 <= 0 (그레인이 챔버보다 큼) → 종료")
            break
        C1 = 1.0e3 * R_specific * T_actual / V_free_mm3       # MPa per kg
        At_term = A_t_now / C_star                            # m_dot_NOZ = P_c·At_term [kg/s]
        P_c = (C1 * (m_REMAIN + m_dot_GEN * t_interval) + P_atm) / (1.0 + C1 * At_term * t_interval)

        m_dot_NOZ = m_dot_NOZ_calculation(P_c, A_t_now, C_star)
        m_dot_REMAIN = m_dot_GEN - m_dot_NOZ
        m_REMAIN += m_dot_REMAIN * t_interval
        if m_REMAIN < 0.0:          # 저장 질량은 음수 불가 → 빈 챔버
            m_REMAIN = 0.0
            P_c = P_atm
        den_REMAIN = Den_REMAIN_calculation(V_c, V_g, m_REMAIN)

        m_dot_GEN_list.append(m_dot_GEN)
        m_dot_NOZ_list.append(m_dot_NOZ)
        m_dot_REMAIN_list.append(m_dot_REMAIN)
        den_REMAIN_list.append(den_REMAIN)
        P_c_list.append(P_c)

        R_dot = R_dot_calculation(a, n, P_c)
        R_dot_list.append(R_dot)

        # 추력, 추력계수, 비추력 계산
        # 주의: SDF 배열도 F라는 이름을 사용하므로 추력은 thrust 변수로 분리
        # F와 Isp는 업데이트된 P_c(= P_c_new)를 일관되게 사용
        if P_c > P_atm:
            Cf = Cf_calculation(gamma, P_c, P_atm, ER)
            thrust = F_calculation(Cf, P_c, A_t_now, NOZ_eff)
            m_dot_for_isp = m_dot_NOZ_calculation(P_c, A_t_now, C_star)
            Isp = Isp_calculation(thrust, m_dot_for_isp)
        else:
            Cf, thrust, Isp = 0.0, 0.0, 0.0
        F_list.append(thrust)
        Cf_list.append(Cf)
        Isp_list.append(Isp)

        t_list.append(t)


        # 스냅샷 저장
        while snap_idx < len(snap_times) and (t + tol) >= snap_times[snap_idx]:
            snapshot_list.append((snap_times[snap_idx], F_eff.copy()))
            snap_idx += 1

        #SDF 가정 하에서 F를 rdot*dt만큼 팽창
        # F는 coord 단위, rdot*dt는 m -> coord로 환산
        #F = F - (r_dot * dt) / scale

        post_SDF_mask = (F_eff <= 0)
        dr_mm = R_dot * t_interval
        dr_accum += dr_mm
        rebuild_counter += 1
        
        if(rebuild_counter >= rebuild_interval) or (dr_accum >= dr_threshold):
            F = rebuild_SDF_from_mask(post_SDF_mask, dx_mm, dy_mm)
            F[outside] = OUTSIDE_SDF     # 재생성 직후에만 재고정

            rebuild_counter = 0
            dr_accum = 0.0

        F -= dr_mm                       # 인플레이스 (외벽 불변식 덕분에 안전)

        # 외벽 도달: 포트가 그레인 경계에 닿으면 종료
        # 경계 픽셀 근처에서 True가 생기면 닿았다고 판단
        if D_gc >= D_g or L_g <= 0 or m_g <= 0:
            if D_gc >= D_g:
                print(f"외벽 도달: D_gc={D_gc:.2f} mm >= D_g={D_g:.2f} mm")
            elif L_g <= 0:
                print(f"높이 도달: L_g={L_g:.2f} mm <= 0 mm")
            elif m_g <= 0:
                print(f"그레인 소진: m_g={m_g:.2f} g <= 0 g")
            break

        t += t_interval

    if e_NOZ and e_NOZ > 0.0:
        D0 = (4.0 * A_NOZ_t / np.pi) ** 0.5
        A_end = throat_area_eroded(
            A_NOZ_t, e_NOZ, 0.0 if m_g0 is None else 1.0 - m_g / m_g0)
        print(f"노즐목 삭마 반영: d_t {D0:.2f} → "
              f"{(4.0*A_end/np.pi)**0.5:.2f} mm "
              f"(면적 +{(A_end/A_NOZ_t-1)*100:.1f}%)")

    # 시각화용 물리좌표 그리드(m)
    X_m = X * scale
    Y_m = Y * scale
    grain_sdf_m = circle(X_m, Y_m, 0.0, 0.0, D_g / 2.0)

    return (
        np.array(t_list),
        np.array(D_gc_list), np.array(L_g_list),
        np.array(A_be_list), np.array(A_bc_list), np.array(A_bg_list),
        np.array(m_g_list), np.array(m_dot_GEN_list), np.array(m_dot_NOZ_list), np.array(m_dot_REMAIN_list),
        np.array(den_REMAIN_list), np.array(P_c_list), np.array(R_dot_list),
        np.array(Kn_list),
        np.array(F_list), np.array(Cf_list), np.array(Isp_list),
        snapshot_list,
        X_m, Y_m,
        grain_sdf_m
    )


def save_srm_json(
    # simulation results
    t_arr, F_arr, P_c_arr, m_dot_NOZ_arr, Isp_arr, m_g_arr, Kn_arr,
    # SRM input params
    gamma, P_atm, A_NOZ_t, ER, C_star, T_actual,
    # rocket body params (for UE)
    body_mass_kg, ref_area_m2, ref_length_m,
    Ixx, Iyy, Izz, XCG_initial_m, XCG_final_m,
    body_length_m, nose_length_m, fin_count, fin_span_m,
    fin_root_chord_m, fin_tip_chord_m, XCP_m,
    # grain geometry (compute_inertia_table 연동용)
    D_g_mm=0.0, D_gc_0_mm=0.0, L_gs_mm=0.0, N_segments=0,
    Den_gcc=0.0, X_grain_front_m=0.0,
    wind_speed_mps=0.0, wind_direction=None,
    n_knots=30,
    output_ue="srm_rocket_ue.json",
    output_detail="srm_simul_detail.json"
):
    """
    SRM 시뮬레이션 결과를 JSON으로 저장 (UE 형식 및 상세 데이터)

    Parameters
    ----------
    t_arr, F_arr, P_c_arr, ...: 시뮬레이션 시간 및 결과 배열들
    gamma, P_atm, A_NOZ_t, ER, C_star, T_actual: SRM 파라미터
    body_mass_kg, ref_area_m2, ...: 로켓 기체 파라미터
    n_knots: 추력곡선 다운샘플링 점 개수
    output_ue, output_detail: 출력 파일 경로
    """
    import json
    from datetime import datetime
    from scipy.interpolate import interp1d

    if len(t_arr) < 2 or len(F_arr) < 2:
        print("경고: 시뮬레이션 데이터가 충분하지 않습니다.")
        return

    # 시뮬레이션 파라미터
    burn_time = t_arr[-1]
    max_thrust = np.max(F_arr)
    total_impulse = np.trapezoid(F_arr, t_arr) if len(t_arr) > 1 else 0.0
    max_pressure = np.max(P_c_arr)
    avg_isp = np.mean(Isp_arr[Isp_arr > 0]) if np.any(Isp_arr > 0) else 0.0

    # T/W ratio
    total_mass = body_mass_kg + m_g_arr[0]
    initial_tw = (max_thrust / 9.80665) / total_mass if total_mass > 0 else 0.0

    # 추력곡선 다운샘플링 (n_knots 개의 균등간격 점)
    if len(t_arr) > n_knots:
        t_knots = np.linspace(0.0, burn_time, n_knots)
        f_interp = interp1d(t_arr, F_arr, kind='linear', fill_value=0.0, bounds_error=False)
        F_knots = f_interp(t_knots)
        # 처음과 끝 확실히 0으로
        F_knots[0] = 0.0
        F_knots[-1] = 0.0
    else:
        t_knots = t_arr
        F_knots = F_arr

    thrust_curve = [[float(t), float(f)] for t, f in zip(t_knots, F_knots)]

    # 압력곡선 다운샘플링 (bar 단위로 변환)
    if len(t_arr) > n_knots:
        p_interp = interp1d(t_arr, P_c_arr * 10.0, kind='linear', fill_value=0.0, bounds_error=False)
        P_knots = p_interp(t_knots)
    else:
        P_knots = P_c_arr * 10.0

    pressure_curve = [[float(t), float(p)] for t, p in zip(t_knots, P_knots)]

    # UE JSON 구조
    ue_json = {
        "_source": {
            "file": "srm_simulation",
            "mode": "srm",
            "burn_time_s": float(burn_time),
            "max_thrust_N": float(max_thrust),
            "total_impulse_Ns": float(total_impulse),
            "max_pressure_bar": float(max_pressure * 10.0),
            "smooth_window": 1,
            "n_knots": n_knots,
            "tms_dry_mass_kg": 0.0,
            "tms_prop_mass_kg": float(m_g_arr[0]),
            "mass_source": "srm_simulation",
            "sim_body_mass_kg": float(body_mass_kg),
            "sim_prop_mass_kg": float(m_g_arr[0]),
            "sim_total_mass_kg": float(body_mass_kg + m_g_arr[0]),
            "isp_s": float(avg_isp),
            "initial_tw_ratio": float(initial_tw),
            "self_check_warnings": [],
            "srm_gamma": float(gamma),
            "srm_C_star_ms": float(C_star),
            "srm_T_actual_K": float(T_actual),
            "srm_ER": float(ER),
            "srm_A_NOZ_t_mm2": float(A_NOZ_t),
            "srm_Kn_max": float(np.max(Kn_arr)) if len(Kn_arr) > 0 else 0.0,
            "srm_Kn_avg": float(np.mean(Kn_arr)) if len(Kn_arr) > 0 else 0.0
        },
        "rocket": {
            "dry_mass_kg": float(body_mass_kg),
            "prop_mass_kg": float(m_g_arr[0]),
            "ref_area_m2": float(ref_area_m2),
            "ref_length_m": float(ref_length_m),
            "Ixx": float(Ixx),
            "Iyy": float(Iyy),
            "Izz": float(Izz),
            "XCG_initial_m": float(XCG_initial_m),
            "XCG_final_m": float(XCG_final_m)
        },
        "geometry": {
            "body_length_m": float(body_length_m),
            "nose_length_m": float(nose_length_m),
            "fin_count": int(fin_count),
            "fin_span_m": float(fin_span_m),
            "fin_root_chord_m": float(fin_root_chord_m),
            "fin_tip_chord_m": float(fin_tip_chord_m),
            "XCP_m": float(XCP_m),
            "XCG_initial_m": float(XCG_initial_m)
        },
        "thrust_curve": thrust_curve,
        "pressure_curve": pressure_curve,
        "environment": {
            "wind_direction": wind_direction if wind_direction is not None else [1.0, 0.0, 0.0],
            "wind_speed_mps": float(wind_speed_mps)
        },
        "simulation": {
            "thrust_aligned_prob": 1.0,
            "thrust_max_misalign_deg": 45.0,
            "launch_rail_length_m": 5.0,
            "rk45_abs_tol": 1e-6,
            "rk45_rel_tol": 1e-6,
            "rk45_init_step": 0.005,
            "rk45_max_step": 0.05,
            "_srm_software": "SRM_simul.py",
            "_srm_generated": datetime.now().isoformat()
        }
    }

    # UE JSON 저장
    with open(output_ue, 'w', encoding='utf-8') as f:
        json.dump(ue_json, f, indent=2, ensure_ascii=False)
    print(f"UE JSON 저장: {output_ue}")

    # 상세 데이터 JSON
    detail_json = {
        "_meta": {
            "description": "SRM 연소 시뮬레이션 상세 데이터",
            "source": "SRM_simul.py",
            "generated": datetime.now().isoformat()
        },
        "inputs": {
            "gamma": float(gamma),
            "P_atm_MPa": float(P_atm),
            "A_NOZ_t_mm2": float(A_NOZ_t),
            "ER": float(ER),
            "C_star_ms": float(C_star),
            "T_actual_K": float(T_actual)
        },
        "grain": {
            "_설명": "BATES 그레인 파라미터 (compute_inertia_table 자동 연동용)",
            "D_g_mm":         float(D_g_mm),
            "D_gc_0_mm":      float(D_gc_0_mm),
            "L_gs_mm":        float(L_gs_mm),
            "N_segments":     int(N_segments),
            "Den_gcc":        float(Den_gcc),
            "X_grain_front_m": float(X_grain_front_m)
        },
        "time_series": {
            "t_s": t_arr.tolist(),
            "P_c_MPa": P_c_arr.tolist(),
            "F_N": F_arr.tolist(),
            "Cf": [],
            "Isp_s": Isp_arr.tolist(),
            "m_dot_NOZ_kgs": m_dot_NOZ_arr.tolist(),
            "m_g_kg": m_g_arr.tolist(),
            "Kn": Kn_arr.tolist()
        },
        "summary": {
            "burn_time_s": float(burn_time),
            "max_thrust_N": float(max_thrust),
            "avg_thrust_N": float(np.mean(F_arr)),
            "max_pressure_MPa": float(max_pressure),
            "total_impulse_Ns": float(total_impulse),
            "avg_isp_s": float(avg_isp),
            "max_Kn": float(np.max(Kn_arr)) if len(Kn_arr) > 0 else 0.0,
            "min_Kn": float(np.min(Kn_arr)) if len(Kn_arr) > 0 else 0.0,
            "prop_mass_consumed_kg": float(m_g_arr[0] - m_g_arr[-1])
        }
    }

    with open(output_detail, 'w', encoding='utf-8') as f:
        json.dump(detail_json, f, indent=2, ensure_ascii=False)
    print(f"상세 JSON 저장: {output_detail}")


def run_self_checks_srm(F_arr, m_dot_NOZ_arr, Isp_arr, Kn_arr, body_mass_kg,
                        prop_mass_kg, P_atm, A_bg_arr=None, m_g_arr=None):
    """
    SRM 시뮬레이션 자체 검사

    Parameters
    ----------
    F_arr, m_dot_NOZ_arr, Isp_arr, Kn_arr: 시뮬레이션 결과 배열
    body_mass_kg: 로켓 건조 질량 [kg] — 모터 하드웨어(케이스·노즐) 포함,
                  추진제는 제외. 0 이면 T/W 판정을 생략한다.
    A_bg_arr    : 총 연소면적 시계열 [mm²] (선택) — Kn 판정용 폴백.
    m_g_arr     : 잔여 추진제 질량 시계열 [kg] (선택) — Kn 판정의 1순위
                  기준. "정상 연소 구간 = 추진제가 초기의 OPERATING_MASS_FRAC
                  이상 남은 구간" 으로 본다. 면적 기준보다 견고한 이유는
                  아래 주석 참조.
    prop_mass_kg: 추진제 질량
    P_atm: 대기압

    Returns
    -------
    warnings: 경고 메시지 리스트
    """
    warnings = []

    # ── T/W ratio ────────────────────────────────────────────────────
    # T/W 는 "모터" 가 아니라 "기체" 의 지표다. 건조 질량이 없으면 분모가
    # 추진제 질량뿐이라 실제의 수십 배가 나온다(예: 추진제 250g 짜리 모터에
    # F_max 211N → 건조질량 0 이면 86, 3.6kg 이면 5.6). 예전 코드는 호출부가
    # body_mass_kg=0.0 을 넘겨 항상 오탐을 냈으므로, 미입력이면 판정 자체를
    # 생략하고 안내만 한다.
    max_thrust = np.max(F_arr) if len(F_arr) > 0 else 0.0
    if body_mass_kg is None or body_mass_kg <= 0.0:
        warnings.append(
            "ℹ️  로켓 건조 질량 미입력 — T/W 판정을 생략했습니다 "
            "(추진제 질량만으로 계산하면 실제의 수십 배가 나옵니다). "
            "모터 하드웨어를 포함한 건조 질량을 입력하면 판정합니다.")
    else:
        total_mass = body_mass_kg + prop_mass_kg
        tw_ratio = max_thrust / (total_mass * 9.80665) if total_mass > 0 else 0.0
        # 하한 2.0: 레일 이탈 속도 확보(핀 공력 안정). 통념은 5:1 이라 이 값은
        #           관대한 편이다.
        # 상한 10 : 저고도 max-Q 항력 손실 · 구조/전장 g 부하.
        # ⚠ 최대 추력 기준이므로 실제 이륙 T/W 보다 높게 나온다.
        if tw_ratio < 2.0:
            warnings.append(f"경고: T/W ratio {tw_ratio:.2f}가 낮습니다 (권장: >= 2.0)")
        elif tw_ratio > 10.0:
            warnings.append(f"경고: T/W ratio {tw_ratio:.2f}가 높습니다 (권장: <= 10.0)")

    # ── Kn range (KN계열 실용 권장: 150 ~ 500) ───────────────────────
    # ⚠ 고체 추진제는 정의상 연소 종료 시 A_b → 0 이므로 Kn → 0 이다.
    #   전체 시계열에 min() 을 걸면 그 마지막 한 샘플이 결과를 지배해
    #   어떤 형상이든 "Kn_min 0 < 100" 오탐이 난다(실측: 1420 샘플 중
    #   딱 1개가 0, 직전 샘플은 163.8). 소진 직후에도 챔버 잔류 가스로
    #   추력은 남아 있어 "추력 5% 이하 제외" 같은 필터로는 걸러지지 않는다.
    #   → 걸러야 하는 기준은 추력이 아니라 연소면적이다.
    #
    #   이 검사가 원래 잡으려는 것은 "연소 도중" Kn 이 떨어져 챔버압이
    #   추진제의 소염한계 아래로 내려가는 상황(처핑·중도 소화)이다.
    #   판정 구간을 "잔여 질량" 으로 잡는 이유:
    #   ① 다중 세그먼트에서는 세그먼트가 하나씩 소진되며 Kn 이 단계적으로
    #      떨어진다. 마지막 세그먼트만 남은 구간(Kn 이 1/N 로 줄어든 구간)은
    #      이미 테일오프이지 정상 운전이 아니다. 면적>0 만으로는 이 구간이
    #      그대로 들어와 "Kn_min < 100" 오탐이 난다.
    #   ② srm_multiseg 의 Kn 표시용 슬리버 테이퍼(KN_TAPER_Q)를 켜면 꼬리가
    #      더 낮게 표시되므로 면적 기준은 더욱 부적절해진다.
    #   질량 기준은 면적을 어떻게 보고하든 흔들리지 않는다.
    OPERATING_MASS_FRAC = 0.05
    Kn_all = np.asarray(Kn_arr, dtype=float)
    burning = None
    if m_g_arr is not None:
        mg = np.asarray(m_g_arr, dtype=float)[:len(Kn_all)]
        if len(mg) and mg[0] > 0:
            burning = mg > OPERATING_MASS_FRAC * mg[0]
    if burning is None:
        if A_bg_arr is not None:
            burning = np.asarray(A_bg_arr, dtype=float)[:len(Kn_all)] > 0.0
        else:
            burning = Kn_all > 0.0      # 폴백: 소진 샘플만 제외
    Kn_op = Kn_all[burning] if burning.any() else Kn_all

    if len(Kn_op) > 0:
        Kn_max = np.max(Kn_op)
        Kn_min = np.min(Kn_op)
        Kn_avg = np.mean(Kn_op)
        n_drop = len(Kn_all) - len(Kn_op)
        tail = f" (테일오프 {n_drop} 샘플 제외)" if n_drop else ""
        if Kn_max > 600:
            warnings.append(f"⚠️  Kn_max {Kn_max:.0f} > 600 — 고압으로 인한 케이스 파열 위험 (CATO 주의)")
        if Kn_max > 450:
            warnings.append(f"⚠️  Kn_max {Kn_max:.0f} > 450 — KN계열 권장 상한 초과")
        if Kn_min < 100:
            warnings.append(
                f"⚠️  연소 중 Kn_min {Kn_min:.0f} < 100{tail} — 챔버압이 "
                f"소염한계 아래로 내려가 처핑·중도 소화 가능성")
        if Kn_avg < 150:
            warnings.append(f"⚠️  Kn_avg {Kn_avg:.0f} < 150{tail} — 저 Kn으로 충분한 연소압 미달 가능")

    # Isp 합리성
    if len(Isp_arr) > 0:
        Isp_avg = np.mean(Isp_arr[Isp_arr > 0])
        if Isp_avg < 50:
            warnings.append(f"경고: 평균 Isp {Isp_avg:.1f}s < 50s (비정상적으로 낮음)")
        elif Isp_avg > 500:
            warnings.append(f"경고: 평균 Isp {Isp_avg:.1f}s > 500s (비정상적으로 높음)")

    # 추력 검증
    if max_thrust < 1.0:
        warnings.append(f"경고: 최대 추력 {max_thrust:.2f} N < 1 N (너무 약함)")

    # 질유량 검증
    max_mdot = np.max(m_dot_NOZ_arr) if len(m_dot_NOZ_arr) > 0 else 0.0
    if max_mdot < 1e-9:
        warnings.append("경고: 질유량이 거의 0입니다")

    return warnings

    # 메인
if __name__ == "__main__":
    print("f(x,y) <= 0 이 포트(가스) 내부입니다.")
    print("사용 가능: circle, box, star, U, I, D")
    print("예) circle(x,y,0,0,3)")
    print("예) U(circle(x,y,5,5,3), circle(x,y,-5,5,3))")
    print("예) star(x,y,0,0, r_inner=1.5, r_outer=3.0, k=5, sharp=8)")

    # 추진제 성질 (KN-series NAKKA reference)
    PROPELLANT = {
        "KNDX": {"Den": 1.879, "gamma": 1.131, "R": 219.7, "T_ideal": 1739, "C_star": 907},
        "KNSB": {"Den": 1.841, "gamma": 1.137, "R": 208.4, "T_ideal": 1600, "C_star": 885},
        "KNSU": {"Den": 1.889, "gamma": 1.133, "R": 196.8, "T_ideal": 1720, "C_star": 908},
        "KNER": {"Den": 1.820, "gamma": 1.143, "R": 227.0, "T_ideal": 1608, "C_star": 876},
        "KNMN": {"Den": 1.854, "gamma": 1.137, "R": 211.2, "T_ideal": 1540, "C_star": 866},
        "KNFR": {"Den": 1.803, "gamma": 1.140, "R": 218.0, "T_ideal": 1560, "C_star": 872},
        "KNPSB": {"Den": 1.832, "gamma": 1.136, "R": 205.0, "T_ideal": 1580, "C_star": 880},
    }

    P_atm = 0.101325    # 대기압 MPa

    print("포트 방정식 입력 (빈 줄 입력 시 완료):")
    exprs = []
    _i = 1
    while True:
        _e = input(f"  {_i}번 식 (완료: 빈 줄): ").strip()
        if not _e:
            if not exprs:
                print("  ※ 최소 1개 이상 입력하세요.")
                continue
            break
        exprs.append(_e)
        _i += 1

    x0, x1 = map(float, input("x 범위 (예: -15 15): ").split())
    y0, y1 = map(float, input("y 범위 (예: -15 15): ").split())

    D_c = float(input("챔버 직경(mm): ").strip())
    L_c = float(input("챔버 길이(mm): ").strip())

    # 추진제 선택
    print("사용 가능한 추진제:", ", ".join(PROPELLANT.keys()))
    Type = input("연료 종류: ").strip().upper()
    if Type not in PROPELLANT:
        Type = "KNSB"
        print(f"알 수 없는 종류. KNSB로 설정됨")

    prop = PROPELLANT[Type]
    Den_grain = prop["Den"]
    gamma = prop["gamma"]
    R_specific = prop["R"]
    T_ideal = prop["T_ideal"]
    C_star = prop["C_star"]

    D_g = float(input("그레인 직경(mm): ").strip())
    D_gc = float(input("코어 직경(mm): ").strip())
    L_gs = float(input("세그먼트 길이(mm): ").strip())
    N = int(input("세그먼트 개수: ").strip())
    Den_ratio = float(input("연료 밀도 비율(0~1): ").strip())
    P_target = float(input("최대 압력(MPa): ").strip())

    # NAKKA 다항식으로 Kn 자동 계산
    Kn_max = kn_nakka(Type, P_target)
    print(f"  → NAKKA 권장 Kn ({Type}, {P_target:.3f} MPa) = {Kn_max:.1f}")

    e_NOZ = float(input("노즐 삭마 정도: ").strip())
    eff_combustion = float(input("연소 효율(0~1): ").strip())
    t_interval = float(input("시간 간격(s): ").strip())
    t_ignition = float(input("점화 램프 시간(s) [0 = 비활성화, 권장: 0.01~0.05]: ").strip())
    ER = float(input("노즐 팽창비 Ae/At (예: 8.0): ").strip())
    NOZ_eff = float(input("노즐 효율(0~1): ").strip())

    V_c, L_g, V_g, V_load, Den_actual, m_g, A_be, A_bc, A_bg, T_actual = grain_chamber_setting(
        D_c, L_c, D_g, D_gc, L_gs, N, Den_grain, Den_ratio, T_ideal, eff_combustion)

    print("초기 조건은 다음과 같습니다.")
    print(f"연료 종류: {Type}")
    print(f"연소 효율: {eff_combustion:.2f}")
    print(f"챔버 직경: {D_c:.2f} mm")
    print(f"챔버 길이: {L_c:.2f} mm")
    print(f"챔버 부피: {V_c:.2f} mm^3")

    print(f"그레인 직경: {D_g:.2f} mm")
    print(f"코어 직경: {D_gc:.2f} mm")
    print(f"세그먼트 길이: {L_gs:.2f} mm")
    print(f"세그먼트 개수: {N}")
    print(f"그레인 길이: {L_g:.2f} mm")
    print(f"그레인 밀도: {Den_actual:.4f} g/cc")
    print(f"그레인 질량: {m_g:.4f} kg")
    print(f"그레인 부피: {V_g:.2f} mm^3")
    print(f"챔버 대비 그레인 부피 비율: {V_load:.4f}")

    print(f"그레인 끝면 연소면적: {A_be:.2f} mm^2")
    print(f"그레인 코어 연소면적: {A_bc:.2f} mm^2")
    print(f"그레인 총 연소면적: {A_bg:.2f} mm^2")  

    print(f"목표 압력: {P_target:.2f} MPa")
    print(f"초기 압력(대기압): {P_atm:.3f} MPa")
    print(f"챔버 온도: {T_actual:.2f} K")
    print(f"기체상수: {R_specific:.2f} J/(kg*K)")
    print(f"비열비: {gamma:.3f}")
    print(f"특성 속도: {C_star:.2f} m/s")
    print(f"노즐 삭마 정도: {e_NOZ:.2f}")
    print(f"노즐 효율: {NOZ_eff:.2f}")

    print(f"시간 간격: {t_interval:.3f} s")

    check = input("시작하시겠습니까? (Y/N): ").strip().lower()
    if check == 'y':
        A_NOZ_t = simulate_NOZ_t(D_gc, D_g, L_g, N, t_interval, Kn_max, r_dot_test=0.49)
        print(f"계산된 노즐 단면적: {A_NOZ_t:.2f} mm^2")
        t, Dgc, Lg, Abe, Abc, Abg, mg, mdotGEN, mdotNOZ, mdotREMAIN, denREMAIN, Pc, Rdot, Kn_list, F, Cf, Isp, shots, Xmm, Ymm, grainsdfm = simulate(
            exprs,
            (x0, x1), (y0, y1),
            V_c,
            D_g, D_gc,
            N,
            L_g,
            V_g,
            Den_actual,
            m_g,
            A_NOZ_t,
            NOZ_eff,
            R_specific,
            gamma,
            C_star,
            T_actual,
            P_target,
            P_atm,
            e_NOZ,
            a=3.907, n=0.535,
            t_end = 100.0, t_interval=t_interval,
            grid=1500,
            snapshot_dt = 0.05,
            ER=ER,
            t_ignition=t_ignition
        )
        plt.figure()
        plt.plot(t, Dgc, label="D_gc (코어 직경)")
        plt.xlabel("Time [s]")
        plt.ylabel("D_gc [mm]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Lg, label="L_g (그레인 길이)")
        plt.xlabel("Time [s]")
        plt.ylabel("L_g [mm]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Abe, label="A_be (끝면 연소면적)")
        plt.xlabel("Time [s]")
        plt.ylabel("A_be [mm²]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Abc, label="A_bc (코어 연소면적)")
        plt.xlabel("Time [s]")
        plt.ylabel("A_bc [mm²]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Abg, label="A_bg (총 연소면적)")
        plt.xlabel("Time [s]")
        plt.ylabel("A_bg [mm²]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mg, label="m_g (그레인 질량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_g [kg]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mdotGEN, label="m_dot_GEN (생성 질유량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_dot_GEN [kg/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mdotNOZ, label="m_dot_NOZ (노즐 배출 질유량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_dot_NOZ [kg/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mdotREMAIN, label="m_dot_REMAIN (남은 질유량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_dot_REMAIN [kg/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, denREMAIN, label="den_REMAIN (잔류 추진제 밀도)")
        plt.xlabel("Time [s]")
        plt.ylabel("den_REMAIN [g/cc]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Pc, label="P_c (연소실 압력)")
        plt.xlabel("Time [s]")
        plt.ylabel("P_c [MPa]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Rdot, label="R_dot (후퇴율)")
        plt.xlabel("Time [s]")
        plt.ylabel("R_dot [mm/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Kn_list, label="Kn (Knudsen 수)")
        plt.xlabel("Time [s]")
        plt.ylabel("Kn (무차원)")
        plt.grid(True)

        plt.figure()
        plt.plot(t, F, label="F (추력)")
        plt.xlabel("Time [s]")
        plt.ylabel("F [N]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Isp, label="Isp (비추력)")
        plt.xlabel("Time [s]")
        plt.ylabel("Isp [s]")
        plt.grid(True)

        # 로켓 기체 파라미터 입력 및 JSON 저장
        # ※ Ixx/Iyy/Izz, XCG, XCP, 형상 파라미터는
        #   rocket_input_builder.py Phase 2 에서 compute_inertia_table 로 자동 계산됩니다.
        #   여기서는 건조 질량만 입력받고 나머지는 placeholder(0.0 / 1) 로 채웁니다.
        print("\n로켓 기체 파라미터 입력:")
        body_mass_kg = float(input("로켓 건조 질량(kg, 모터 제외): ").strip())
        print("  ※ 나머지 파라미터(MOI, CG, 형상, XCP 등)는 rocket_input_builder.py Phase 2 에서 자동 계산됩니다.")
        ref_area_m2     = 0.0
        ref_length_m    = 0.0
        Ixx             = 0.0
        Iyy             = 0.0
        Izz             = 0.0
        XCG_initial_m   = 0.0
        XCG_final_m     = 0.0
        body_length_m   = 0.0
        nose_length_m   = 0.0
        fin_count       = 0
        fin_span_m      = 0.0
        fin_root_chord_m = 0.0
        fin_tip_chord_m  = 0.0
        XCP_m           = 0.0

        # JSON 저장
        save_srm_json(
            t, F, Pc, mdotNOZ, Isp, mg, Kn_list,
            gamma, P_atm, A_NOZ_t, ER, C_star, T_actual,
            body_mass_kg, ref_area_m2, ref_length_m,
            Ixx, Iyy, Izz, XCG_initial_m, XCG_final_m,
            body_length_m, nose_length_m, fin_count, fin_span_m,
            fin_root_chord_m, fin_tip_chord_m, XCP_m,
            D_g_mm=D_g, D_gc_0_mm=D_gc, L_gs_mm=L_gs,
            N_segments=N, Den_gcc=Den_actual, X_grain_front_m=0.0,
            output_ue="srm_rocket_ue.json",
            output_detail="srm_simul_detail.json"
        )

        # 자체 검사
        warnings = run_self_checks_srm(F, mdotNOZ, Isp, Kn_list, body_mass_kg, m_g, P_atm)
        if warnings:
            print("\n자체 검사 경고:")
            for w in warnings:
                print(f"  {w}")

        # 단면 변화(연료/포트/외벽 같이 표시)
        max_plots = 10
        if len(shots) > max_plots:
            idx = np.linspace(0, len(shots)-1, max_plots, dtype=int)
            shots_plot = [shots[i] for i in idx]
        else:
            shots_plot = shots
        plt.figure(figsize=(7, 7))
        plt.contour(Xmm, Ymm, grainsdfm, levels=[0], colors="k", linewidths=2)
    
        cmap = plt.get_cmap("tab10")
        handles = []

        if len(shots_plot) > 0:
            t0, F0 = shots_plot[0]
            port0 = (F0 <= 0)
            solid0 = (grainsdfm <= 0) & (~port0)
            plt.contourf(Xmm, Ymm, solid0.astype(float),
                         levels=[-0.5, 0.5, 1.5], alpha=0.25)

        for i, (ti, Fsnap) in enumerate(shots_plot):
            color = cmap(i % 10)
            plt.contour(
                Xmm, Ymm, 
                Fsnap, 
                levels=[0.0],
                colors=[color], 
                linewidths=1
            )
            line = mlines.Line2D([], [], color=color, linewidth=2, label=f"t={ti:.2f}s")
            handles.append(line)

        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Port Evolution Over Time")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.grid(True, alpha=0.3)
        plt.legend(handles=handles, loc="best")
        print("시뮬레이션이 완료되었습니다.")
        plt.show()
    else:
        print("시뮬레이션이 취소되었습니다.")
