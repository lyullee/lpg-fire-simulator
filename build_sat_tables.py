#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 포화선 테이블 생성기  —  build_sat_tables.py
==============================================================================
 목적
   CoolProp 으로 프로판/부탄 혼합물의 포화선을 조성 × 온도 2차원 그리드로
   미리 계산해 .npz 로 저장한다.
   → 배포본(GUI/웹/exe)은 CoolProp 없이 이 파일만 로드해서 동작한다.

 왜 필요한가
   - 시뮬레이션 루프는 CoolProp 을 전혀 호출하지 않는다(uv_flash 는 보간만 사용).
   - CoolProp 이 쓰이는 곳은 오직 '포화선 테이블 생성' 한 곳뿐.
   - 따라서 테이블을 미리 뽑아두면 배포본에서 CoolProp 의존성이 사라진다.
       · PyInstaller exe : ~200 MB → ~40 MB
       · Streamlit       : C++ 빌드 불필요
       · Pyodide(브라우저): 서버 없이 실행 가능

 출력
   sat_tables.npz
     x        : (Nx,)        프로판 몰분율 그리드
     T        : (Nx, NT)     온도 [K]  (조성마다 임계온도가 달라 행별로 다름)
     P,rl,rv,ul,uv,hl,hv,k  : (Nx, NT)  포화 물성
     meta     : 생성 조건 기록

 사용
   python build_sat_tables.py                 # 기본 그리드
   python build_sat_tables.py --nx 51 --nt 800 --out my_tables.npz
   python build_sat_tables.py --xmin 0.0 --xmax 1.0    # 부탄 100% 까지
==============================================================================
"""
import argparse
import time
import sys
import numpy as np
import CoolProp.CoolProp as CP


# ══════════════════════════════════════════════════════════════
#  설정 (명령행으로 덮어쓰기 가능)
# ══════════════════════════════════════════════════════════════
DEF = dict(
    xmin=0.50,      # 프로판 몰분율 최소
    xmax=1.00,      # 프로판 몰분율 최대
    nx=51,          # 조성 격자 수  (0.50~1.00, 0.01 간격)
    nt=600,         # 조성당 온도 격자 수
    tmin_c=-20.0,   # 온도 하한 [℃]
    tmargin=5.0,    # 임계온도로부터 띄울 여유 [K]
    out="sat_tables.npz",
)

KEYS = ("P", "rl", "rv", "ul", "uv", "hl", "hv", "k")


# ══════════════════════════════════════════════════════════════
#  유틸
# ══════════════════════════════════════════════════════════════
def make_state(xp):
    """프로판 몰분율 xp 에 대한 AbstractState"""
    xb = 1.0 - xp
    if xb < 1e-12:
        return CP.AbstractState("HEOS", "Propane"), "Propane"
    if xp < 1e-12:
        return CP.AbstractState("HEOS", "n-Butane"), "n-Butane"
    st = CP.AbstractState("HEOS", "Propane&n-Butane")
    st.set_mole_fractions([xp, xb])
    return st, f"Propane[{xp:.3f}]&n-Butane[{xb:.3f}]"


def critical_T(st, xp):
    """임계온도. 혼합물은 T_critical() 실패 가능 → 몰분율 가중 근사"""
    try:
        return st.T_critical()
    except Exception:
        Tc_p = CP.PropsSI("TCRIT", "", 0, "", 0, "Propane")     # 369.89 K
        Tc_b = CP.PropsSI("TCRIT", "", 0, "", 0, "n-Butane")    # 425.13 K
        return xp * Tc_p + (1.0 - xp) * Tc_b


def progress(i, n, xp, t0, width=34):
    frac = (i + 1) / n
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    el = time.time() - t0
    eta = el / frac - el if frac > 1e-9 else 0.0
    sys.stdout.write(
        f"\r  |{bar}| {frac*100:5.1f}%   "
        f"x_C3={xp:.3f}   경과 {el/60:4.1f}분   잔여 {eta/60:4.1f}분   "
    )
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════
#  단일 조성 포화선
# ══════════════════════════════════════════════════════════════
def build_one(xp, nt, tmin_c, tmargin):
    """반환: (T[nt], {key: arr[nt]})  — 실패점은 이웃 보간으로 메움"""
    st, name = make_state(xp)
    Tc = critical_T(st, xp)

    Tmin = tmin_c + 273.15
    Tmax = Tc - tmargin
    if Tmax <= Tmin:
        raise RuntimeError(f"x={xp}: 온도 범위 무효 (Tmax {Tmax:.1f} <= Tmin {Tmin:.1f})")

    T = np.linspace(Tmin, Tmax, nt)
    col = {k: np.full(nt, np.nan) for k in KEYS}

    for i, Ti in enumerate(T):
        try:
            st.update(CP.QT_INPUTS, 0, Ti)            # 포화액
            col["P"][i]  = st.p()
            col["rl"][i] = st.rhomass()
            col["ul"][i] = st.umass()
            col["hl"][i] = st.hmass()

            st.update(CP.QT_INPUTS, 1, Ti)            # 포화증기
            col["rv"][i] = st.rhomass()
            col["uv"][i] = st.umass()
            col["hv"][i] = st.hmass()
            col["k"][i]  = st.cpmass() / st.cvmass()
        except Exception:
            pass                                      # NaN 유지 → 아래서 메움

    # NaN 보간 (양끝은 최근접값으로 채움)
    ok = ~np.isnan(col["P"])
    n_bad = nt - ok.sum()
    if ok.sum() < nt * 0.5:
        raise RuntimeError(f"x={xp}: 유효점 부족 ({ok.sum()}/{nt})")
    if n_bad:
        for k in KEYS:
            col[k] = np.interp(T, T[ok], col[k][ok])

    return T, col, name, n_bad


# ══════════════════════════════════════════════════════════════
#  전체 그리드
# ══════════════════════════════════════════════════════════════
def build_all(cfg):
    xs = np.linspace(cfg["xmin"], cfg["xmax"], cfg["nx"])
    nt = cfg["nt"]

    T_all = np.empty((cfg["nx"], nt))
    dat = {k: np.empty((cfg["nx"], nt)) for k in KEYS}

    print("=" * 74)
    print(" 포화선 테이블 생성  (Propane / n-Butane)")
    print("=" * 74)
    print(f" 조성 그리드 : x_C3 = {cfg['xmin']:.3f} ~ {cfg['xmax']:.3f}  "
          f"({cfg['nx']} 점, 간격 {(cfg['xmax']-cfg['xmin'])/max(cfg['nx']-1,1):.4f})")
    print(f" 온도 그리드 : {cfg['tmin_c']:.0f}℃ ~ (Tc − {cfg['tmargin']:.0f}K)  "
          f"({nt} 점)")
    print(f" 총 계산점   : {cfg['nx'] * nt:,} × 2(액/기)")
    print(f" 출력        : {cfg['out']}")
    print("-" * 74)

    t0 = time.time()
    total_bad = 0
    for i, xp in enumerate(xs):
        try:
            T, col, name, n_bad = build_one(xp, nt, cfg["tmin_c"], cfg["tmargin"])
        except Exception as e:
            print(f"\n  ✘ x_C3={xp:.3f} 실패: {e}")
            raise
        T_all[i] = T
        for k in KEYS:
            dat[k][i] = col[k]
        total_bad += n_bad
        progress(i, cfg["nx"], xp, t0)

    el = time.time() - t0
    print(f"\n{'-'*74}")
    print(f" 완료: {el/60:.2f} 분  (보간 보정점 {total_bad:,} / {cfg['nx']*nt:,})")

    meta = np.array([
        f"generated={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"CoolProp={CP.get_global_param_string('version')}",
        f"x_range={cfg['xmin']}~{cfg['xmax']} nx={cfg['nx']}",
        f"T_min_C={cfg['tmin_c']} T_margin_K={cfg['tmargin']} nt={nt}",
        "note=x is Propane mole fraction; T rows differ per composition (Tc varies)",
    ])

    np.savez_compressed(cfg["out"], x=xs, T=T_all, meta=meta, **dat)

    import os
    mb = os.path.getsize(cfg["out"]) / 1e6
    print(f" 저장: {cfg['out']}   ({mb:.2f} MB)")
    print("=" * 74)
    return xs, T_all, dat


# ══════════════════════════════════════════════════════════════
#  검증 (저장본 재로드 → CoolProp 직접값과 대조)
# ══════════════════════════════════════════════════════════════
def verify(path, n_check=6):
    print("\n[검증] 저장본 vs CoolProp 직접 계산")
    print("-" * 74)
    d = np.load(path, allow_pickle=True)
    xs, T_all = d["x"], d["T"]

    rng = np.random.default_rng(0)
    err = {k: [] for k in ("P", "rl", "rv")}
    print(f"{'x_C3':>7}{'T[C]':>8}{'P_tab':>10}{'P_CP':>10}{'err%':>9}"
          f"{'rl_err%':>10}{'rv_err%':>10}")
    for _ in range(n_check):
        i = rng.integers(0, len(xs))
        j = rng.integers(int(0.1*T_all.shape[1]), int(0.9*T_all.shape[1]))
        xp, T = xs[i], T_all[i, j]
        st, _ = make_state(xp)
        st.update(CP.QT_INPUTS, 0, T); P_cp, rl_cp = st.p(), st.rhomass()
        st.update(CP.QT_INPUTS, 1, T); rv_cp = st.rhomass()

        eP = (d["P"][i, j]/P_cp - 1)*100
        el = (d["rl"][i, j]/rl_cp - 1)*100
        ev = (d["rv"][i, j]/rv_cp - 1)*100
        err["P"].append(abs(eP)); err["rl"].append(abs(el)); err["rv"].append(abs(ev))
        print(f"{xp:7.3f}{T-273.15:8.2f}{d['P'][i,j]/1e6:10.5f}{P_cp/1e6:10.5f}"
              f"{eP:9.5f}{el:10.5f}{ev:10.5f}")
    print("-" * 74)
    print(f" 최대오차   P {max(err['P']):.6f}%   ρl {max(err['rl']):.6f}%   "
          f"ρv {max(err['rv']):.6f}%")
    print(" (격자점 위 값이므로 오차 0 이 정상. 0 이 아니면 저장/로드 문제)")


# ══════════════════════════════════════════════════════════════
#  조성축 보간 로더 예시  (배포본에서 사용할 코드)
# ══════════════════════════════════════════════════════════════
def load_for_composition(path, xp):
    """임의 프로판 몰분율 xp 에 대해 (T, {key:arr}) 반환.
       조성 격자 사이는 선형 보간.
       ※ 배포본(CoolProp 없는 환경)에서 이 함수만 있으면 된다."""
    d = np.load(path, allow_pickle=True)
    xs, T_all = d["x"], d["T"]

    if xp <= xs[0]:
        i, w = 0, 0.0
    elif xp >= xs[-1]:
        i, w = len(xs) - 2, 1.0
    else:
        i = int(np.searchsorted(xs, xp)) - 1
        w = (xp - xs[i]) / (xs[i+1] - xs[i])

    # 온도축이 조성마다 달라 → 공통 온도축으로 재보간
    T0 = T_all[i]; T1 = T_all[i+1]
    T = T0 + w * (T1 - T0)                    # 목표 온도축

    out = {}
    for k in KEYS:
        a0 = np.interp(T, T_all[i],   d[k][i])
        a1 = np.interp(T, T_all[i+1], d[k][i+1])
        out[k] = a0 + w * (a1 - a0)
    return T, out


def demo_interp(path):
    """조성축 보간 정확도 확인 (격자 사이 값 → CoolProp 대조)"""
    print("\n[조성축 보간 정확도] 격자 사이 조성에서 CoolProp 대조")
    print("-" * 74)
    d = np.load(path, allow_pickle=True)
    xs = d["x"]
    if len(xs) < 2:
        print(" 격자 1개 → 생략"); return
    # 격자 사이 임의 조성
    xp = 0.5 * (xs[len(xs)//2] + xs[len(xs)//2 + 1])
    T, tab = load_for_composition(path, xp)

    st, name = make_state(xp)
    print(f" 테스트 조성: {name}")
    print(f"{'T[C]':>8}{'P_interp':>12}{'P_CP':>12}{'err%':>9}"
          f"{'rl_err%':>10}{'rv_err%':>10}")
    for j in np.linspace(int(0.15*len(T)), int(0.85*len(T)), 5).astype(int):
        Tj = T[j]
        try:
            st.update(CP.QT_INPUTS, 0, Tj); P_cp, rl_cp = st.p(), st.rhomass()
            st.update(CP.QT_INPUTS, 1, Tj); rv_cp = st.rhomass()
        except Exception:
            continue
        print(f"{Tj-273.15:8.2f}{tab['P'][j]/1e6:12.5f}{P_cp/1e6:12.5f}"
              f"{(tab['P'][j]/P_cp-1)*100:9.4f}"
              f"{(tab['rl'][j]/rl_cp-1)*100:10.4f}"
              f"{(tab['rv'][j]/rv_cp-1)*100:10.4f}")
    print("-" * 74)
    print(" → 조성 격자를 촘촘히 할수록 이 오차가 줄어든다.")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="포화선 테이블 생성기")
    ap.add_argument("--xmin", type=float, default=DEF["xmin"], help="프로판 몰분율 최소")
    ap.add_argument("--xmax", type=float, default=DEF["xmax"], help="프로판 몰분율 최대")
    ap.add_argument("--nx", type=int, default=DEF["nx"], help="조성 격자 수")
    ap.add_argument("--nt", type=int, default=DEF["nt"], help="조성당 온도 격자 수")
    ap.add_argument("--tmin-c", type=float, default=DEF["tmin_c"], help="온도 하한 [℃]")
    ap.add_argument("--tmargin", type=float, default=DEF["tmargin"], help="임계온도 여유 [K]")
    ap.add_argument("--out", type=str, default=DEF["out"], help="출력 파일")
    ap.add_argument("--no-verify", action="store_true", help="검증 생략")
    a = ap.parse_args()

    cfg = dict(xmin=a.xmin, xmax=a.xmax, nx=a.nx, nt=a.nt,
               tmin_c=a.tmin_c, tmargin=a.tmargin, out=a.out)

    build_all(cfg)

    if not a.no_verify:
        verify(cfg["out"])
        demo_interp(cfg["out"])

    print("\n다음 단계:")
    print(f"  배포본에서  T, tab = load_for_composition('{cfg['out']}', x_propane)")
    print("  → 이후 uv_flash / simulate 는 CoolProp 없이 동작")