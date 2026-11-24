

"""Generic LPG tank fire-exposure and pressure-relief calculation. Public synthetic examples only."""
import math, os
import numpy as np


try:
    import CoolProp.CoolProp as CP
    HAS_CP = True
except ImportError:
    CP = None
    HAS_CP = False

_KEYS = ("P", "rl", "rv", "ul", "uv", "hl", "hv", "k")


def load_table(path, xp):
    """조성 xp(프로판 몰분율) 포화선 로드. 조성 격자 사이는 선형보간."""
    d = np.load(path, allow_pickle=False)
    xs, T_all = d["x"], d["T"]
    if xp <= xs[0]:
        i, w = 0, 0.0
    elif xp >= xs[-1]:
        i, w = len(xs) - 2, 1.0
    else:
        i = int(np.searchsorted(xs, xp)) - 1
        w = (xp - xs[i]) / (xs[i+1] - xs[i])
    T = T_all[i] + w * (T_all[i+1] - T_all[i])
    out = {}
    for k in _KEYS:
        a0 = np.interp(T, T_all[i],   d[k][i])
        a1 = np.interp(T, T_all[i+1], d[k][i+1])
        out[k] = a0 + w * (a1 - a0)
    return T, out









X_PROPANE = 0.95        
X_BUTANE  = 0.05        





Q_MODE = 'empirical'    
                        


C_FIRE   = 43_200.0     
                        
                        
F_ENV    = 1.0          
AREA_EXP = 0.82         


HEAT_FLUX = 79.0e3      
                        
                        
                        




VALVES = {
    "가상 밸브 A": dict(A=100e-6, rated=2000),   
    "가상 밸브 B": dict(A=200e-6, rated=4000),   
}
P_SET   = 1.5e6        
P_CLOSE = 1.3e6        
P_ACC   = 1.8e6        
P_DESIGN = 1.5e6       
P_HYDRO  = 2.0e6       
KD      = 0.975         
                        
K_CAP   = 1.15          
VALVE_MODE = "pop"      
                        
                        




ID_TANK = 0.8         
L_CYL   = 1.2         
HEAD_R  = 4.0           




T_INIT = 28.0           
L0     = 0.85           




DT      = 0.01          
T_MAX   = 12_000        
L_END   = 0.005         
SAT_TABLE = "sat_tables.npz"   
                               
                               




SHOW_PLOT = True        
SHOW_PROGRESS = True    
SAVE_PNG  = "firecase_v6.png"   









PATM = 101_325.0


_xs = X_PROPANE + X_BUTANE
if abs(_xs - 1.0) > 1e-6:
    raise ValueError(f"조성 합이 1이 아님: {_xs}")
if X_BUTANE < 1e-9:
    FLUID = "Propane"
elif X_PROPANE < 1e-9:
    FLUID = "n-Butane"
else:
    FLUID = f"HEOS::Propane[{X_PROPANE}]&n-Butane[{X_BUTANE}]"
_BACKEND = "HEOS"
_COMPS   = FLUID.split("::")[-1] if "::" in FLUID else FLUID


ID, LC = ID_TANK, L_CYL
HD, R  = ID / HEAD_R, ID / 2
V_CYL  = math.pi / 4 * ID**2 * LC
V_HEAD = (2/3) * math.pi * R**2 * HD
V_TANK = V_CYL + 2 * V_HEAD
_e     = math.sqrt(1 - (HD / R)**2)
A_HEAD = math.pi * R**2 * (1 + ((1 - _e**2) / _e) * math.atanh(_e))
A_TOT  = math.pi * ID * LC + 2 * A_HEAD

T0 = T_INIT + 273.15





def _make_state():
    if not HAS_CP:
        raise RuntimeError("CoolProp 미설치")
    if "&" in FLUID:
        st = CP.AbstractState(_BACKEND, "Propane&n-Butane")
        st.set_mole_fractions([X_PROPANE, X_BUTANE])
    else:
        st = CP.AbstractState(_BACKEND, FLUID)
    return st





def A_wet_linear(V_liq):
    """접액면적을 현재 액체 체적분율에 비례하여 근사한다."""
    return A_TOT * (V_liq / V_TANK)





def fire_heat(A_w):
    """화재 입열 [W].  empirical → C·F·A^n  |  flux → q×A"""
    A = max(A_w, 1e-9)
    if Q_MODE == "empirical":
        return C_FIRE * F_ENV * A**AREA_EXP
    elif Q_MODE == "flux":
        return HEAT_FLUX * A
    raise ValueError(f"Q_MODE 오류: {Q_MODE}")



def _fire_label():
    if Q_MODE == "empirical":
        tag = ("배수·소방설비 有" if abs(C_FIRE - 43_200) < 1 else
               "배수·소방설비 無" if abs(C_FIRE - 70_900) < 1 else "사용자 지정")
        return f"C = {C_FIRE:,.0f} W  ({tag})"
    return f"q = {HEAT_FLUX/1000:.0f} kW/m²"





import sys, time

def _progress(label, frac, t_sim, extra="", width=28):
    """콘솔 한 줄 진행바. frac: 0~1"""
    if not SHOW_PROGRESS:
        return
    frac = min(max(frac, 0.0), 1.0)
    n = int(frac * width)
    bar = "█" * n + "░" * (width - n)
    sys.stdout.write(f"\r    {label:18s} |{bar}| {frac*100:5.1f}%  "
                     f"t={t_sim/60:5.1f}min {extra}   ")
    sys.stdout.flush()

def _progress_done(label, t_sim, elapsed):
    if not SHOW_PROGRESS:
        return
    sys.stdout.write(f"\r    {label:18s} |{'█'*28}| 100.0%  "
                     f"t={t_sim/60:5.1f}min  ({elapsed:.1f}s 소요)          \n")
    sys.stdout.flush()













_SAT_N = 600            

def _build_sat_table():
    """CoolProp 직접 생성 (테이블 파일이 없을 때만 사용)"""
    st = _make_state()
    try:
        Tc = st.T_critical()
    except Exception:
        Tc_p = CP.PropsSI("TCRIT", "", 0, "", 0, "Propane")
        Tc_b = CP.PropsSI("TCRIT", "", 0, "", 0, "n-Butane")
        Tc = X_PROPANE * Tc_p + X_BUTANE * Tc_b

    Ts = np.linspace(T0 - 25.0, Tc - 5.0, _SAT_N)
    cols = {k: np.empty(_SAT_N) for k in
            ("P", "rl", "rv", "ul", "uv", "hl", "hv", "k")}
    ok = np.ones(_SAT_N, dtype=bool)
    for i, Ti in enumerate(Ts):
        try:
            st.update(CP.QT_INPUTS, 0, Ti)
            cols["P"][i]  = st.p()
            cols["rl"][i] = st.rhomass()
            cols["ul"][i] = st.umass()
            cols["hl"][i] = st.hmass()
            st.update(CP.QT_INPUTS, 1, Ti)
            cols["rv"][i] = st.rhomass()
            cols["uv"][i] = st.umass()
            cols["hv"][i] = st.hmass()
            cols["k"][i]  = st.cpmass() / st.cvmass()
        except Exception:
            ok[i] = False
    if ok.sum() < 20:
        raise RuntimeError("포화선 테이블 생성 실패 — 조성/온도범위 확인")
    return Ts[ok], {k: v[ok] for k, v in cols.items()}


if SAT_TABLE and os.path.exists(SAT_TABLE):
    if SHOW_PROGRESS:
        print(f"포화선 테이블 로드: {SAT_TABLE}", end="", flush=True)
    _ST, _SC = load_table(SAT_TABLE, X_PROPANE)
    if SHOW_PROGRESS:
        print(f"  → {len(_ST)} pts, {_ST[0]-273.15:.0f}~{_ST[-1]-273.15:.0f}℃  [CoolProp 불필요]")
elif HAS_CP:
    if SHOW_PROGRESS:
        print("테이블 없음 → CoolProp 으로 생성 중...", end="", flush=True)
    _ST, _SC = _build_sat_table()
    if SHOW_PROGRESS:
        print(f" 완료 ({len(_ST)} pts, {_ST[0]-273.15:.0f}~{_ST[-1]-273.15:.0f}℃)")
else:
    raise RuntimeError(
        f"포화선 테이블('{SAT_TABLE}')이 없고 CoolProp 도 없습니다.\n"
        "  → build_sat_tables.py 로 테이블을 생성하거나 CoolProp 을 설치하세요.")

_TMIN, _TMAX = _ST[0], _ST[-1]
if not (_TMIN <= T0 <= _TMAX):
    raise ValueError(f"초기온도 {T0-273.15:.1f}℃ 가 테이블 범위 "
                     f"[{_TMIN-273.15:.1f}, {_TMAX-273.15:.1f}]℃ 밖입니다.")


def _ip(T, key):
    """포화 물성 보간 (스칼라)"""
    return np.interp(T, _ST, _SC[key])



_A_P  = _SC["P"];  _A_RL = _SC["rl"]; _A_RV = _SC["rv"]
_A_UL = _SC["ul"]; _A_UV = _SC["uv"]
_A_HV = _SC["hv"]; _A_HL = _SC["hl"]; _A_K  = _SC["k"]
_A_VL = 1.0 / _A_RL          
_A_VV = 1.0 / _A_RV
_N    = len(_ST)




def _u_at_node(i, v):
    """격자점 i 에서 비체적 v 에 대응하는 내부에너지"""
    vl = _A_VL[i]; vv = _A_VV[i]
    q  = (v - vl) / (vv - vl)
    return _A_UL[i] + q * (_A_UV[i] - _A_UL[i])


def uv_flash(rho, u):
    """자체 U-V flash.  반환: (T, P, q, rho_l, rho_v, h_v, k)

    [알고리즘]
      잔차 f(T) = u(T, q(T,ρ)) − u  는 T 에 대해 단조 증가.
      1) 격자 인덱스에 대한 이분탐색으로 f 의 부호가 바뀌는 구간 [i, i+1] 을 찾음
         (보간 없이 배열 접근만 → 빠름)
      2) 그 구간 안에서 f 를 선형으로 보고 T 를 1회 내삽
      3) T 로부터 나머지 물성을 선형보간
    """
    v = 1.0 / rho

    
    lo, hi = 0, _N - 1
    if _u_at_node(lo, v) - u > 0.0:          
        i = 0
    elif _u_at_node(hi, v) - u < 0.0:        
        i = _N - 2
    else:
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if _u_at_node(mid, v) - u > 0.0:
                hi = mid
            else:
                lo = mid
        i = lo

    
    f0 = _u_at_node(i,     v) - u
    f1 = _u_at_node(i + 1, v) - u
    df = f1 - f0
    w  = 0.0 if abs(df) < 1e-30 else -f0 / df
    if   w < 0.0: w = 0.0
    elif w > 1.0: w = 1.0

    T  = _ST[i] + w * (_ST[i+1] - _ST[i])

    
    rl = _A_RL[i] + w * (_A_RL[i+1] - _A_RL[i])
    rv = _A_RV[i] + w * (_A_RV[i+1] - _A_RV[i])
    P  = _A_P[i]  + w * (_A_P[i+1]  - _A_P[i])
    hv = _A_HV[i] + w * (_A_HV[i+1] - _A_HV[i])
    k  = _A_K[i]  + w * (_A_K[i+1]  - _A_K[i])

    vl = 1.0 / rl; vv = 1.0 / rv
    q  = (v - vl) / (vv - vl)
    if   q < 0.0: q = 0.0
    elif q > 1.0: q = 1.0

    return T, P, q, rl, rv, hv, k





def choked(P_abs, rho_v, k, A):
    k = min(k, K_CAP)
    return KD * A * P_abs * math.sqrt(k * rho_v / P_abs * (2/(k+1))**((k+1)/(k-1)))







def simulate(A_orifice, A_wet_fn=A_wet_linear,
             dt=DT, t_max=T_MAX, L_end=L_END, mode=VALVE_MODE, label=""):
    
    rho_l0, u_l0 = _ip(T0, "rl"), _ip(T0, "ul")
    rho_v0, u_v0 = _ip(T0, "rv"), _ip(T0, "uv")

    m = L0 * V_TANK * rho_l0 + (1 - L0) * V_TANK * rho_v0
    U = (L0 * V_TANK * rho_l0 * u_l0 + (1 - L0) * V_TANK * rho_v0 * u_v0)

    P_re = P_CLOSE
    is_open, t, rec = False, 0.0, []

    _t_start = time.time()
    _next_upd = 0.0
    _lvl_peak = L0
    _upd_every = max(int(0.05 / dt), 1)
    _step = 0

    while t <= t_max:
        
        T, P, q, rho_l, rho_v, h_v, k = uv_flash(m / V_TANK, U / m)

        if 0.0 < q < 1.0:
            V_liq = m * (1 - q) / rho_l
        else:
            V_liq = V_TANK if q <= 0 else 0.0
            k = K_CAP
        T_l = T

        level = min(V_liq / V_TANK, 1.0)
        A_w   = min(A_wet_fn(V_liq), A_TOT)
        Q     = fire_heat(A_w)

        
        Pg = P - PATM
        if mode == "pop":
            if not is_open and Pg >= P_SET: is_open = True
            elif is_open and Pg <= P_re:    is_open = False
            frac = 1.0 if is_open else 0.0
        else:
            frac = min(max((Pg - P_re) / (P_SET - P_re), 0.0), 1.0)
            is_open = frac > 0

        W = choked(P, rho_v, k, A_orifice) * frac

        rec.append((t, Pg/1e6, level*100, W*3600, Q/1000, A_w, T_l-273.15, frac))

        
        _step += 1
        if SHOW_PROGRESS and label and _step % _upd_every == 0:
            _now = time.time()
            if _now - _next_upd > 0.3:
                _lvl_peak = max(_lvl_peak, level)
                denom = max(_lvl_peak - L_end, 1e-9)
                _progress(label, (_lvl_peak - level) / denom, t,
                          f"P={Pg/1e6:.2f}MPag L={level*100:4.1f}%")
                _next_upd = _now

        if level <= L_end or level >= 0.9999:
            break

        U += (Q - W * h_v) * dt
        m -= W * dt
        t += dt

    if SHOW_PROGRESS and label:
        _progress_done(label, t, time.time() - _t_start)

    return np.array(rec)


T_, P_, LV_, W_, Q_, AW_, TL_, OP_ = range(8)





def _hvap(T_c):
    """포화 증발잠열 [J/kg]. 테이블 보간."""
    T = T_c + 273.15
    return _ip(T, "hv") - _ip(T, "hl")


def report(name, r, rated):
    i, j = r[:, P_].argmax(), r[:, W_].argmax()
    op   = r[:, OP_] > 0
    duty = op.mean() * 100
    ok   = "적합" if r[i, P_] <= P_ACC/1e6 else "★부적합★"
    Lh    = _hvap(r[i, TL_])
    W_req = r[i, Q_] * 1000 / Lh * 3600

    print(f"  ┌ {name}   →  {ok}")
    print(f"  │ Pmax       {r[i,P_]:.4f} MPag  @{r[i,T_]/60:5.2f} min   (한계 {P_ACC/1e6})")
    print(f"  │ 액온 Tl    max {r[:,TL_].max():5.2f} ℃   Pmax시점 {r[i,TL_]:5.2f} ℃   "
          f"개방후 변동폭 {(r[op][:,TL_].max()-r[op][:,TL_].min()) if op.any() else 0:4.2f} ℃   "
          f"종료 {r[-1,TL_]:5.2f} ℃")
    print(f"  │ 유량  (※순간값 — 정격과 직접 비교 금지)")
    print(f"  │   Wmax(순간)  {r[j,W_]:6.0f} kg/h    [밸브식 검증: 정격 {rated} kg/h]")
    print(f"  │   시간평균 W  {r[:,W_].mean():6.0f} kg/h    duty {duty:.0f} %")
    print(f"  │   필요방출량  {W_req:6.0f} kg/h  (= Q/Lvap @Pmax, 입열이 결정)")
    print(f"  │ 액위  초기 {L0*100:.1f} %  피크 {r[:,LV_].max():.1f} %  종료 {r[-1,LV_]:.2f} %")
    print(f"  │ 입열  초기 {r[0,Q_]:.1f} kW → 종료 {r[-1,Q_]:.1f} kW   "
          f"(Awet {r[0,AW_]:.3f} → {r[-1,AW_]:.4f} m²)")
    if op.any():
        print(f"  └ 개방 {r[op][0,T_]:.0f} s | 종료 {r[-1,T_]/60:.1f} min\n")
    else:
        print(f"  └ 밸브 미개방 | 종료 {r[-1,T_]/60:.1f} min\n")





if __name__ == "__main__":
    A0 = A_wet_linear(L0 * V_TANK)
    print(f"유체 = {_COMPS}   (프로판 {X_PROPANE:.2f} / 부탄 {X_BUTANE:.2f})")
    print(f"탱크 = 공개용 가상 예제   V_TANK = {V_TANK*1000:.0f} L   A_TOT = {A_TOT:.3f} m²")
    print(f"압력 기준: 설계/설정 {P_DESIGN/1e6:.2f} | 최대축적 {P_ACC/1e6:.2f} | "
          f"수압시험 {P_HYDRO/1e6:.2f} MPag")
    if Q_MODE == "empirical":
        print(f"입열방식 = empirical   Q = C·{F_ENV}·A^{AREA_EXP} [W]")
    else:
        print(f"입열방식 = flux        Q = q × A_wet")
    print(f"화재조건 = {_fire_label()}")
    print(f"초기 Awet({L0*100:.0f}%) = {A0:.3f} m² → Q0 = {fire_heat(A0)/1000:.1f} kW\n")

    
    try:
        rho28 = _ip(T0, "rl")
        _f = L0 * rho28 / _SC["rl"] - 1.0
        _idx = np.where(np.diff(np.sign(_f)))[0]
        if len(_idx):
            i = _idx[0]
            Tf = np.interp(0.0, [_f[i], _f[i+1]], [_ST[i], _ST[i+1]])
            Pf = _ip(Tf, "P") / 1e6 - 0.1013
            verdict = (f"축적압 {P_ACC/1e6:.2f} 초과, 도달 불가" if Pf > P_ACC/1e6
                       else "★도달 가능 — 주의★")
            print(f"[액충만 조건] T = {Tf-273.15:.1f} ℃,  P = {Pf:.3f} MPag  → {verdict}\n")
        else:
            print("[액충만 조건] 테이블 범위 내 미도달\n")
    except Exception as e:
        print(f"[액충만 조건] 점검 생략 ({type(e).__name__})\n")

    print("═"*70); print(f" {_fire_label()}"); print("═"*70)
    results = {}
    for name, v in VALVES.items():
        r = simulate(v["A"], label=name)
        results[name] = r
        report(name, r, v["rated"])

    
    try:
        import matplotlib
        if not SHOW_PLOT:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
            if any(f in x.name for x in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = f; break
        plt.rcParams["axes.unicode_minus"] = False

        if SHOW_PROGRESS:
            print("\n[그래프용 modulating 모드 계산]")
        fig, ax = plt.subplots(3, 2, figsize=(12, 10.5))
        for (name, v), col in zip(VALVES.items(), ["crimson", "navy"]):
            r  = results[name]                                   
            rm = simulate(v["A"], mode="mod", label=f"{name[:8]} mod")
            t, tm = r[:, T_]/60, rm[:, T_]/60
            dts = r[1, T_] - r[0, T_]
            cum = np.cumsum(r[:, W_] / 3600.0) * dts

            ax[0,0].plot(t,  r[:,P_],  col, lw=.6, alpha=.45)
            ax[0,0].plot(tm, rm[:,P_], col, lw=1.4, label=name)
            ax[0,1].plot(t,  r[:,TL_], col, lw=.6, alpha=.45)
            ax[0,1].plot(tm, rm[:,TL_],col, lw=1.4, label=name)
            ax[1,0].plot(t,  r[:,LV_], col, label=name)
            ax[1,1].plot(t,  r[:,W_],  col, lw=.7)
            ax[2,0].plot(t,  cum,      col, lw=1.6, label=f"{name}  ({cum[-1]:.0f} kg)")
            ax[2,1].plot(t,  r[:,Q_],  col)

        m0 = L0 * V_TANK * _ip(T0, "rl")
        ax[2,0].axhline(m0, ls="--", c="k", lw=.8)
        ax[2,0].text(1, m0*1.01, f"초기 액 인벤토리 {m0:.0f} kg", fontsize=8)

        
        ax[0,0].axhline(P_HYDRO/1e6,  ls="-",  c="purple", lw=1.3, alpha=.8)
        ax[0,0].axhline(P_ACC/1e6,    ls="--", c="red",    lw=1.5)
        ax[0,0].axhline(P_DESIGN/1e6, ls="-.", c="darkorange", lw=1.2)
        ax[0,0].axhline(P_SET/1e6,    ls=":",  c="gray",   lw=1.0)
        _xr = ax[0,0].get_xlim()[1]
        ax[0,0].text(_xr*0.02, P_HYDRO/1e6+0.02,  f"수압시험압 {P_HYDRO/1e6:.2f}",
                     fontsize=7.5, color="purple")
        ax[0,0].text(_xr*0.02, P_ACC/1e6+0.02,    f"최대축적압 {P_ACC/1e6:.2f}",
                     fontsize=7.5, color="red")
        ax[0,0].text(_xr*0.02, P_DESIGN/1e6-0.09, f"설계압/설정압 {P_DESIGN/1e6:.2f}",
                     fontsize=7.5, color="darkorange")
        ax[0,0].set_ylim(0, max(P_HYDRO/1e6*1.15, ax[0,0].get_ylim()[1]))

        titles = ["Pressure [MPag]  (연=pop, 진=modulating)", "Liquid Temperature [℃]",
                  "Liquid Level [%]", "SV Mass Flow (instant) [kg/h]",
                  "Cumulative Discharge [kg]", "Fire Heat Input [kW]"]
        for a, ti in zip(ax.flat, titles):
            a.set_xlabel("Time [min]"); a.set_ylabel(ti); a.grid(alpha=.3)
        for a in (ax[0,0], ax[0,1], ax[1,0], ax[2,0]):
            a.legend(fontsize=8)

        fig.suptitle(f"LPG 0.5t Fire Case — {_COMPS} | {_fire_label()} | dynamic wetted area", y=.995)
        fig.tight_layout()

        if SAVE_PNG:
            fig.savefig(SAVE_PNG, dpi=130); print(f"saved: {SAVE_PNG}")

        
        print(f"\n[누적 방출량]  초기 액 인벤토리 = {m0:.1f} kg")
        for name, v in VALVES.items():
            r = results[name]
            dts = r[1,T_]-r[0,T_]
            cum = np.cumsum(r[:,W_]/3600.0)*dts
            half = np.argmax(cum >= cum[-1]/2)
            print(f"  {name:16s} 총 {cum[-1]:6.1f} kg | 50% 배출 {r[half,T_]/60:5.1f} min | "
                  f"종료 {r[-1,T_]/60:.1f} min | 평균 {cum[-1]/(r[-1,T_]/3600):.0f} kg/h")

        if SHOW_PLOT:
            print("\n그래프 창을 닫으면 프로그램이 종료됩니다.")
            plt.show()

    except ImportError:
        pass
