

"""Generic LPG tank fire-exposure and pressure-relief calculation. Public synthetic examples only."""
import os, math, time, io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import base64
from pathlib import Path






TANK_PRESETS = {
    "가상 소형 탱크": dict(id=0.8, l=1.2, v=None, verified=False, note="공개용 가상 치수"),
    "가상 중형 탱크": dict(id=1.0, l=2.0, v=None, verified=False, note="공개용 가상 치수"),
    "직접 입력": dict(id=0.8, l=1.2, v=None, verified=False, note="사용자가 지정한 치수"),
}


def _setup_font():
    for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if any(cand in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            return cand
    for p in ("fonts/NanumGothic.ttf", "NanumGothic.ttf"):
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = "NanumGothic"
            return "NanumGothic (bundled)"
    return None


_FONT = _setup_font()
plt.rcParams["axes.unicode_minus"] = False

PATM = 101_325.0
T_, P_, LV_, W_, Q_, AW_, TL_, OP_ = range(8)
KEYS = ("P", "rl", "rv", "ul", "uv", "hl", "hv", "k")
SAT_TABLE = "sat_tables.npz"
PALETTE = ["crimson", "navy", "darkgreen", "darkorange"]


def _r(v, nd=0):
    """NaN-safe round -> 표시용. 유효하지 않으면 '—'"""
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(fv):
        return "—"
    return round(fv, nd) if nd else round(fv)





@st.cache_resource(show_spinner=False)
def load_raw(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in ("x", "T", *KEYS)}


@st.cache_data(show_spinner=False)
def load_table(path, xp):
    """조성 xp(프로판 몰분율) 포화선. 조성축 선형보간."""
    d = load_raw(path)
    xs, T_all = d["x"], d["T"]
    if xp <= xs[0]:
        i, w = 0, 0.0
    elif xp >= xs[-1]:
        i, w = len(xs) - 2, 1.0
    else:
        i = int(np.searchsorted(xs, xp)) - 1
        w = (xp - xs[i]) / (xs[i + 1] - xs[i])
    T = T_all[i] + w * (T_all[i + 1] - T_all[i])
    out = {}
    for k in KEYS:
        a0 = np.interp(T, T_all[i], d[k][i])
        a1 = np.interp(T, T_all[i + 1], d[k][i + 1])
        out[k] = a0 + w * (a1 - a0)
    return T, out





class Engine:
    def __init__(self, p, ST, SC):
        self.p = p
        self.diverged = None        
        self.liquid_full_at = None  
        self._ST, self._SC = ST, SC
        self._VL = 1.0 / SC["rl"]
        self._VV = 1.0 / SC["rv"]
        self._N = len(ST)

        ID, LC = p["id_tank"], p["l_cyl"]
        HD, R = ID / p["head_r"], ID / 2
        self.V_TANK = math.pi / 4 * ID ** 2 * LC + 2 * (2 / 3) * math.pi * R ** 2 * HD
        e = math.sqrt(1 - (HD / R) ** 2)
        A_HEAD = math.pi * R ** 2 * (1 + ((1 - e ** 2) / e) * math.atanh(e))
        self.A_TOT = math.pi * ID * LC + 2 * A_HEAD

        self.T0 = p["t_init"] + 273.15
        self.P_ACC = p["p_acc"]  
        if not (ST[0] <= self.T0 <= ST[-1]):
            raise ValueError(
                f"초기온도 {p['t_init']:.1f}℃ 가 테이블 범위 "
                f"[{ST[0]-273.15:.1f}, {ST[-1]-273.15:.1f}]℃ 밖입니다.")

    def _ip(self, T, k):
        return np.interp(T, self._ST, self._SC[k])

    def _u_node(self, i, v):
        vl, vv = self._VL[i], self._VV[i]
        q = (v - vl) / (vv - vl)
        ul, uv = self._SC["ul"][i], self._SC["uv"][i]
        return ul + q * (uv - ul)

    def uv_flash(self, rho, u):
        v = 1.0 / rho
        lo, hi = 0, self._N - 1
        if self._u_node(lo, v) - u > 0.0:
            i = 0
        elif self._u_node(hi, v) - u < 0.0:
            i = self._N - 2
        else:
            while hi - lo > 1:
                mid = (lo + hi) >> 1
                if self._u_node(mid, v) - u > 0.0:
                    hi = mid
                else:
                    lo = mid
            i = lo
        f0 = self._u_node(i, v) - u
        f1 = self._u_node(i + 1, v) - u
        df = f1 - f0
        w = 0.0 if abs(df) < 1e-30 else -f0 / df
        w = min(max(w, 0.0), 1.0)
        S, C = self._ST, self._SC
        T = S[i] + w * (S[i + 1] - S[i])
        rl = C["rl"][i] + w * (C["rl"][i + 1] - C["rl"][i])
        rv = C["rv"][i] + w * (C["rv"][i + 1] - C["rv"][i])
        P = C["P"][i] + w * (C["P"][i + 1] - C["P"][i])
        hv = C["hv"][i] + w * (C["hv"][i + 1] - C["hv"][i])
        k = C["k"][i] + w * (C["k"][i + 1] - C["k"][i])
        q = (v - 1.0 / rl) / (1.0 / rv - 1.0 / rl)
        return T, P, min(max(q, 0.0), 1.0), rl, rv, hv, k

    def A_wet(self, V_liq):
        """선형 접액면적 근사: A_TOT x 액위 (선형)"""
        return self.A_TOT * (V_liq / self.V_TANK)

    def fire_heat(self, A_w):
        p = self.p
        A = max(A_w, 1e-9)
        if p["q_mode"] == "empirical":
            return p["c_fire"] * p["f_env"] * A ** p["area_exp"]
        return p["heat_flux"] * A

    def fire_label(self):
        p = self.p
        if p["q_mode"] == "empirical":
            c = p["c_fire"]
            tag = ("소방설비 有" if abs(c - 43200) < 1 else
                   "소방설비 無" if abs(c - 70900) < 1 else "사용자")
            return f"C = {c:,.0f} W ({tag})"
        return f"q = {p['heat_flux']/1000:.0f} kW/m2"

    def choked(self, P_abs, rho_v, k, A):
        k = min(k, self.p["k_cap"])
        return (self.p["kd"] * A * P_abs
                * math.sqrt(k * rho_v / P_abs * (2 / (k + 1)) ** ((k + 1) / (k - 1))))

    def hvap(self, T_c):
        T = T_c + 273.15
        return self._ip(T, "hv") - self._ip(T, "hl")

    def liquid_full(self):
        rho0 = self._ip(self.T0, "rl")
        f = self.p["l0"] * rho0 / self._SC["rl"] - 1.0
        idx = np.where(np.diff(np.sign(f)))[0]
        if not len(idx):
            return None, None, "테이블 범위 내 미도달"
        i = idx[0]
        Tf = np.interp(0.0, [f[i], f[i + 1]], [self._ST[i], self._ST[i + 1]])
        Pf = self._ip(Tf, "P") / 1e6 - 0.1013
        reach = Pf <= self.P_ACC / 1e6
        msg = ("★도달 가능 - 주의★" if reach
               else f"축적압 {self.P_ACC/1e6:.2f} 초과, 도달 불가")
        return Tf - 273.15, Pf, msg

    def simulate(self, A_or, mode, cb=None):
        p = self.p
        dt, L0, V = p["dt"], p["l0"], self.V_TANK
        rl0, ul0 = self._ip(self.T0, "rl"), self._ip(self.T0, "ul")
        rv0, uv0 = self._ip(self.T0, "rv"), self._ip(self.T0, "uv")
        m = L0 * V * rl0 + (1 - L0) * V * rv0
        U = L0 * V * rl0 * ul0 + (1 - L0) * V * rv0 * uv0

        is_open, t, rec = False, 0.0, []
        lvl_peak, step = L0, 0
        upd = max(int(1.0 / dt), 1)
        LVL_FULL = 0.995      

        while t <= p["t_max"]:
            
            if m <= 1e-6 or U <= 0.0 or not (np.isfinite(m) and np.isfinite(U)):
                self.diverged = f"질량/에너지 발산 (t = {t/60:.2f} min)"
                break

            T, P, q, rl, rv, hv, k = self.uv_flash(m / V, U / m)

            
            if not (np.isfinite(T) and np.isfinite(P) and np.isfinite(rl)
                    and np.isfinite(rv) and rl > 0.0):
                self.diverged = f"물성 계산 발산 (t = {t/60:.2f} min)"
                break

            if 0.0 < q < 1.0:
                V_liq = m * (1 - q) / rl
            else:
                V_liq = V if q <= 0 else 0.0
                k = p["k_cap"]

            level = min(V_liq / V, 1.0)

            
            if level >= LVL_FULL:
                self.liquid_full_at = t
                break

            A_w = min(self.A_wet(V_liq), self.A_TOT)
            Q = self.fire_heat(A_w)

            Pg = P - PATM
            if mode == "pop":
                if not is_open and Pg >= p["p_set"]:
                    is_open = True
                elif is_open and Pg <= p["p_close"]:
                    is_open = False
                frac = 1.0 if is_open else 0.0
            else:
                frac = min(max((Pg - p["p_close"]) / (p["p_set"] - p["p_close"]), 0.0), 1.0)

            W = self.choked(P, rv, k, A_or) * frac
            if not np.isfinite(W):
                self.diverged = f"방출유량 발산 (t = {t/60:.2f} min)"
                break

            
            rec.append((t, Pg / 1e6, level * 100, W * 3600, Q / 1000, A_w, T - 273.15, frac))

            step += 1
            if cb and step % upd == 0:
                lvl_peak = max(lvl_peak, level)
                cb(min(max((lvl_peak - level) / max(lvl_peak - p["l_end"], 1e-9), 0), 1),
                   t, Pg / 1e6, level * 100)

            if level <= p["l_end"]:
                break

            U += (Q - W * hv) * dt
            m -= W * dt
            t += dt

        if len(rec) < 2:
            if not self.diverged:
                self.diverged = "유효 데이터 부족 (초기 스텝에서 종료)"
            return np.empty((0, 8))
        return np.array(rec)

    def summary(self, r, rated):
        """유효 행이 없으면 None 반환 (호출부에서 '계산 실패' 처리)"""
        if r is None or len(r) < 2:
            return None
        good = np.isfinite(r).all(axis=1)
        if good.sum() < 2:
            return None
        r = r[good]

        i, j = r[:, P_].argmax(), r[:, W_].argmax()
        op = r[:, OP_] > 0
        Lh = self.hvap(r[i, TL_])
        wreq = (r[i, Q_] * 1000 / Lh * 3600) if (np.isfinite(Lh) and Lh > 0) else float("nan")
        return dict(
            ok=bool(r[i, P_] <= self.P_ACC / 1e6),
            pmax=r[i, P_], t_pmax=r[i, T_] / 60,
            tl_max=r[:, TL_].max(), tl_at=r[i, TL_],
            wmax=r[j, W_], wmean=r[:, W_].mean(), duty=op.mean() * 100,
            wreq=wreq,
            lvl_peak=r[:, LV_].max(), lvl_end=r[-1, LV_],
            q0=r[0, Q_], qend=r[-1, Q_],
            t_open=(r[op][0, T_] if op.any() else None),
            t_end=r[-1, T_] / 60, rated=rated,
        )





st.set_page_config(page_title="Pressure Safety Valve Simulator for LPG Fire Cases",
                   page_icon=("assets/favicon.png"
                              if os.path.exists("assets/favicon.png") else "🔥"),
                   layout="wide")


st.title("LPG 탱크 화재·안전밸브 비교 — 가상 예제")
st.caption("API STD 521 Fire Case · 포화물성 기반 U-V flash · dynamic wetted area")

if not os.path.exists(SAT_TABLE):
    st.error(f"물성 테이블 `{SAT_TABLE}` 이 없습니다. "
             "`python build_sat_tables.py` 로 먼저 생성하세요.")
    st.stop()

with st.sidebar:
    st.header("시뮬레이션 환경 설정")

    
    st.markdown("""
                <style>
                /* 사이드바 스크롤 컨테이너의 첫 자식(버튼 블록)을 고정 */
                section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]
                    > div:has(div[data-testid="stButton"]) {
                    position: sticky;
                    top: 0;
                    z-index: 999;
                    background-color: var(--secondary-background-color);
                    padding: 0.5rem 0 0.25rem 0;
                }

                /* 계산 실행 버튼: 글자 크게, 굵게 */
                section[data-testid="stSidebar"] div[data-testid="stButton"] button {
                    height: 3rem;
                }
                section[data-testid="stSidebar"] div[data-testid="stButton"] button p {
                    font-size: 1.15rem;
                    font-weight: 700;
                    letter-spacing: 0.02em;
                }
                </style>
            """, unsafe_allow_html=True)

    run = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.divider()

    with st.expander("① 유체 조성", expanded=False):
        xp = st.slider("프로판 몰분율", 0.50, 1.00, 0.95, 0.01)
        st.caption(f"n-부탄 = {1-xp:.2f}")

    with st.expander("② 화재 입열", expanded=False):
        q_mode = st.radio("산출 방식", ["empirical", "flux"], horizontal=True,
                          help="empirical: Q=C·F·A^n (API 521) / flux: Q=q×A")
        if q_mode == "empirical":
            c_fire = st.selectbox(
                "C 상수 [W]", [43200.0, 70900.0], index=1,
                format_func=lambda x: f"{x:,.0f}  ({'배수·소방설비 有' if x==43200 else '배수·소방설비 無'})")
            f_env = st.number_input("F 환경계수", 0.02, 1.0, 1.0, 0.01,
                                    help="1.0=나관, 0.3=수분무/저단열")
            area_exp = st.number_input("n 면적지수", 0.5, 1.0, 0.82, 0.01,
                                       help="0.82=개방 풀화재, 1.0=밀폐")
            heat_flux = 79e3
        else:
            heat_flux = st.select_slider(
                "열유속 [kW/m2]", [41, 50, 79, 150], 79,
                format_func=lambda x: f"{x} kW/m2") * 1e3
            c_fire, f_env, area_exp = 43200.0, 1.0, 0.82

    with st.expander("③ 안전밸브", expanded=False):
        n_valve = st.radio("검토 대상", ["단일 밸브", "두 밸브 비교"], index=1, horizontal=True)

        def valve_input(key, def_name, def_A, def_rated):
            """밸브 1개 입력. 반환 (밸브명, 면적[m2], 정격[kg/h])"""
            name = st.text_input("밸브명", def_name, key=f"N_{key}")
            c1, c2 = st.columns(2)
            A = c1.number_input("오리피스 [mm²]", 0.1, 9999.0, def_A, 0.01,
                                format="%.2f", key=f"A_{key}")
            rated = c2.number_input("정격 [kg/h]", 1, 999999, def_rated, 1,
                                    key=f"R_{key}")
            return (name.strip() or def_name), A * 1e-6, rated

        st.caption("　오리피스 유효면적 —  양정식: A = πDL　|　전량식: A = πD²/4")

        if n_valve == "단일 밸브":
            VALVES = [valve_input("v1", "가상 밸브 A", 100.0, 2000)]
        else:
            st.markdown("**밸브 A**")
            v1 = valve_input("v1", "가상 밸브 A", 100.0, 2000)
            st.divider()
            st.markdown("**밸브 B**")
            v2 = valve_input("v2", "가상 밸브 B", 200.0, 4000)
            if v1[0] == v2[0]:
                v2 = (v2[0] + " (B)", v2[1], v2[2])   
            VALVES = [v1, v2]

        st.divider()
        c1, c2 = st.columns(2)
        p_set = c1.number_input("설정압 [MPag]", 0.1, 5.0, 1.5, 0.01) * 1e6
        p_close = c2.number_input("재폐압 [MPag]", 0.1, 5.0, 1.3, 0.01) * 1e6

        
        acc_base = st.radio(
            "축적압 기준", ["설계압 기준", "설정압 기준"], horizontal=True,
            help="설계압 기준: KGS AC111 (P_design × 배수)  |  "
                 "설정압 기준: API 521 / ASME (P_set × 배수)")
        acc_mult = st.number_input(
            "축적압 배수", 1.00, 1.50,
            1.20 if acc_base == "설계압 기준" else 1.21, 0.01,
            key=f"accmult_{acc_base}",  
            help="KGS 1.20 (20% 과압) / API 1.21 (21% 과압)")

        c1, c2 = st.columns(2)
        kd = c1.number_input("배출계수 Kd", 0.5, 1.0, 0.975, 0.005)
        k_cap = c2.number_input("비열비 상한", 1.0, 1.7, 1.15, 0.01)
        valve_mode = st.radio("밸브 거동", ["pop", "mod"], horizontal=True,
                              help="pop = 팝액션(채터링) / mod = 모듈레이팅")

    with st.expander("④ 탱크"):
        tank_name = st.selectbox("탱크 규격", list(TANK_PRESETS), index=0)
        tp = TANK_PRESETS[tank_name]
        custom_tank = (tank_name == "직접 입력")

        c1, c2 = st.columns(2)
        id_tank = c1.number_input("내경 [m]", 0.1, 5.0, float(tp["id"]), 0.001,
                                  format="%.4f", key=f"ID_{tank_name}",
                                  disabled=not custom_tank)
        l_cyl = c2.number_input("실린더 길이 [m]", 0.1, 20.0, float(tp["l"]), 0.001,
                                format="%.4f", key=f"L_{tank_name}",
                                disabled=not custom_tank)
        head_r = st.number_input("경판비 (ID/깊이)", 1.0, 10.0, 4.0, 0.5,
                                 key=f"H_{tank_name}", disabled=not custom_tank)

        
        _HD, _R = id_tank / head_r, id_tank / 2
        _V = math.pi / 4 * id_tank**2 * l_cyl + 2 * (2/3) * math.pi * _R**2 * _HD
        _e = math.sqrt(1 - (_HD / _R)**2)
        _Ah = math.pi * _R**2 * (1 + ((1 - _e**2) / _e) * math.atanh(_e))
        _A = math.pi * id_tank * l_cyl + 2 * _Ah

        c1, c2 = st.columns(2)
        c1.metric("내용적", f"{_V*1000:.0f} L",
                  delta=(f"{_V*1000 - tp['v']:+.0f} L vs 규격표" if tp["v"] else None))
        c2.metric("전체 표면적", f"{_A:.3f} m²")
        st.caption(f"　{tp['note']}")

        st.divider()
        c1, c2 = st.columns(2)
        p_design = c1.number_input("설계압 [MPag]", 0.1, 5.0, 1.5, 0.01) * 1e6
        p_hydro = c2.number_input("수압시험압 [MPag]", 0.1, 6.0, 2.0, 0.01) * 1e6

    with st.expander("⑤ 초기조건"):
        t_init = st.number_input("초기 온도 [℃]", -20.0, 60.0, 28.0, 0.5)
        l0 = st.slider("초기 액위", 0.10, 0.95, 0.85, 0.01)

    with st.expander("⑥ 수치"):
        
        _q_ref = heat_flux if q_mode == "flux" else 79e3
        _strong = _q_ref >= 100e3

        precise = st.checkbox("정밀 계산 (dt=0.01s, 느림)", value=_strong,
                              help="시간간격에 대한 수렴성 확인이 필요합니다. "
                                   "열유속 100 kW/m² 이상에서는 권장.")
        dt = 0.01 if precise else 0.5
        if _strong and not precise:
            st.warning(f"⚠ 열유속 {_q_ref/1000:.0f} kW/m² — 정밀 계산 권장")
        st.caption(f"dt = {dt} s")
        t_max = st.number_input("최대 해석시간 [s]", 600, 30000, 12000, 600)
        l_end = st.number_input("종료 액위", 0.001, 0.1, 0.005, 0.001, format="%.3f")

    
    _base_p = p_design if acc_base == "설계압 기준" else p_set
    p_acc = _base_p * acc_mult

    st.divider()
    st.divider()
    st.caption("연구·교육용 가상 예제")

    with st.expander("모델과 적용 범위"):
        st.markdown("가상 입력을 사용하는 연구·교육용 계산입니다. 고정 전체조성 포화선 보간과 균일 평형을 가정하며, 기상 소멸 시 계산을 중단합니다. 실제 시설이나 제조사 제품의 성능 검증 결과가 아닙니다.")
    st.caption(f"v1.0　·　{time.strftime('%Y-%m-%d')} 빌드")


P = dict(x_propane=xp, q_mode=q_mode, c_fire=c_fire, f_env=f_env,
         area_exp=area_exp, heat_flux=heat_flux,
         p_set=p_set, p_close=p_close, p_acc=p_acc,     
         kd=kd, k_cap=k_cap,
         id_tank=id_tank, l_cyl=l_cyl, head_r=head_r,
         t_init=t_init, l0=l0, dt=dt, t_max=float(t_max), l_end=l_end)


if not run:
    st.info("👈 좌측에서 시뮬에이션 환경을 설정하고 **Run Simulation**을 누르세요.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 선택된 밸브")
        for nm, A, rt in VALVES:
            st.markdown(f"- **{nm}** — A = {A*1e6:.1f} mm², 정격 {rt:,} kg/h")
    with c2:
        st.markdown("#### 이 도구의 범위")
        st.markdown("""
| | 내용 |
|---|---|
| **하는 것** | 액-기 평형 · 화재 입열 · 증기 방출 · 압력 추적 → **축적압 이내인가?** (과압 검증) |
| **하지 않는 것** | 벽 금속온도 · UTS 저하 · BLEVE 파열 판정 |
        """)
    with st.expander("ℹ️  해석 특성"):
        st.markdown("""
- 화재 입열은 액위에 따라 접액면적이 변하는 **동적 입열**입니다.
  (`A_wet = A_TOT × 액위`, 선형 접액면적 근사)
- 액위↓ → 접액면적↓ → 입열↓ 의 **자기 안정화(self-limiting)** 효과가 반영됩니다.
- `Wmax` 는 **순간값**이며 정격과 직접 비교하면 안 됩니다 (duty cycle 참조).
- 물성은 사전 생성한 포화선 테이블(`sat_tables.npz`)을 보간합니다.
  새 입력조건의 정확도는 별도 검증이 필요합니다.
- 작은 탱크 + 강한 화재에서는 액이 팽창해 **기상이 소멸**할 수 있습니다.
  이 경우 해석을 중단하고 사유를 표시합니다.
        """)
    st.stop()



try:
    ST, SC = load_table(SAT_TABLE, xp)
    eng = Engine(P, ST, SC)
except Exception as e:
    st.error(f"초기화 오류: {e}")
    st.stop()

A0 = eng.A_wet(l0 * eng.V_TANK)
c1, c2, c3, c4 = st.columns(4)
c1.metric("탱크 내용적", f"{eng.V_TANK*1000:.0f} L")
c2.metric("전체 표면적", f"{eng.A_TOT:.3f} m²")
c3.metric("초기 접액면적", f"{A0:.3f} m²")
c4.metric("초기 입열 Q0", f"{eng.fire_heat(A0)/1000:.1f} kW")

Tf, Pf, msg = eng.liquid_full()
if Tf is not None:
    (st.warning if "★" in msg else st.info)(
        f"**[액충만 조건]** T = {Tf:.1f} ℃, P = {Pf:.3f} MPag → {msg}")

res, prog, stat = {}, st.progress(0.0), st.empty()
n_total = len(VALVES) * 2
t0 = time.time()

for n, (nm, A, rated) in enumerate(VALVES):
    def cb(f, t, p_, l, _n=n, _nm=nm):
        prog.progress((_n + f) / n_total)
        stat.text(f"{_nm} 계산 중… {f*100:.0f}%   t={t/60:.1f} min   "
                  f"P={p_:.3f} MPag   L={l:.1f}%")
    r = eng.simulate(A, valve_mode, cb)
    res[nm] = dict(r=r, s=eng.summary(r, rated))

other = "mod" if valve_mode == "pop" else "pop"
for n, (nm, A, _) in enumerate(VALVES):
    stat.text(f"{nm} · {other} 모드 계산 중…")
    prog.progress((len(VALVES) + n + 1) / n_total)
    res[nm]["r2"] = eng.simulate(A, other)

prog.progress(1.0)
stat.success(f"완료 — {time.time()-t0:.1f} 초  |  화재조건: {eng.fire_label()}  |  "
             f"판정 한계 = 축적압 {eng.P_ACC/1e6:.3f} MPag "
             f"({acc_base} × {acc_mult:.2f})")


if eng.liquid_full_at is not None:
    st.error(
        f"⚠ **액충만 발생** (t = {eng.liquid_full_at/60:.2f} 분)  \n"
        "액이 팽창해 기상부가 소멸했습니다. 이 지점에서 해석을 중단했습니다.  \n\n"
        "→ 해당 밸브는 이 화재조건에서 **압력을 통제하지 못합니다.** "
        "표시된 Pmax 는 중단 시점까지의 값입니다.")
elif eng.diverged:
    st.error(
        f"⚠ **계산 발산** — {eng.diverged}  \n\n"
        "‘⑥ 수치’ 에서 **정밀 계산(dt=0.01s)** 을 켜거나, "
        "화재조건·탱크 치수를 확인하세요.")


if all(res[nm]["s"] is None for nm, _, _ in VALVES):
    st.warning("유효한 계산 결과가 없어 그래프와 표를 표시할 수 없습니다.")
    st.stop()



st.subheader("판정 결과")
cols = st.columns(len(VALVES))
for col, (nm, A, rated) in zip(cols, VALVES):
    s = res[nm]["s"]
    with col:
        if s is None:
            st.error(f"### ⚠ {nm} — 계산 실패")
            st.write("해석이 중단되어 판정할 수 없습니다.")
            st.write(f"- 오리피스 **{A*1e6:.1f} mm²**  ·  정격 **{rated:,} kg/h**")
            continue
        if s["ok"]:
            st.success(f"### ✅ {nm} — 적합")
        else:
            st.error(f"### ❌ {nm} — 부적합")
        st.metric("최대 상승압력 Pmax", f"{_r(s['pmax'],4)} MPag",
                  delta=f"{s['pmax']-eng.P_ACC/1e6:+.4f} vs 축적압",
                  delta_color="inverse")
        st.write(f"- 오리피스 **{A*1e6:.1f} mm²**  ·  정격 **{rated:,} kg/h**")
        st.write(f"- 도달 시각 **{_r(s['t_pmax'],2)} 분**")
        st.write(f"- 최대 액온 **{_r(s['tl_max'],2)} ℃**")
        st.write(f"- 필요방출량 **{_r(s['wreq'])} kg/h** (입열이 결정)")
        st.write(f"- Wmax(순간) **{_r(s['wmax'])} kg/h**  ·  duty **{_r(s['duty'])}%**")
        st.caption("※ Wmax 는 순간값 — 정격과 직접 비교 금지")

if len(VALVES) == 2:
    n1, n2 = VALVES[0][0], VALVES[1][0]
    s1, s2 = res[n1]["s"], res[n2]["s"]
    if s1 is None or s2 is None:
        st.warning("한쪽 밸브의 해석이 중단되어 ΔPmax 를 비교할 수 없습니다.")
    else:
        dP = s2["pmax"] - s1["pmax"]
        st.info(
            f"**밸브 비교**　ΔPmax = **{dP:+.4f} MPag**　"
            f"({n1} {s1['pmax']:.4f} → {n2} {s2['pmax']:.4f})　|　"
            f"판정: {n1} **{'적합' if s1['ok'] else '부적합'}** / "
            f"{n2} **{'적합' if s2['ok'] else '부적합'}**")

with st.expander("상세 수치"):
    rows = []
    for nm, A, rated in VALVES:
        s = res[nm]["s"]
        if s is None:
            rows.append({"밸브": nm, "판정": "계산 실패",
                         "오리피스 [mm²]": round(A * 1e6, 1),
                         "정격 [kg/h]": rated})
            continue
        rows.append({
            "밸브": nm,
            "판정": "적합" if s["ok"] else "부적합",
            "오리피스 [mm²]": round(A * 1e6, 1),
            "Pmax [MPag]": _r(s["pmax"], 4),
            "Pmax 시각 [min]": _r(s["t_pmax"], 2),
            "액온 max [℃]": _r(s["tl_max"], 2),
            "Wmax 순간 [kg/h]": _r(s["wmax"]),
            "시간평균 W [kg/h]": _r(s["wmean"]),
            "duty [%]": _r(s["duty"]),
            "필요방출량 [kg/h]": _r(s["wreq"]),
            "정격 [kg/h]": rated,
            "액위 피크 [%]": _r(s["lvl_peak"], 1),
            "밸브개방 [s]": (_r(s["t_open"]) if s["t_open"] else "미개방"),
            "종료 [min]": _r(s["t_end"], 1),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)



st.subheader("결과 그래프")
fig, ax = plt.subplots(3, 2, figsize=(13, 10))

for (nm, _, _), col in zip(VALVES, PALETTE):
    r, r2 = res[nm]["r"], res[nm]["r2"]
    if len(r) < 2:
        continue                      
    t = r[:, T_] / 60
    dts = r[1, T_] - r[0, T_]
    cum = np.cumsum(r[:, W_] / 3600.0) * dts
    ax[0, 0].plot(t, r[:, P_], col, lw=.6, alpha=.45)
    ax[0, 1].plot(t, r[:, TL_], col, lw=.6, alpha=.45)
    if len(r2) >= 2:                  
        t2 = r2[:, T_] / 60
        ax[0, 0].plot(t2, r2[:, P_], col, lw=1.4, label=nm)
        ax[0, 1].plot(t2, r2[:, TL_], col, lw=1.4, label=nm)
    else:
        ax[0, 0].plot(t, r[:, P_], col, lw=1.4, label=nm)
        ax[0, 1].plot(t, r[:, TL_], col, lw=1.4, label=nm)
    ax[1, 0].plot(t, r[:, LV_], col, label=nm)
    ax[1, 1].plot(t, r[:, W_], col, lw=.7, label=nm)
    ax[2, 0].plot(t, cum, col, lw=1.6, label=f"{nm} ({cum[-1]:.0f} kg)")
    ax[2, 1].plot(t, r[:, Q_], col, label=nm)

a = ax[0, 0]
a.set_xlim(left=0)        
a.margins(x=0)
a.axhline(p_hydro / 1e6, ls="-", c="purple", lw=1.3, alpha=.8)
a.axhline(eng.P_ACC / 1e6, ls="--", c="red", lw=1.5)
a.axhline(p_design / 1e6, ls="-.", c="darkorange", lw=1.2)
xr = a.get_xlim()[1]
a.text(xr * .02, p_hydro / 1e6 + .02, f"수압시험압 {p_hydro/1e6:.2f}", fontsize=7.5, color="purple")
a.text(xr * .02, eng.P_ACC / 1e6 + .02, f"최대축적압 {eng.P_ACC/1e6:.3f}", fontsize=7.5, color="red")
a.text(xr * .02, p_design / 1e6 - .09, f"설계압 {p_design/1e6:.2f}", fontsize=7.5, color="darkorange")
a.set_ylim(0, max(p_hydro / 1e6 * 1.15, a.get_ylim()[1]))

m0 = l0 * eng.V_TANK * eng._ip(eng.T0, "rl")
ax[2, 0].axhline(m0, ls="--", c="k", lw=.8)
ax[2, 0].text(.5, m0 * 1.01, f"초기 액 인벤토리 {m0:.0f} kg", fontsize=8)

titles = [f"Pressure [MPag] (연={valve_mode}, 진={other})", "Liquid Temperature [℃]",
          "Liquid Level [%]", "SV Mass Flow (instant) [kg/h]",
          "Cumulative Discharge [kg]", "Fire Heat Input [kW]"]
for axi, ti in zip(ax.flat, titles):
    axi.set_xlabel("Time [min]"); axi.set_ylabel(ti)
    axi.grid(alpha=.3)
    axi.set_xlim(left=0)          
    axi.margins(x=0)              
    if axi.get_legend_handles_labels()[0]:
        axi.legend(fontsize=7)

fig.suptitle(f"LPG {eng.V_TANK*1000:.0f}L Fire Case — Propane[{xp:.2f}] | "
             f"{eng.fire_label()} | dynamic wetted area", y=.995)
fig.tight_layout()
st.pyplot(fig)



c1, c2 = st.columns(2)
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150)
c1.download_button("📥 그래프 PNG", buf.getvalue(), "firecase.png", "image/png",
                   use_container_width=True)

csv = io.StringIO()
csv.write("valve,t_s,P_MPag,level_%,W_kgh,Q_kW,A_wet_m2,T_liq_C,open_frac\n")
for nm, _, _ in VALVES:
    for row in res[nm]["r"]:
        if not np.isfinite(row).all():
            continue
        csv.write(f"{nm}," + ",".join(f"{v:.5g}" for v in row) + "\n")
c2.download_button("📥 시계열 CSV", csv.getvalue(), "firecase.csv", "text/csv",
                   use_container_width=True)

if _FONT is None:
    st.caption("⚠️ 한글 폰트를 찾지 못해 그래프 라벨이 깨질 수 있습니다. "
               "`fonts/NanumGothic.ttf` 를 추가하세요.")
