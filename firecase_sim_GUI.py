

"""Generic LPG tank fire-exposure and pressure-relief calculation. Public synthetic examples only."""
import math, time, threading, queue, traceback, os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import font_manager


try:
    import CoolProp.CoolProp as CP
    HAS_CP = True
except ImportError:
    CP = None
    HAS_CP = False


SAT_TABLE = "sat_tables.npz"     
KEYS = ("P", "rl", "rv", "ul", "uv", "hl", "hv", "k")


def load_table(path, xp):
    """조성 xp(프로판 몰분율)에 대한 포화선.  반환 (T[K], {key: arr})
       조성 격자 사이는 선형보간."""
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
    for k in KEYS:
        a0 = np.interp(T, T_all[i],   d[k][i])
        a1 = np.interp(T, T_all[i+1], d[k][i+1])
        out[k] = a0 + w * (a1 - a0)
    return T, out


for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

PATM = 101_325.0
T_, P_, LV_, W_, Q_, AW_, TL_, OP_ = range(8)





class FireCaseEngine:
    def __init__(self, p, log=print, prog=None, stop_flag=None):
        self.p = p
        self.log = log
        self.prog = prog or (lambda *a: None)
        self.stop = stop_flag or (lambda: False)
        self._setup()

    
    def _setup(self):
        p = self.p
        xp, xb = p["x_propane"], p["x_butane"]
        if abs(xp + xb - 1.0) > 1e-6:
            raise ValueError(f"조성 합이 1이 아닙니다: {xp + xb:.4f}")

        if xb < 1e-9:
            self.fluid, self.comps = "Propane", "Propane"
        elif xp < 1e-9:
            self.fluid, self.comps = "n-Butane", "n-Butane"
        else:
            self.fluid = f"HEOS::Propane[{xp}]&n-Butane[{xb}]"
            self.comps = f"Propane[{xp:.2f}]&n-Butane[{xb:.2f}]"

        
        ID, LC = p["id_tank"], p["l_cyl"]
        HD, R = ID / p["head_r"], ID / 2
        self.ID, self.LC = ID, LC
        self.V_TANK = math.pi / 4 * ID**2 * LC + 2 * (2/3) * math.pi * R**2 * HD
        e = math.sqrt(1 - (HD / R)**2)
        A_HEAD = math.pi * R**2 * (1 + ((1 - e**2) / e) * math.atanh(e))
        self.A_TOT = math.pi * ID * LC + 2 * A_HEAD

        self.T0 = p["t_init"] + 273.15
        self.P_ACC = p["p_set"] * (1 + p["overpressure"])

        self.log(f"유체     : {self.comps}")
        self.log(f"탱크     : V = {self.V_TANK*1000:.0f} L,  A_TOT = {self.A_TOT:.3f} m²")

        tbl = p.get("sat_table", SAT_TABLE)
        if tbl and os.path.exists(tbl):
            self.log(f"포화선 테이블 로드: {tbl}")
            self._ST, self._SC = load_table(tbl, xp)
            self.log(f"  → 완료 ({len(self._ST)} pts, "
                     f"{self._ST[0]-273.15:.0f}~{self._ST[-1]-273.15:.0f}℃)  [CoolProp 불필요]")
        elif HAS_CP:
            self.log("포화선 테이블 없음 → CoolProp 으로 직접 생성 중...")
            self._build_table_cp()
            self.log(f"  → 완료 ({len(self._ST)} pts, "
                     f"{self._ST[0]-273.15:.0f}~{self._ST[-1]-273.15:.0f}℃)")
        else:
            raise RuntimeError(
                f"포화선 테이블('{tbl}')이 없고 CoolProp 도 설치되지 않았습니다.\n"
                "build_sat_tables.py 로 테이블을 생성하거나 CoolProp 을 설치하세요.")

        self._VL = 1.0 / self._SC["rl"]
        self._VV = 1.0 / self._SC["rv"]
        self._N = len(self._ST)

        
        if not (self._ST[0] <= self.T0 <= self._ST[-1]):
            raise ValueError(
                f"초기온도 {self.T0-273.15:.1f}℃ 가 테이블 범위 "
                f"[{self._ST[0]-273.15:.1f}, {self._ST[-1]-273.15:.1f}]℃ 밖입니다.")

    def _make_state(self):
        if not HAS_CP:
            raise RuntimeError("CoolProp 미설치")
        if "&" in self.fluid:
            st = CP.AbstractState("HEOS", "Propane&n-Butane")
            st.set_mole_fractions([self.p["x_propane"], self.p["x_butane"]])
        else:
            st = CP.AbstractState("HEOS", self.fluid)
        return st

    def _build_table_cp(self):
        """CoolProp 직접 생성 (테이블 파일이 없을 때 fallback)"""
        N = self.p["sat_n"]
        st = self._make_state()
        try:
            Tc = st.T_critical()
        except Exception:
            Tc = (self.p["x_propane"] * CP.PropsSI("TCRIT", "", 0, "", 0, "Propane")
                  + self.p["x_butane"] * CP.PropsSI("TCRIT", "", 0, "", 0, "n-Butane"))
        Ts = np.linspace(self.T0 - 25.0, Tc - 5.0, N)
        col = {k: np.empty(N) for k in KEYS}
        ok = np.ones(N, bool)
        for i, Ti in enumerate(Ts):
            try:
                st.update(CP.QT_INPUTS, 0, Ti)
                col["P"][i], col["rl"][i] = st.p(), st.rhomass()
                col["ul"][i], col["hl"][i] = st.umass(), st.hmass()
                st.update(CP.QT_INPUTS, 1, Ti)
                col["rv"][i], col["uv"][i] = st.rhomass(), st.umass()
                col["hv"][i] = st.hmass()
                col["k"][i] = st.cpmass() / st.cvmass()
            except Exception:
                ok[i] = False
        if ok.sum() < 20:
            raise RuntimeError("포화선 테이블 생성 실패 — 조성/온도 범위 확인")
        self._ST = Ts[ok]
        self._SC = {k: v[ok] for k, v in col.items()}

    def _ip(self, T, key):
        return np.interp(T, self._ST, self._SC[key])

    
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
                if self._u_node(mid, v) - u > 0.0: hi = mid
                else: lo = mid
            i = lo
        f0 = self._u_node(i, v) - u
        f1 = self._u_node(i + 1, v) - u
        df = f1 - f0
        w = 0.0 if abs(df) < 1e-30 else -f0 / df
        w = min(max(w, 0.0), 1.0)

        S, C = self._ST, self._SC
        T  = S[i] + w * (S[i+1] - S[i])
        rl = C["rl"][i] + w * (C["rl"][i+1] - C["rl"][i])
        rv = C["rv"][i] + w * (C["rv"][i+1] - C["rv"][i])
        P  = C["P"][i]  + w * (C["P"][i+1]  - C["P"][i])
        hv = C["hv"][i] + w * (C["hv"][i+1] - C["hv"][i])
        k  = C["k"][i]  + w * (C["k"][i+1]  - C["k"][i])
        q  = (v - 1.0/rl) / (1.0/rv - 1.0/rl)
        return T, P, min(max(q, 0.0), 1.0), rl, rv, hv, k

    
    def A_wet(self, V_liq):
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
            tag = ("배수·소방설비 有" if abs(c - 43200) < 1 else
                   "배수·소방설비 無" if abs(c - 70900) < 1 else "사용자 지정")
            return f"C = {c:,.0f} W ({tag})"
        return f"q = {p['heat_flux']/1000:.0f} kW/m²"

    def choked(self, P_abs, rho_v, k, A):
        k = min(k, self.p["k_cap"])
        return (self.p["kd"] * A * P_abs
                * math.sqrt(k * rho_v / P_abs * (2/(k+1))**((k+1)/(k-1))))

    def hvap(self, T_c):
        T = T_c + 273.15
        return self._ip(T, "hv") - self._ip(T, "hl")

    def liquid_full_check(self):
        try:
            rho0 = self._ip(self.T0, "rl")
            f = self.p["l0"] * rho0 / self._SC["rl"] - 1.0
            idx = np.where(np.diff(np.sign(f)))[0]
            if not len(idx):
                return "테이블 범위 내 미도달"
            i = idx[0]
            Tf = np.interp(0.0, [f[i], f[i+1]], [self._ST[i], self._ST[i+1]])
            Pf = self._ip(Tf, "P") / 1e6 - 0.1013
            v = ("축적압 %.2f 초과, 도달 불가" % (self.P_ACC/1e6)) \
                if Pf > self.P_ACC/1e6 else "★도달 가능 — 주의★"
            return f"T = {Tf-273.15:.1f} ℃, P = {Pf:.3f} MPag  →  {v}"
        except Exception as e:
            return f"점검 생략 ({type(e).__name__})"

    
    def simulate(self, A_or, mode, label=""):
        p = self.p
        dt, L0 = p["dt"], p["l0"]

        
        rl0 = self._ip(self.T0, "rl"); ul0 = self._ip(self.T0, "ul")
        rv0 = self._ip(self.T0, "rv"); uv0 = self._ip(self.T0, "uv")

        V = self.V_TANK
        m = L0*V*rl0 + (1-L0)*V*rv0
        U = L0*V*rl0*ul0 + (1-L0)*V*rv0*uv0

        is_open, t, rec = False, 0.0, []
        lvl_peak, t0w, nxt = L0, time.time(), 0.0
        upd = max(int(0.05/dt), 1); step = 0

        while t <= p["t_max"]:
            if self.stop():
                raise KeyboardInterrupt("사용자 중단")
            T, P, q, rl, rv, hv, k = self.uv_flash(m/V, U/m)
            if 0.0 < q < 1.0:
                V_liq = m*(1-q)/rl
            else:
                V_liq = V if q <= 0 else 0.0
                k = p["k_cap"]

            level = min(V_liq/V, 1.0)
            A_w = min(self.A_wet(V_liq), self.A_TOT)
            Q = self.fire_heat(A_w)

            Pg = P - PATM
            if mode == "pop":
                if not is_open and Pg >= p["p_set"]: is_open = True
                elif is_open and Pg <= p["p_close"]: is_open = False
                frac = 1.0 if is_open else 0.0
            else:
                frac = min(max((Pg - p["p_close"])/(p["p_set"] - p["p_close"]), 0.0), 1.0)

            W = self.choked(P, rv, k, A_or) * frac
            rec.append((t, Pg/1e6, level*100, W*3600, Q/1000, A_w, T-273.15, frac))

            step += 1
            if label and step % upd == 0:
                now = time.time()
                if now - nxt > 0.15:
                    lvl_peak = max(lvl_peak, level)
                    pr = (lvl_peak - level) / max(lvl_peak - p["l_end"], 1e-9)
                    self.prog(label, min(max(pr, 0), 1), t, Pg/1e6, level*100)
                    nxt = now

            if level <= p["l_end"] or level >= 0.9999:
                break
            U += (Q - W*hv) * dt
            m -= W * dt
            t += dt

        if label:
            self.prog(label, 1.0, t, Pg/1e6, level*100)
            self.log(f"    {label:16s} 완료  t = {t/60:.1f} min  ({time.time()-t0w:.1f}s)")
        return np.array(rec)

    
    def report(self, name, r, rated):
        i, j = r[:, P_].argmax(), r[:, W_].argmax()
        op = r[:, OP_] > 0
        ok = r[i, P_] <= self.P_ACC/1e6
        Lh = self.hvap(r[i, TL_])
        W_req = r[i, Q_] * 1000 / Lh * 3600
        L = self.log
        L("")
        L(f"┌ {name}   →  {'적합' if ok else '★ 부적합 ★'}")
        L(f"│ Pmax       {r[i,P_]:.4f} MPag  @ {r[i,T_]/60:.2f} min   (한계 {self.P_ACC/1e6:.2f})")
        L(f"│ 액온       max {r[:,TL_].max():.2f} ℃   Pmax시점 {r[i,TL_]:.2f} ℃   "
          f"종료 {r[-1,TL_]:.2f} ℃")
        L(f"│ 유량 (※순간값 — 정격과 직접 비교 금지)")
        L(f"│   Wmax(순간)  {r[j,W_]:.0f} kg/h    [정격 {rated} kg/h]")
        L(f"│   시간평균 W  {r[:,W_].mean():.0f} kg/h    duty {op.mean()*100:.0f} %")
        L(f"│   필요방출량  {W_req:.0f} kg/h  (= Q/Lvap @Pmax)")
        L(f"│ 액위  초기 {self.p['l0']*100:.1f} %  피크 {r[:,LV_].max():.1f} %  "
          f"종료 {r[-1,LV_]:.2f} %")
        L(f"│ 입열  {r[0,Q_]:.1f} → {r[-1,Q_]:.1f} kW   "
          f"(Awet {r[0,AW_]:.3f} → {r[-1,AW_]:.4f} m²)")
        if op.any():
            L(f"└ 밸브개방 {r[op][0,T_]:.0f} s | 종료 {r[-1,T_]/60:.1f} min")
        else:
            L(f"└ 밸브 미개방 | 종료 {r[-1,T_]/60:.1f} min")
        return ok, r[i, P_]





DEFAULTS = dict(
    x_propane=1.00, x_butane=0.00,
    q_mode="empirical", c_fire=43_200.0, f_env=1.0, area_exp=0.82, heat_flux=79.0e3,
    p_set=1.5e6, p_close=1.3e6, overpressure=0.21,
    p_design=1.5e6, p_hydro=2.0e6, kd=0.975, k_cap=1.15, valve_mode="pop",
    a_lift=100e-6, rated_lift=2000, a_full=200e-6, rated_full=4000,
    id_tank=0.8, l_cyl=1.2, head_r=4.0,
    t_init=28.0, l0=0.85,
    dt=0.01, t_max=12000.0, l_end=0.005, sat_n=600,
    sat_table=SAT_TABLE,
)


class App:
    def __init__(self, root):
        self.root = root
        root.title("LPG 소형저장탱크 화재 안전밸브 과압 검증")
        root.geometry("1500x920")

        self.q = queue.Queue()
        self.vars = {}
        self._stop = False
        self.running = False

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(main, width=380)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_params(left)
        self._build_output(right)
        self.root.after(80, self._pump)

    
    def _build_params(self, parent):
        cv = tk.Canvas(parent, width=360, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=cv.yview)
        fr = ttk.Frame(cv)
        fr.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=fr, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def sec(title):
            f = ttk.LabelFrame(fr, text=title, padding=8)
            f.pack(fill="x", padx=6, pady=4)
            return f

        def row(par, key, label, unit="", width=12):
            f = ttk.Frame(par); f.pack(fill="x", pady=1)
            ttk.Label(f, text=label, width=20).pack(side="left")
            v = tk.StringVar(value=str(DEFAULTS[key]))
            self.vars[key] = v
            ttk.Entry(f, textvariable=v, width=width).pack(side="left")
            if unit:
                ttk.Label(f, text=unit, width=8).pack(side="left")

        
        s = sec("① 유체 조성 (몰분율, 합=1)")
        row(s, "x_propane", "프로판", "mol")
        row(s, "x_butane", "n-부탄", "mol")

        
        s = sec("② 화재 입열")
        f = ttk.Frame(s); f.pack(fill="x", pady=2)
        ttk.Label(f, text="산출 방식", width=20).pack(side="left")
        self.vars["q_mode"] = tk.StringVar(value=DEFAULTS["q_mode"])
        cb = ttk.Combobox(f, textvariable=self.vars["q_mode"], width=13,
                          values=["empirical", "flux"], state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._toggle_qmode())

        self.f_emp = ttk.Frame(s); self.f_emp.pack(fill="x")
        ttk.Label(self.f_emp, text="  Q = C·F·A^n   (API 521)",
                  foreground="#555").pack(anchor="w", pady=(4, 2))
        row(self.f_emp, "c_fire", "  C  상수", "W")
        ttk.Label(self.f_emp, text="   43,200 = 소방설비 有 / 70,900 = 無",
                  foreground="#777", font=("", 8)).pack(anchor="w")
        row(self.f_emp, "f_env", "  F  환경계수", "-")
        row(self.f_emp, "area_exp", "  n  면적지수", "-")

        self.f_flux = ttk.Frame(s)
        ttk.Label(self.f_flux, text="  Q = q × A_wet",
                  foreground="#555").pack(anchor="w", pady=(4, 2))
        row(self.f_flux, "heat_flux", "  q  열유속", "W/m²")
        ttk.Label(self.f_flux, text="   150e3 제트 / 79e3 포위 / 50e3 풀화재",
                  foreground="#777", font=("", 8)).pack(anchor="w")
        self._toggle_qmode()

        
        s = sec("③ 안전밸브")
        row(s, "p_set", "설정압 P_set", "Pag")
        row(s, "p_close", "재폐압 P_close", "Pag")
        row(s, "overpressure", "화재 과압률", "-")
        row(s, "kd", "배출계수 Kd", "-")
        row(s, "k_cap", "비열비 상한", "-")
        f = ttk.Frame(s); f.pack(fill="x", pady=2)
        ttk.Label(f, text="밸브 거동", width=20).pack(side="left")
        self.vars["valve_mode"] = tk.StringVar(value=DEFAULTS["valve_mode"])
        ttk.Combobox(f, textvariable=self.vars["valve_mode"], width=13,
                     values=["pop", "mod"], state="readonly").pack(side="left")
        ttk.Separator(s).pack(fill="x", pady=4)
        row(s, "a_lift", "양정식 면적", "m²")
        row(s, "rated_lift", "양정식 정격", "kg/h")
        row(s, "a_full", "전량식 면적", "m²")
        row(s, "rated_full", "전량식 정격", "kg/h")

        
        s = sec("④ 탱크")
        row(s, "id_tank", "내경 ID", "m")
        row(s, "l_cyl", "실린더 길이", "m")
        row(s, "head_r", "경판비 (ID/깊이)", "-")
        row(s, "p_design", "설계압", "Pag")
        row(s, "p_hydro", "수압시험압", "Pag")

        
        s = sec("⑤ 초기조건")
        row(s, "t_init", "초기 온도", "℃")
        row(s, "l0", "초기 액위", "-")

        
        s = sec("⑥ 수치 설정")
        row(s, "dt", "시간스텝 dt", "s")
        row(s, "t_max", "최대 해석시간", "s")
        row(s, "l_end", "종료 액위", "-")

        
        s = sec("⑦ 물성 테이블")
        f = ttk.Frame(s); f.pack(fill="x", pady=1)
        ttk.Label(f, text="테이블 파일", width=20).pack(side="left")
        self.vars["sat_table"] = tk.StringVar(value=DEFAULTS["sat_table"])
        ttk.Entry(f, textvariable=self.vars["sat_table"], width=12).pack(side="left")
        ttk.Button(f, text="…", width=3, command=self._pick_table).pack(side="left")
        stat = ("✔ 발견" if os.path.exists(DEFAULTS["sat_table"])
                else ("CoolProp 사용" if HAS_CP else "✘ 없음"))
        self.lbl_tbl = ttk.Label(s, text=f"  상태: {stat}",
                                 foreground="#2a7" if "✔" in stat else "#a70",
                                 font=("", 8))
        self.lbl_tbl.pack(anchor="w")
        row(s, "sat_n", "  (fallback) 생성 pts", "-")

        
        bf = ttk.Frame(fr); bf.pack(fill="x", padx=6, pady=10)
        self.btn_run = ttk.Button(bf, text="▶  계산 실행", command=self.run)
        self.btn_run.pack(fill="x", pady=2)
        self.btn_stop = ttk.Button(bf, text="■  중단", command=self.stop, state="disabled")
        self.btn_stop.pack(fill="x", pady=2)
        ttk.Button(bf, text="기본값 복원", command=self.reset).pack(fill="x", pady=2)
        ttk.Button(bf, text="그래프 저장", command=self.save_fig).pack(fill="x", pady=2)

    def _pick_table(self):
        fn = filedialog.askopenfilename(filetypes=[("NPZ", "*.npz"), ("All", "*.*")])
        if fn:
            self.vars["sat_table"].set(fn)
            self.lbl_tbl.config(text="  상태: ✔ 발견", foreground="#2a7")

    def _toggle_qmode(self):
        if self.vars["q_mode"].get() == "empirical":
            self.f_flux.pack_forget(); self.f_emp.pack(fill="x")
        else:
            self.f_emp.pack_forget(); self.f_flux.pack(fill="x")

    
    def _build_output(self, parent):
        pw = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        pw.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(pw); pw.add(top, weight=0)
        bot = ttk.Frame(pw); pw.add(bot, weight=1)

        
        lf = ttk.LabelFrame(top, text="계산 과정", padding=4)
        lf.pack(fill="both", expand=True)
        self.txt = tk.Text(lf, height=13, wrap="none", font=("Consolas", 9),
                           bg="#1e1e1e", fg="#d4d4d4", insertbackground="#fff")
        ys = ttk.Scrollbar(lf, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=ys.set)
        self.txt.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        for tag, col in [("ok", "#4ec9b0"), ("bad", "#f48771"),
                         ("hdr", "#569cd6"), ("dim", "#808080")]:
            self.txt.tag_config(tag, foreground=col)

        
        pf = ttk.Frame(top); pf.pack(fill="x", pady=3)
        self.lbl_prog = ttk.Label(pf, text="대기 중", width=52)
        self.lbl_prog.pack(side="left", padx=4)
        self.pbar = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.pbar.pack(side="left", fill="x", expand=True, padx=4)

        
        gf = ttk.LabelFrame(bot, text="결과 그래프", padding=2)
        gf.pack(fill="both", expand=True)
        self.fig = plt.Figure(figsize=(11, 8), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, gf).update()
        self._blank_fig()

    def _blank_fig(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(.5, .5, "파라미터를 입력하고 [계산 실행]을 누르세요",
                ha="center", va="center", fontsize=13, color="#888")
        ax.axis("off")
        self.canvas.draw()

    
    def log(self, msg, tag=None):
        self.q.put(("log", msg, tag))

    def prog(self, label, frac, t_sim, P, lvl):
        self.q.put(("prog", label, frac, t_sim, P, lvl))

    def _pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "log":
                    _, msg, tag = item
                    self.txt.insert("end", msg + "\n", tag or ())
                    self.txt.see("end")
                elif item[0] == "prog":
                    _, lb, fr, ts, P, lv = item
                    self.pbar["value"] = fr * 100
                    self.lbl_prog.config(
                        text=f"{lb:16s} {fr*100:5.1f}%   t={ts/60:5.1f} min   "
                             f"P={P:.3f} MPag   L={lv:4.1f}%")
                elif item[0] == "done":
                    self._finish(item[1])
                elif item[0] == "error":
                    self._error(item[1])
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    
    def _collect(self):
        p = {}
        for k, v in self.vars.items():
            s = v.get().strip()
            if k in ("q_mode", "valve_mode", "sat_table"):
                p[k] = s
            elif k in ("rated_lift", "rated_full", "sat_n"):
                p[k] = int(float(s))
            else:
                p[k] = float(s)
        return p

    def run(self):
        if self.running:
            return
        try:
            p = self._collect()
        except ValueError as e:
            messagebox.showerror("입력 오류", f"숫자 형식이 잘못되었습니다.\n{e}")
            return
        self.txt.delete("1.0", "end")
        self._stop = False
        self.running = True
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self._worker, args=(p,), daemon=True).start()

    def stop(self):
        self._stop = True
        self.log("\n[중단 요청]", "bad")

    def _worker(self, p):
        try:
            t0 = time.time()
            eng = FireCaseEngine(p, log=lambda m, t=None: self.log(m, t),
                                 prog=self.prog, stop_flag=lambda: self._stop)
            self.log("")
            self.log(f"화재조건 : {eng.fire_label()}", "hdr")
            A0 = eng.A_wet(p["l0"] * eng.V_TANK)
            self.log(f"초기 Awet({p['l0']*100:.0f}%) = {A0:.3f} m²  →  "
                     f"Q0 = {eng.fire_heat(A0)/1000:.1f} kW")
            self.log(f"압력기준 : 설계 {p['p_design']/1e6:.2f} | "
                     f"축적 {eng.P_ACC/1e6:.2f} | 수압 {p['p_hydro']/1e6:.2f} MPag")
            self.log(f"[액충만] {eng.liquid_full_check()}", "dim")
            self.log("")
            self.log("═" * 62, "hdr")

            valves = [("양정식", p["a_lift"], p["rated_lift"]),
                      ("전량식", p["a_full"], p["rated_full"])]
            res = {}
            for nm, A, rated in valves:
                self.log(f"\n▶ {nm} 계산 중  (A = {A*1e6:.1f} mm²)")
                r = eng.simulate(A, p["valve_mode"], label=nm)
                ok, pmax = eng.report(nm, r, rated)
                res[nm] = dict(r=r, ok=ok, pmax=pmax, rated=rated)
                self.log(f"  {'✔ 적합' if ok else '✘ 부적합'}  Pmax = {pmax:.4f} MPag",
                         "ok" if ok else "bad")

            
            other = "mod" if p["valve_mode"] == "pop" else "pop"
            self.log(f"\n▶ 그래프용 {other} 모드 계산")
            for nm, A, _ in valves:
                res[nm]["r2"] = eng.simulate(A, other, label=f"{nm}·{other}")

            self.log(f"\n총 소요 {time.time()-t0:.1f} s", "dim")
            self.q.put(("done", (eng, p, res)))
        except KeyboardInterrupt:
            self.q.put(("error", "사용자에 의해 중단되었습니다."))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def _error(self, msg):
        self.running = False
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl_prog.config(text="중단/오류")
        self.log("\n" + msg, "bad")

    
    def _finish(self, payload):
        eng, p, res = payload
        self.running = False
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.pbar["value"] = 100
        self.lbl_prog.config(text="완료")

        self.fig.clear()
        ax = self.fig.subplots(3, 2)
        cols = {"양정식": "crimson", "전량식": "navy"}
        prim = p["valve_mode"]

        for nm, col in cols.items():
            d = res[nm]
            r, r2 = d["r"], d["r2"]
            t, t2 = r[:, T_]/60, r2[:, T_]/60
            dts = r[1, T_] - r[0, T_]
            cum = np.cumsum(r[:, W_]/3600.0) * dts

            ax[0,0].plot(t, r[:, P_], col, lw=.6, alpha=.45)
            ax[0,0].plot(t2, r2[:, P_], col, lw=1.4, label=nm)
            ax[0,1].plot(t, r[:, TL_], col, lw=.6, alpha=.45)
            ax[0,1].plot(t2, r2[:, TL_], col, lw=1.4, label=nm)
            ax[1,0].plot(t, r[:, LV_], col, label=nm)
            ax[1,1].plot(t, r[:, W_], col, lw=.7, label=nm)
            ax[2,0].plot(t, cum, col, lw=1.6, label=f"{nm} ({cum[-1]:.0f} kg)")
            ax[2,1].plot(t, r[:, Q_], col, label=nm)

        
        a = ax[0,0]
        a.axhline(p["p_hydro"]/1e6, ls="-", c="purple", lw=1.3, alpha=.8)
        a.axhline(eng.P_ACC/1e6, ls="--", c="red", lw=1.5)
        a.axhline(p["p_design"]/1e6, ls="-.", c="darkorange", lw=1.2)
        xr = a.get_xlim()[1]
        a.text(xr*.02, p["p_hydro"]/1e6+.02, f"수압시험압 {p['p_hydro']/1e6:.2f}",
               fontsize=7.5, color="purple")
        a.text(xr*.02, eng.P_ACC/1e6+.02, f"최대축적압 {eng.P_ACC/1e6:.2f}",
               fontsize=7.5, color="red")
        a.text(xr*.02, p["p_design"]/1e6-.09, f"설계압/설정압 {p['p_design']/1e6:.2f}",
               fontsize=7.5, color="darkorange")
        a.set_ylim(0, max(p["p_hydro"]/1e6*1.15, a.get_ylim()[1]))

        m0 = p["l0"] * eng.V_TANK * eng._ip(eng.T0, "rl")
        ax[2,0].axhline(m0, ls="--", c="k", lw=.8)
        ax[2,0].text(.5, m0*1.01, f"초기 액 인벤토리 {m0:.0f} kg", fontsize=8)

        titles = [f"Pressure [MPag]  (연={prim}, 진={'mod' if prim=='pop' else 'pop'})",
                  "Liquid Temperature [℃]", "Liquid Level [%]",
                  "SV Mass Flow (instant) [kg/h]",
                  "Cumulative Discharge [kg]", "Fire Heat Input [kW]"]
        for axi, ti in zip(ax.flat, titles):
            axi.set_xlabel("Time [min]"); axi.set_ylabel(ti)
            axi.grid(alpha=.3); axi.legend(fontsize=7)

        self.fig.suptitle(f"LPG {eng.V_TANK*1000:.0f}L Fire Case — {eng.comps} | "
                          f"{eng.fire_label()} | dynamic wetted area", y=.995)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_fig(self):
        fn = filedialog.asksaveasfilename(defaultextension=".png",
                                          filetypes=[("PNG", "*.png")])
        if fn:
            self.fig.savefig(fn, dpi=140)
            messagebox.showinfo("저장", f"저장되었습니다.\n{fn}")

    def reset(self):
        for k, v in DEFAULTS.items():
            if k in self.vars:
                self.vars[k].set(str(v))
        self._toggle_qmode()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()
