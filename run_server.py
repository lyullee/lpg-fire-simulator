#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCharm 실행용 런처
이 파일을 [Run] 하면 Streamlit 서버가 뜹니다.
"""

import sys, socket, subprocess, os, time, threading, webbrowser
from pathlib import Path

# ───── 설정 ─────
PORT      = 8085
HOST      = "127.0.0.1"     # LAN 개방하려면 "0.0.0.0"
AUTO_OPEN = True
APP       = "app.py"
TABLE     = "sat_tables.npz"
# ────────────────

BASE = Path(__file__).resolve().parent


def find_free_port(start, tries=20):
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError(f"{start}~{start+tries-1} 빈 포트 없음")


def check_env():
    ok = True
    if not (BASE / APP).exists():
        print(f"  X {APP} 없음 ({BASE})"); ok = False

    tbl = BASE / TABLE
    if not tbl.exists():
        print(f"  X {TABLE} 없음")
        print("    -> 먼저 실행: python build_sat_tables.py")
        ok = False
    else:
        print(f"  O {TABLE}  ({tbl.stat().st_size/1e6:.2f} MB)")

    for mod in ("streamlit", "numpy", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  O {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  X {mod} 미설치 -> pip install {mod}"); ok = False
    return ok


def main():
    print("=" * 66)
    print(" LPG Fire Case - Streamlit 서버 (PyCharm 런처)")
    print("=" * 66)
    print(f" 작업 폴더 : {BASE}")
    print("-" * 66)

    if not check_env():
        print("-" * 66)
        print(" 위 문제를 해결한 뒤 다시 실행하세요.")
        sys.exit(1)

    port = find_free_port(PORT)
    if port != PORT:
        print(f"  ! 포트 {PORT} 사용중 -> {port} 로 변경")

    print("-" * 66)
    print(f" 서버 주소 : http://localhost:{port}")
    if HOST == "0.0.0.0":
        try:
            ip = socket.gethostbyname(socket.gethostname())
            print(f" LAN 접속  : http://{ip}:{port}  (방화벽 인바운드 허용)")
        except Exception:
            pass
    print(" 종료      : PyCharm [Stop] 또는 Ctrl+C")
    print("=" * 66)

    # 최초 실행 시 이메일 프롬프트 방지
    cred = Path.home() / ".streamlit" / "credentials.toml"
    if not cred.exists():
        cred.parent.mkdir(parents=True, exist_ok=True)
        cred.write_text('[general]\nemail = ""\n', encoding="utf-8")
        print("  ! streamlit 최초 실행 → credentials.toml 자동 생성")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(BASE / APP),
        "--server.port", str(port),
        "--server.address", HOST,
        "--server.headless", "true",  # 프롬프트 방지 (항상 true)
        "--browser.gatherUsageStats", "false",
    ]

    # 브라우저는 런처가 직접 연다
    if AUTO_OPEN:
        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open, daemon=True).start()

    try:
        subprocess.run(cmd, cwd=str(BASE))
    except KeyboardInterrupt:
        print("\n서버를 종료했습니다.")


if __name__ == "__main__":
    main()
