# LPG Fire Simulator — public source edition

Korean Streamlit and Tkinter interfaces for time-dependent LPG tank pressure,
temperature, liquid inventory and pressure-relief comparison under fire exposure.
Tank and valve defaults are synthetic examples, not manufacturer specifications.

## Install and run

Python 3.12 or later:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_server.py
```

The launcher binds to loopback (127.0.0.1). Choose a synthetic tank or enter your
own dimensions, set the fire and valve conditions, and press Run Simulation.
Use `python firecase_sim_GUI.py` for the desktop interface, or
`python firecase_sim_CLI.py` for a configurable CLI example.

The bundled numeric saturation table avoids a CoolProp runtime requirement.
Install requirements-optional.txt to regenerate it with build_sat_tables.py.
The approximation uses fixed-composition saturation properties, uniform tank
equilibrium, time-dependent inventory and valve opening/reseating hysteresis.
It does not calculate wall-strength failure or certify BLEVE prevention.

## Publication scope

Includes application/calculation code and numeric property tables only.
Manufacturer drawings and presets, internal review reports, invention forms,
historical analysis workbooks, manuscript results, logos and old Git history
are excluded. No historical accuracy claims transfer to these synthetic cases.
Confirm all input values and applicability before engineering use.

## Rights

Source-visible; all rights reserved. LICENSE does not grant an open-source
reuse license. See THIRD_PARTY_NOTICES.md for external components.
