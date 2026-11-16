# circlehole1 benchmark

Open-hole tension. Use --angle 0/15/30/45/60 and --mesh to select an input. Specify the Gc-correction options explicitly when comparing solvers.

`main.py`: DEM-FEM. `main_fem.py`: conventional FEM reference. Run through the root `run_case.py` launcher; arguments after `--` go to these drivers. New outputs are isolated under root `runs/`. Archived numerical results are in `results/circlehole1/`. Automatic plotting has been removed.
