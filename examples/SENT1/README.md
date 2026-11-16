# SENT1 benchmark

Off-axis plate. Use --angle 30/45/60/90. The archived 8,627-node DEM30 field has no matching supplied connectivity; a newer 9,890-node result is also retained.

`main.py`: DEM-FEM. `main_fem.py`: conventional FEM reference. Run through the root `run_case.py` launcher; arguments after `--` go to these drivers. New outputs are isolated under root `runs/`. Archived numerical results are in `results/SENT1/`. Automatic plotting has been removed.
