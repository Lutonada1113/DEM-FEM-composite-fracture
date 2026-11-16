# hole benchmark

Equal-area opening shapes. Use --case C/S/T for circular/square/triangular openings.

`main.py`: DEM-FEM. `main_fem.py`: conventional FEM reference. Run through the root `run_case.py` launcher; arguments after `--` go to these drivers. New outputs are isolated under root `runs/`. Archived numerical results are in `results/hole/`. Automatic plotting has been removed.
