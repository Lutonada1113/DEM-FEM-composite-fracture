# interface benchmark

Finite-width material band. Use --case weak/strong1/strong2. The launcher selects interface_crack.msh; only weak/strong1 have archived results.

`main.py`: DEM-FEM. `main_fem.py`: conventional FEM reference. Run through the root `run_case.py` launcher; arguments after `--` go to these drivers. New outputs are isolated under root `runs/`. Archived numerical results are in `results/interface/`. Automatic plotting has been removed.
