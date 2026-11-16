# verification benchmark

Two fiber-orientation layouts. Use --mesh 1/2 and --tensor A1/A2/A3. Read the global reproducibility notes before choosing lm/refinement parameters.

`main.py`: DEM-FEM. `main_fem.py`: conventional FEM reference. Run through the root `run_case.py` launcher; arguments after `--` go to these drivers. New outputs are isolated under root `runs/`. Archived numerical results are in `results/verification/`. Automatic plotting has been removed.
