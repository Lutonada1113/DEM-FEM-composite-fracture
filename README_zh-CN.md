# DEM-FEM复合材料各向异性相场断裂计算代码与结果

本文件夹保留五类算例的求解代码、输入网格及已有数值结果，不包含论文、PPT、图片、绘图脚本、无关笔记本、重复压缩包和逐加载步断点快照。

## 目录

- `dsolve.py`：公共网络、材料、能量、历史场和相场计算模块。
- `examples/`：五组算例，每组均含DEM-FEM的`main.py`、FEM参考解的`main_fem.py`和网格输入。
- `results/`：81份节点场CSV、81份反力CSV、52份逐步耗时CSV、1份原耗时汇总表、32份各工况最新可用断点，共247份数值文件，约1.87 GiB。
- `run_case.py`：在新的`runs/`子文件夹中运行算例，避免覆盖归档结果。
- `load_results.py`：读取节点结果，正确处理续算产生的重复记录。
- `docs/`：数据格式、复现差异及整理说明。

安装依赖后运行环境自检：

```bash
python -m pip install -r requirements.txt
python check_env.py
```

CUDA版PyTorch按[官方安装页面](https://pytorch.org/get-started/locally/)选择。神经位移子问题可以使用GPU；本包中的相场稀疏求解和Python FEM参考程序使用NumPy/SciPy在CPU上执行。本轮验证使用CPU环境，不将该环境的耗时与历史归档计时等同。

## 运行示例

```bash
# 两个epoch、一个加载步的启动检查，仅用于确认程序可运行，不代表收敛结果。
python run_case.py circlehole1 --solver dem-fem --seed 7 --threads 1 -- --angle 30 --mesh 2 --no-effective-gc --steps 1 --epochs 2 --no-stag

# FEM单步启动检查。
python run_case.py circlehole1 --solver fem --threads 1 -- --angle 30 --mesh 2 --steps 1

# 开孔算例：使用原始Gc，指定30度、0.15 mm网格。
python run_case.py circlehole1 --solver dem-fem -- --angle 30 --mesh 0.15 --no-effective-gc --device cuda

# 界面/弱带算例：启动器使用与现有弱带结果对应的interface_crack.msh。
python run_case.py interface --solver dem-fem -- --case weak --device cpu

# 查看原求解器参数。
python run_case.py SENT1 --solver dem-fem -- --help
```

`--`之后是原求解器的命令行选项。`--steps`仅限制本次最多运行多少步，不改变完整加载方案的步长。若需要改完整加载方案中的常量，可显式使用`--set N_STEPS=数值`、`--set U_MAX=数值`。长度尺度、各向异性参数等也可用`--set L=0.2`或`--set BETA=10`设置。

每个新算例均保存独立的代码、输入和`run_config.json`。已有运行文件夹不会被覆盖。续算时直接运行该文件夹内复制的`main.py`，并沿用相同参数。根目录归档结果不会被自动当作新算例断点。

## 读取已有结果

```bash
python load_results.py results/interface/weak/results_weak.csv
```

部分原始CSV因续算含重复的`(step,node)`记录。数据文件保持原样，读取工具与原结果读取程序一致，保留同一键最后一次写入的记录。实际节点数、保存步、重复数量、损伤值范围和对应网格见`results/field_catalog.json`；文件来源和哈希见`results/file_manifest.csv`。

需要注意的具体复现差异已列在`docs/reproducibility.md`：包括旧SENT30神经结果的8,627节点网格缺失、部分历史运行参数没有完整保存，以及当前脚本默认值与某些归档工况不同。整理时未重新计算或改写已有数值结果。

本文件夹尚未指定开源许可证，由作者确定后再加入。
