"""Run a numerical benchmark in a new, isolated output directory.

Example:
    python run_case.py circlehole1 --solver dem-fem -- --angle 30 --no-effective-gc
All arguments after '--' are passed to the original numerical driver.
"""
from pathlib import Path
import argparse,datetime,importlib.util,json,shutil,subprocess,sys,uuid

ROOT=Path(__file__).resolve().parent
CASES=('verification','SENT1','circlehole1','hole','interface')

def inside(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError(f'Path must stay inside {root.name}: {relative}')
    return path

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    split=argv.index('--') if '--' in argv else len(argv)
    forwarded=argv[split+1:] if split<len(argv) else []
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('benchmark',choices=CASES)
    p.add_argument('--solver',choices=['dem-fem','fem'],default='dem-fem')
    p.add_argument('--name',help='new directory name under runs/; existing directories are never overwritten')
    p.add_argument('--set',action='append',default=[],metavar='KEY=JSON',help='override a numerical module constant, e.g. L=0.2 or BETA=10')
    p.add_argument('--replace-mesh',action='append',default=[],metavar='TARGET=SOURCE',help='copy a different supplied .msh inside the run, e.g. SENT45/SENT45.msh=SENT45beta.msh')
    p.add_argument('--seed',type=int,default=None,help='optional NumPy/PyTorch seed for a new run')
    p.add_argument('--threads',type=int,default=None,help='optional PyTorch CPU thread count')
    p.add_argument('--dry-run',action='store_true',help='print the plan without importing numerical libraries or writing files')
    args=p.parse_args(argv[:split])
    if args.threads is not None and args.threads<1:p.error('--threads must be positive')
    overrides={}
    for item in args.set:
        try:key,value=item.split('=',1);value=json.loads(value)
        except (ValueError,json.JSONDecodeError):p.error('--set requires KEY=JSON')
        if not key.isidentifier() or not key.isupper() or key in {'BASE','MSH','MSH_FILE','MESH_FILES'}:p.error(f'Unsupported numerical constant: {key}')
        overrides[key]=value
    name=args.name or f'{args.benchmark}-{args.solver}-{uuid.uuid4().hex[:8]}'
    if Path(name).name!=name or name in {'.','..'} or '\\' in name or '/' in name:p.error('--name must be a single directory name')
    run=ROOT/'runs'/name
    if run.exists():p.error(f'Run directory already exists: {run}. Resume its copied driver directly, or choose another name.')
    source=ROOT/'examples'/args.benchmark
    driver='main.py' if args.solver=='dem-fem' else 'main_fem.py'
    plan={'benchmark':args.benchmark,'solver':args.solver,'driver':driver,'arguments':forwarded,
          'overrides':overrides,'mesh_replacements':args.replace_mesh,'seed':args.seed,
          'threads':args.threads,'output_directory':str(run.relative_to(ROOT)),
          'interface_mesh':'interface_crack.msh' if args.benchmark=='interface' else None}
    if args.dry_run:print(json.dumps(plan,indent=2));return
    if '--help' in forwarded or '-h' in forwarded:
        raise SystemExit(subprocess.run([sys.executable,str(source/driver)]+forwarded,check=False).returncode)
    # Validate all replacement paths before creating the workspace.
    replacements=[]
    for item in args.replace_mesh:
        try:target,origin=item.split('=',1)
        except ValueError:p.error('--replace-mesh requires TARGET=SOURCE')
        src=inside(source,origin);dst=inside(source,target)
        if src.suffix!='.msh' or dst.suffix!='.msh' or not src.is_file():p.error(f'Invalid mesh replacement: {item}')
        replacements.append((target,src))
    run.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ROOT/'dsolve.py',run/'dsolve.py')
    work=run/'examples'/args.benchmark
    for file in source.rglob('*'):
        if file.is_file() and file.suffix in {'.py','.msh','.inp'}:
            dest=work/file.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    for target,src in replacements:shutil.copy2(src,inside(work,target))
    sys.path.insert(0,str(run))
    import numpy as np
    import torch
    import scipy
    if args.seed is not None:
        np.random.seed(args.seed);torch.manual_seed(args.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    if args.threads is not None:torch.set_num_threads(args.threads)
    spec=importlib.util.spec_from_file_location('_demfem_benchmark',work/driver)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for key,value in overrides.items():
        if not hasattr(module,key):raise ValueError(f'Unknown constant for this driver: {key}')
        setattr(module,key,value)
    if args.benchmark=='interface':module.MSH_FILE=str(work/'interface_crack.msh')
    plan['environment']={'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
                         'torch':torch.__version__,'cuda_available':torch.cuda.is_available()}
    plan['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan['numerical_constants']={k:v for k,v in vars(module).items() if k.isupper() and isinstance(v,(str,int,float,bool,list,tuple,dict))}
    plan['status']='running'
    record=run/'run_config.json'
    record.write_text(json.dumps(plan,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    print('Run directory:',run,flush=True)
    old_argv=sys.argv;sys.argv=[str(work/driver)]+forwarded
    try:
        module.main();plan['status']='completed'
    except BaseException as exc:
        plan['status']='failed';plan['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        sys.argv=old_argv
        record.write_text(json.dumps(plan,ensure_ascii=False,indent=2,default=str),encoding='utf8')

if __name__=='__main__':main()

