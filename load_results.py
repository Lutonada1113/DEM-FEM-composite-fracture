"""Read a nodal CSV without visualization dependencies.

Repeated (step,node) entries are resolved by keeping the last saved row, as in
the original result readers. Original archived CSV files are never rewritten.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np

def load_step(path,step=None):
    """Return step, node IDs, coordinates, displacement and damage arrays.

    If step is None, select the largest saved step. Node rows are sorted by
    their original zero-based node IDs; no connectivity is inferred.
    """
    target=step;nodes={}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        required={'step','node','x','y','u_x','u_y','d'}
        if not required.issubset(rd.fieldnames or []):raise ValueError('Expected a nodal results CSV')
        for row in rd:
            n=int(row['step'])
            if step is None and (target is None or n>target):target=n;nodes.clear()
            if n==target:nodes[int(row['node'])]=[float(row[k]) for k in ['x','y','u_x','u_y','d']]
    if not nodes:raise ValueError(f'No nodal results at step {target}: {path}')
    ids=np.array(sorted(nodes),dtype=np.int64);values=np.array([nodes[int(n)] for n in ids],dtype=np.float64)
    return {'step':target,'node_ids':ids,'coordinates':values[:,:2],
            'displacement':values[:,2:4],'damage':values[:,4]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_file');p.add_argument('--step',type=int,default=None)
    args=p.parse_args();data=load_step(args.csv_file,args.step)
    print(json.dumps({'step':data['step'],'nodes':len(data['node_ids']),
                      'damage_min':float(data['damage'].min()),'damage_max':float(data['damage'].max()),
                      'displacement_shape':list(data['displacement'].shape)},indent=2))
if __name__=='__main__':main()

