"""Check numerical dependencies, an included mesh and automatic differentiation."""
from pathlib import Path
import json,sys
import numpy as np
import scipy
import torch
import dsolve

def main():
    root=Path(__file__).resolve().parent
    mesh=root/'examples/circlehole1/circlehole-m2.msh'
    nodes,tri=dsolve.read_gmsh_mesh(mesh)
    tensor=torch.tensor([1.0,2.0],dtype=torch.float64,requires_grad=True)
    tensor.square().sum().backward()
    assert torch.allclose(tensor.grad,torch.tensor([2.0,4.0],dtype=torch.float64))
    print(json.dumps({'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
                     'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
                     'test_mesh_nodes':len(nodes),'test_mesh_elements':len(tri),
                     'autograd':'passed'},indent=2))
if __name__=='__main__':main()
