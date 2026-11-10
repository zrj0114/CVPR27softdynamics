"""Fixed 384-cell domain retry; unchanged dx, inputs, time integration and contact."""
import argparse
from pathlib import Path
from .adapters import pad_empm
from .io import sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--task',required=True)
    args=parser.parse_args()
    original_domain=pad_empm.simulation_domain
    def expanded(*a,**kw):
        size,shift=original_domain(*a,**kw)
        dx=a[3] if len(a)>3 else kw['dx']
        return 384,shift+(384-size)//2*dx
    pad_empm.simulation_domain=expanded
    base=pad_empm.PadRuntime
    class LargeGridRuntime(base):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.source_hashes['domain_adapter']=sha256(Path(__file__))
    pad_empm.PadRuntime=LargeGridRuntime
    from .elastic_data import collect
    if not collect(args.root,args.task,'cuda:0')['valid']:raise SystemExit(2)

if __name__=='__main__':main()
