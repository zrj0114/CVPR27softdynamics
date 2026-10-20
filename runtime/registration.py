"""Initial-frame registration; proper rigid poses and positive smooth maps."""
import itertools
import numpy as np
from scipy.spatial import cKDTree


def subsample(x,n=4000):
    return x[np.linspace(0,len(x)-1,min(n,len(x))).astype(int)]


def rigid_fit(a,b,weights=None):
    if weights is None:weights=np.ones(len(a))
    weights=weights/weights.sum()
    ca=(weights[:,None]*a).sum(0);cb=(weights[:,None]*b).sum(0)
    u,s,vt=np.linalg.svd((a-ca).T@(weights[:,None]*(b-cb)))
    fix=np.eye(3);fix[2,2]=np.linalg.det(vt.T@u.T)
    r=vt.T@fix@u.T
    return r,cb-r@ca


def rigid_icp(source,target,starts=None,iterations=50):
    source=subsample(source);target=subsample(target)
    cs=source.mean(0);ct=target.mean(0)
    if starts is None:
        _,vs=np.linalg.eigh(np.cov(source.T));_,vt=np.linalg.eigh(np.cov(target.T))
        vs=vs[:,::-1];vt=vt[:,::-1]
        starts=[]
        # Principal axes plus all proper signed permutations avoid yaw/axis
        # ambiguity. A final metric compares every initial-frame point only.
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([-1,1],repeat=3):
                q=np.eye(3)[:,perm]*np.asarray(signs)
                r=vt@q@vs.T
                if np.linalg.det(r)>.999:starts.append((r,ct-r@cs))
        starts.append((np.eye(3),ct-cs))
    tree=cKDTree(target);results=[]
    for r,t in starts:
        previous=np.inf
        for it in range(iterations):
            transformed=source@r.T+t
            distance,ids=tree.query(transformed)
            cutoff=np.quantile(distance,.90)
            keep=distance<=cutoff
            correction,translation=rigid_fit(transformed[keep],target[ids[keep]])
            r=correction@r;t=correction@t+translation
            error=float(distance[keep].mean())
            if abs(previous-error)<1.e-8:break
            previous=error
        transformed=source@r.T+t
        error=.5*(tree.query(transformed)[0].mean()+cKDTree(transformed).query(target)[0].mean())
        results.append((float(error),r,t))
    results.sort(key=lambda z:z[0])
    return results[0],results


def farthest_points(x,n):
    n=min(n,len(x));ids=np.zeros(n,dtype=int);d=np.full(len(x),np.inf)
    k=np.argmax(np.sum((x-x.mean(0))**2,axis=1))
    for i in range(n):
        ids[i]=k;d=np.minimum(d,np.sum((x-x[k])**2,axis=1));k=np.argmax(d)
    return x[ids]


def basis(x,centers,width):
    return np.exp(-.5*np.sum((x[:,None,:]-centers[None,:,:])**2,axis=2)/width**2)


def map_points(x,r,t,centers,coefficients,width,gradient=False):
    """world x = R @ (X + smooth displacement(X)) + t; F = R @ (I+du/dX)."""
    result=[];gradients=[]
    for start in range(0,len(x),4096):
        part=x[start:start+4096]
        diff=part[:,None,:]-centers[None,:,:]
        phi=np.exp(-.5*np.sum(diff*diff,axis=2)/width**2)
        q=part+phi@coefficients
        result.append(q@r.T+t)
        if gradient:
            derivative=-phi[:,:,None]*diff/width**2
            jac=np.eye(3)[None]+np.einsum('ck,ncj->nkj',coefficients,derivative)
            gradients.append(np.einsum('ij,njk->nik',r,jac))
    if gradient:return np.concatenate(result),np.concatenate(gradients)
    return np.concatenate(result)


def fit_deformation(surface,target,centers,r,t,width=.035,iterations=12):
    """Low-dimensional regularized initial shape fit, with no future frames."""
    surface=subsample(surface,3000);target=subsample(target,4000)
    local=(target-t)@r;tree=cKDTree(local);phi=basis(surface,centers,width)
    coefficients=np.zeros((len(centers),3));history=[]
    regularization=.15
    for i in range(iterations):
        q=surface+phi@coefficients
        dist,ids=tree.query(q)
        # Symmetric correspondences restrain one-sided surface collapse.
        reverse=cKDTree(q).query(local)[1]
        a=np.concatenate([phi,phi[reverse]])
        b=np.concatenate([local[ids]-surface,local-surface[reverse]])
        d=np.r_[dist,np.linalg.norm(q[reverse]-local,axis=1)]
        weights=np.minimum(1.,.003/np.maximum(d,1.e-9))
        lhs=a.T@(a*weights[:,None])+regularization*np.eye(len(centers))
        rhs=a.T@(b*weights[:,None])
        candidate=np.linalg.solve(lhs,rhs)
        coefficients=.5*coefficients+.5*candidate
        history.append(float(d.mean()*1000))
    return coefficients,history


def positive_map(surface,particles,target,centers,r,t,coefficients,width):
    """Choose initial-only fit under strain and invertibility bounds."""
    tree=cKDTree(target);best=None;trials=[]
    for alpha in [1.,.75,.5,.25,0.]:
        c=coefficients*alpha
        p,F=map_points(particles,r,t,centers,c,width,gradient=True)
        sv=np.linalg.svd(F,compute_uv=False);j=np.linalg.det(F)
        mapped=map_points(subsample(surface),r,t,centers,c,width)
        error=.5*(tree.query(mapped)[0].mean()+cKDTree(mapped).query(target)[0].mean())*1000
        valid=bool(j.min()>.5 and sv.min()>.70 and sv.max()<1.35)
        # Initial registration must not clip material points at the ground.
        # A single rigid lift corrects tiny voxel-support penetration only.
        lift=max(0.,-.0115-float(p[:,1].min()))
        valid=valid and lift<=.005
        trials.append(dict(alpha=alpha,error_mm=float(error),minJ=float(j.min()),min_stretch=float(sv.min()),max_stretch=float(sv.max()),ground_lift_m=lift,valid=valid))
        if valid and (best is None or error<best[0]):best=(error,c,p,F,lift)
    if best is None:raise ValueError('No positive initial mapping under strain/ground bounds')
    return best,trials
