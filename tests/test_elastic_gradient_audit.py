import numpy as np
import torch

from softdynamics.elastic_gradient_audit import precision_audit,relative_error


def curved(point):
    return 61.+.4*point[0]+.7*point[1]+300000.*point[0]**3


def test_small_double_steps_resolve_a_locally_correct_but_curved_derivative():
    point=torch.zeros(2,dtype=torch.float64);gradient=np.array([.4,.7])
    delta=torch.tensor([.001,0.],dtype=torch.float64)
    coarse=(curved(point+delta)-curved(point-delta))/.002
    assert relative_error(gradient,[float(coarse),.7])>.25
    result=precision_audit(curved,point,gradient.astype('f'))
    assert result['passed']
    assert result['finite_difference_convergence']<.01


def test_correct_double_reference_cannot_hide_a_wrong_optimizer_gradient():
    result=precision_audit(curved,torch.zeros(2),np.array([-.4,-.7]))
    assert not result['passed']
    assert 'Optimizer gradient differs from the double-precision derivative' in result['failure_reasons']


def test_converged_finite_differences_reject_an_incorrect_backward_function():
    class WrongBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return x.clone()
        @staticmethod
        def backward(ctx,g):return .1*g
    def function(point):return 61.+(WrongBackward.apply(point)*point.new_tensor([.4,.7])).sum()
    result=precision_audit(function,torch.zeros(2),np.array([.04,.07]))
    assert not result['passed']
    assert 'Small-step finite differences disagree with autograd' in result['failure_reasons']
