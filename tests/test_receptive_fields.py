from fusecam.manipimg import receptive_fields
import torch
import numpy as np
import pytest

def test_extract_linearize_and_reconstruct():
    inp_tensor = torch.rand( (3,32,32,32) )
    C,X,Y,Z = inp_tensor.shape
    M=3
    result = receptive_fields.extract_and_linearize_neighborhoods_with_channels( inp_tensor, M )
    N, K = result.shape
    assert N == X*Y*Z
    assert K == ((2*M+1)**3)*C
    d = (2*M+1)
    deshifted = receptive_fields.produce_individual_tensors(result, d, d, d, C, X, Y, Z)
    assert deshifted.shape
    a,c,x,y,z = deshifted.shape
    assert a == ((2*M+1)**3)
    assert c == C
    assert x == X
    assert y == Y
    assert z == Z
    s = torch.std(deshifted, dim=0)[:,M:-M-1,M:-M-1,M:M-1,]
    s = torch.sum(s*s)
    assert s < 1e-5





if __name__ =="__main__":
    test_extract_linearize_and_reconstruct()

