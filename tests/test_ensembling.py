import pytest
import numpy as np
from fusecam.aiutil.ensembling import construct_3dsms_ensembler


def test_ensembler():
    N = 100
    pcount = construct_3dsms_ensembler(N,
                                       1,
                                       1,
                               layers=10,
                               alpha = 0.5,
                               gamma = 0.0,
                               hidden_channels = [3],
                               parameter_bounds = None,
                               parameter_counts_only = True
                               )
    median = np.median( np.array(pcount) )
    assert median > 10000
    assert median < 14000
    assert len(pcount) == 100


