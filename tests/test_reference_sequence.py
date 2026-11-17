import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maize_regulatory.pooled_sequence import moment_mse, ridge_path
from maize_regulatory.reference_sequence import reference_moments, subset_moments


def test_factorized_reference_moments_equal_every_explicit_pair():
    rng = np.random.default_rng(167)
    nr, ng = 17, 8
    b = np.column_stack([np.ones(nr), rng.normal(size=(nr, 3))])
    p, x, y = rng.normal(size=(ng, 2)), rng.normal(size=(nr, 5)), rng.normal(size=(nr, ng))
    m = reference_moments(b, p, x, y)
    design = np.column_stack([np.repeat(b, ng, axis=0), np.tile(p, (nr, 1)), np.repeat(x, ng, axis=0)])
    assert m.n == nr*ng
    assert np.allclose(m.xx, design.T@design)
    assert np.allclose(m.xy, design.T@y.ravel())
    assert np.isclose(m.yy, y.ravel()@y.ravel())
    ix = np.array([0, 2, 4, 5])
    smaller = subset_moments(m, ix, 6)
    columns = np.r_[ix, np.arange(6, 11)]
    assert np.allclose(smaller.xx, design[:, columns].T@design[:, columns])


def test_motif_penalty_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(568)
    b = np.column_stack([np.ones(20), rng.normal(size=20)])
    p, x, y = rng.normal(size=(8, 2)), rng.normal(size=(20, 6)), rng.normal(size=(20, 8))
    moments = reference_moments(b, p, x, y)
    path = ridge_path(moments, 4, [.1], penalty_multiplier=1.)
    a = np.column_stack([np.repeat(b, 8, axis=0), np.tile(p, (20, 1))])
    z = (np.repeat(x, 8, axis=0)-path['mean'][3:])/path['scale'][3:]
    design = np.column_stack([a, z])
    penalty = np.column_stack([np.zeros((6, 4)), np.sqrt(160*.1)*np.eye(6)])
    expected = np.linalg.lstsq(np.vstack([design, penalty]), np.r_[y.ravel(), np.zeros(6)], rcond=None)[0]
    raw_design = np.column_stack([a, np.repeat(x, 8, axis=0)])
    actual = raw_design@path['coefficients'][.1]
    assert np.allclose(actual, design@expected, atol=1e-9)
    assert np.isclose(moment_mse(moments, path['coefficients'][.1]), np.mean((y.ravel()-actual)**2))


def test_sparse_cortex_bridge_excludes_unselected_cells(tmp_path):
    rscript = shutil.which('Rscript')
    if not rscript:
        pytest.skip('Rscript unavailable')
    matrix = tmp_path/'cells.rds'
    create = '''args<-commandArgs(TRUE);library(Matrix)
x<-as(Matrix(matrix(c(1,0,1,1,5,0,0,1,1,0,0,7),nrow=4),sparse=TRUE),"dgCMatrix")
rownames(x)<-paste0("c",1:4);colnames(x)<-paste0("acr",1:3);saveRDS(x,args[1])'''
    subprocess.run([rscript, '-e', create, str(matrix)], check=True, capture_output=True)
    pd.DataFrame(dict(cellID=['c1','c2','c3','c4'], celltype_full=['cortex']*3+['other'])).to_csv(tmp_path/'meta.csv',index=False)
    pd.DataFrame(dict(matrix_row=[1,2,3],genotype_index=[0,1,0])).to_csv(tmp_path/'selected.csv',index=False)
    pd.DataFrame(dict(matrix_column=[1,2,3],ACR=['acr1','acr2','acr3'])).to_csv(tmp_path/'acrs.csv',index=False)
    binary=tmp_path/'counts.int32'
    script=Path(__file__).resolve().parents[1]/'scripts/pool_reference_cortex.R'
    subprocess.run([rscript,str(script),str(matrix),str(tmp_path/'meta.csv'),str(tmp_path/'selected.csv'),
        str(tmp_path/'acrs.csv'),str(binary),'2'],check=True,capture_output=True)
    expected=np.array([[2,0],[1,0],[1,0]])
    assert np.array_equal(np.fromfile(binary,dtype='<i4').reshape(3,2),expected)
