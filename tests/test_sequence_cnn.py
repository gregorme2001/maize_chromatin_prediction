import json
import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')

from maize_regulatory.sequence_cnn import (ancestry_projection,standardize_training,
    SequenceCNN,EncodedSequences,reverse_complement,fit_network,predict,encode_inputs)


def test_balanced_factorization_matches_full_pair_loss_and_gradient():
    rng=np.random.default_rng(8)
    y=rng.normal(size=(13,7));p=rng.normal(size=(7,3));train=np.arange(9)
    a=ancestry_projection(y,p,train)
    fitted=torch.tensor(rng.normal(size=13),requires_grad=True)
    yy=torch.tensor(y);effect=torch.tensor(a['effect'])
    full=((yy-fitted[:,None]-effect[None,:])**2).mean()
    region=((yy.mean(1)-fitted)**2).mean()
    constant=((yy-yy.mean(1,keepdim=True)-effect[None,:])**2).mean()
    assert torch.allclose(full,region+constant,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(full,fitted,retain_graph=True)[0],
                          torch.autograd.grad(region,fitted)[0],atol=1e-12)
    changed=y.copy();changed[9:]+=10000
    assert np.array_equal(ancestry_projection(changed,p,train)['effect'],a['effect'])


def test_scaling_and_length_buckets_do_not_use_validation_values_or_batch_companions():
    z=np.array([[0.],[2.],[100.]]);target=np.array([1.,3.,999.])
    _,_,transform=standardize_training(z,target,np.array([0,1]))
    assert transform['covariate_mean'][0]==1 and transform['target_mean']==2
    code=np.full((3,512),4,np.uint8);starts=np.array([181,106,181]);ends=starts+[150,300,150]
    for i in range(3):code[i,starts[i]:ends[i]]=i
    seq=EncodedSequences(code,starts,ends,'cpu')
    batches=list(seq.batches(np.array([2,1,0]),2,True))
    seen=np.concatenate([positions for positions,_ in batches]);assert sorted(seen)==[0,1,2]
    assert torch.equal(seq.batch(np.array([0]),True)[0],seq.batch(np.array([2,0]),True)[1])
    x=seq.batch(np.array([0]),True)
    assert torch.equal(reverse_complement(reverse_complement(x)),x)


def test_joint_training_recovers_sequence_and_linear_signal():
    torch.set_num_threads(1)
    rng=np.random.default_rng(5);n=64
    group=rng.integers(0,2,n);codes=np.repeat(group[:,None],256,axis=1).astype(np.uint8)
    z=rng.normal(size=(n,1));target=2*z[:,0]+3*group
    seq=EncodedSequences(codes,np.zeros(n,int),np.full(n,256),'cpu')
    setting=dict(width=4,depth=1,dropout=0.,learning_rate=.01,l2=0.)
    model,scaled,transform,info=fit_network(seq,z,target,np.arange(48),None,setting,
        seed=14,core_only=True,fixed_epochs=35,batch_size=16)
    fitted=predict(model,seq,scaled,np.arange(48,64),16,True)*transform['target_sd']+transform['target_mean']
    assert np.mean((fitted-target[48:])**2)<.15
    # Covariate coefficients are learnable, and excluded from explicit L2.
    penalty=model.weight_penalty();gradient=torch.autograd.grad(penalty,model.linear.weight,
                                                               allow_unused=True)[0]
    assert gradient is None
    assert info['epochs_run']==35


def test_context_encoding_preserves_cores_and_separates_adjacent_block_inputs(tmp_path):
    text='ACGT'*100
    path=tmp_path/'genome.fa';path.write_text('>chr1\n'+text+'\n')
    (tmp_path/'genome.fa.fai').write_text('chr1\t400\t6\t400\t401\n')
    coords=pd.DataFrame(dict(chrom=[1,1],start=[100,200],end=[120,220],
                             length=[20,20],block=[0,1]))
    data=dict(coordinates=coords,sequences=pd.DataFrame({'sequence':[text[100:120],text[200:220]]}))
    result=encode_inputs(data,path,width=128)
    # Inter-block cut is 160; both windows are padded rather than sharing flanks.
    assert result['audit']['clipped_windows']==2
    for i in range(2):
        a,b=result['context_start'][i],result['context_end'][i]
        assert np.array_equal(result['context'][i,a:b],np.tile(np.arange(4,dtype=np.uint8),5))
