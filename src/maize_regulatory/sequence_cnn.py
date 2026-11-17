"""Residual sequence CNN and exact balanced region/genotype loss factorization."""

from pathlib import Path
import copy
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def ancestry_projection(y, pcs, train):
    """Profile global ancestry exactly; sequence/covariate optimization stays joint.

    With the same genotypes at every ACR and centered PCs, the full pair MSE is
    mean-ACR MSE plus an ancestry-only term independent of the sequence model.
    This is an exact identity, not fitting a frozen region-covariate baseline.
    """
    y, pcs = np.asarray(y,float), np.asarray(pcs,float)
    if y.ndim != 2 or pcs.shape[0] != y.shape[1] or not len(train):
        raise ValueError('Balanced ACR/genotype axes or training rows invalid')
    center = pcs.mean(axis=0)
    p = pcs-center
    target = (y[train]-y[train].mean(axis=1,keepdims=True)).mean(axis=0)
    coefficient = np.linalg.lstsq(p,target,rcond=1e-12)[0]
    return dict(coefficient=coefficient, pc_center=center, effect=p@coefficient)


def standardize_training(z, target, train):
    z, target = np.asarray(z,float), np.asarray(target,float)
    mean, sd = z[train].mean(axis=0), z[train].std(axis=0)
    sd[sd < 1e-10] = 1
    ym, ys = float(target[train].mean()), float(target[train].std())
    if ys <= 0:
        raise ValueError('Constant training target')
    return ((z-mean)/sd).astype('float32'), ((target-ym)/ys).astype('float32'), dict(
        covariate_mean=mean,covariate_sd=sd,target_mean=ym,target_sd=ys)


class ResidualBlock(nn.Module):
    def __init__(self,width,dilation,dropout):
        super().__init__()
        self.first=nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation)
        self.second=nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        update=self.second(self.dropout(F.gelu(self.first(x))))
        return F.gelu(x+update)


class SequenceCNN(nn.Module):
    def __init__(self,n_covariates,width=64,depth=3,dropout=.1):
        super().__init__()
        self.stem=nn.Conv1d(5,width,15,padding=7,stride=4)
        self.blocks=nn.ModuleList([ResidualBlock(width,2**j,dropout) for j in range(depth)])
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(width*8,128),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(128,1))
        self.linear=nn.Linear(n_covariates,1)
        # Starting at the intercept is stable; all region coefficients remain trainable.
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)

    def forward(self,sequence,covariates):
        x=F.max_pool1d(F.gelu(self.stem(sequence)),4)
        for block in self.blocks:
            x=block(x)
            if x.shape[-1]>=16:
                x=F.avg_pool1d(x,2)
        # Eight positional bins; explicit padding avoids unsupported MPS adaptive sizes.
        length=x.shape[-1];padding=(-length)%8
        values=F.pad(x,(0,padding)).reshape(x.shape[0],x.shape[1],8,-1).sum(-1)
        valid=F.pad(torch.ones_like(x[:,:1]),(0,padding)).reshape(x.shape[0],1,8,-1).sum(-1).clamp_min(1)
        pooled=values/valid
        return (self.head(pooled)+self.linear(covariates)).squeeze(1)

    def weight_penalty(self):
        # Explicit L2, not AdamW's decoupled weight decay. Biases/nuisance are unpenalized.
        return sum(p.square().sum() for name,p in self.named_parameters()
                   if p.ndim>=2 and not name.startswith('linear.'))


def reverse_complement(x):
    return x[:,[3,2,1,0,4],:].flip(-1)


class EncodedSequences:
    def __init__(self,codes,core_start,core_end,device):
        self.device=torch.device(device)
        self.codes=torch.tensor(np.asarray(codes),device=self.device,dtype=torch.uint8)
        self.start=torch.as_tensor(core_start,device=self.device,dtype=torch.int64)
        self.end=torch.as_tensor(core_end,device=self.device,dtype=torch.int64)
        self.lengths=np.asarray(core_end)-np.asarray(core_start)
        self.width=codes.shape[1]
        self.buckets=np.minimum(self.width,np.maximum(256,
            np.ceil((self.lengths+2)/128).astype(int)*128))

    def batches(self,rows,batch_size,core_only,rng=None):
        batches=[]
        categories=self.buckets[rows] if core_only else np.zeros(len(rows),int)
        for category in np.unique(categories):
            positions=np.flatnonzero(categories==category)
            if rng is not None:positions=rng.permutation(positions)
            batches.extend(positions[j:j+batch_size] for j in range(0,len(positions),batch_size))
        if rng is not None:rng.shuffle(batches)
        for positions in batches:yield positions,rows[positions]

    def batch(self,rows,core_only=False):
        left,right=0,self.width
        if core_only:
            if len(np.unique(self.buckets[rows]))!=1:raise ValueError('Mixed core-length buckets')
            width=int(self.buckets[rows[0]])
            left=(self.width-width)//2;right=left+width
        ids=torch.as_tensor(rows,device=self.device,dtype=torch.int64)
        c=self.codes[ids,left:right].long()
        onehot=F.one_hot(c,num_classes=5)[...,:4].transpose(1,2).float()
        position=torch.arange(left,right,device=self.device)
        core=((position[None,:]>=self.start[ids,None]) &
              (position[None,:]<self.end[ids,None])).float()[:,None,:]
        return torch.cat([onehot,core],dim=1)


def synchronize(device):
    if str(device)=='mps':torch.mps.synchronize()
    elif str(device).startswith('cuda'):torch.cuda.synchronize()


@torch.no_grad()
def predict(model,sequences,z,rows,batch_size,core_only,rc=True):
    model.eval();output=np.empty(len(rows),dtype=np.float32)
    for positions,at in sequences.batches(rows,batch_size,core_only):
        x=sequences.batch(at,core_only)
        zz=z[torch.as_tensor(at,device=z.device)]
        context=torch.autocast(sequences.device.type,dtype=torch.float16) if sequences.device.type in ['mps','cuda'] else nullcontext()
        with context:
            value=model(x,zz)
            if rc:value=(value+model(reverse_complement(x),zz))/2
        output[positions]=value.float().cpu().numpy()
    return output


def fit_network(sequences,covariates,target,train,validation,setting,*,seed,
                core_only,max_epochs=16,patience=4,fixed_epochs=None,batch_size=512):
    """Validation is used only for early stopping; fixed-epoch refits never examine it."""
    torch.manual_seed(seed)
    if sequences.device.type=='mps':torch.mps.manual_seed(seed)
    rng=np.random.default_rng(seed)
    z,yy,transform=standardize_training(covariates,target,train)
    z=torch.as_tensor(z,device=sequences.device); yy=torch.as_tensor(yy,device=sequences.device)
    model=SequenceCNN(covariates.shape[1],setting['width'],setting['depth'],setting['dropout']).to(sequences.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=setting['learning_rate'])
    mixed=sequences.device.type in ['mps','cuda']
    scaler=torch.amp.GradScaler(sequences.device.type,enabled=mixed)
    # A train-only OLS initialization is followed by joint gradient updates.
    a=np.column_stack([z[train].cpu().numpy(),np.ones(len(train))])
    coefficient=np.linalg.lstsq(a,yy[train].cpu().numpy(),rcond=1e-10)[0]
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor(coefficient[:-1][None,:],device=sequences.device,dtype=torch.float32))
        model.linear.bias.copy_(torch.tensor(coefficient[-1:],device=sequences.device,dtype=torch.float32))
    best=float('inf');best_epoch=0;checkpoint=None;stale=0;history=[]
    start=time.monotonic();epochs=fixed_epochs or max_epochs
    for epoch in range(1,epochs+1):
        model.train()
        for _,at in sequences.batches(train,batch_size,core_only,rng):
            ids=torch.as_tensor(at,device=sequences.device)
            x=sequences.batch(at,core_only)
            if rng.random()<.5:x=reverse_complement(x)
            optimizer.zero_grad(set_to_none=True)
            context=torch.autocast(sequences.device.type,dtype=torch.float16) if mixed else nullcontext()
            with context:
                fitted=model(x,z[ids])
                loss=F.mse_loss(fitted,yy[ids])+setting['l2']*model.weight_penalty()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CNN loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update()
        if fixed_epochs is not None:
            history.append(dict(epoch=epoch))
            continue
        fitted=predict(model,sequences,z,validation,batch_size,core_only)
        score=float(np.mean((fitted-yy[validation].cpu().numpy())**2))*transform['target_sd']**2
        history.append(dict(epoch=epoch,validation_mean_acr_mse=score))
        if score<best:
            best=score;best_epoch=epoch;stale=0
            checkpoint={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=patience:break
    if fixed_epochs is None:
        if checkpoint is None:raise RuntimeError('No valid inner checkpoint')
        model.load_state_dict(checkpoint)
    synchronize(sequences.device)
    return model,z,transform,dict(best_epoch=best_epoch if fixed_epochs is None else fixed_epochs,
        epochs_run=len(history),history=history,seconds=time.monotonic()-start,
        n_parameters=sum(p.numel() for p in model.parameters()))


class IndexedFasta:
    """Read existing FASTA using its existing index without mutating source files."""
    def __init__(self,path):
        self.handle=Path(path).open('rb');self.entries={}
        for line in Path(str(path)+'.fai').read_text().splitlines():
            fields=line.replace('\\t','\t').split('\t')
            self.entries[fields[0]]=tuple(map(int,fields[1:5]))

    def fetch(self,chrom,start,end):
        length,offset,bases,bytes_per_line=self.entries[chrom]
        if start<0 or end>length or end<=start:raise ValueError('FASTA coordinates out of bounds')
        first=offset+(start//bases)*bytes_per_line+start%bases
        last=offset+((end-1)//bases)*bytes_per_line+(end-1)%bases+1
        self.handle.seek(first)
        text=self.handle.read(last-first).replace(b'\n',b'').replace(b'\r',b'').upper()
        if len(text)!=end-start:raise ValueError('FASTA index/length mismatch')
        return text


def encode_inputs(data,fasta_path,width=2048):
    """Preserve full ACRs, mark cores, and clip flanks at label-free block partitions."""
    coordinates=data['coordinates'];n=len(coordinates)
    lengths=coordinates['length'].to_numpy(int)
    core_width=int(np.ceil((lengths.max()+2)/32))*32
    core=np.full((n,core_width),4,dtype=np.uint8)
    context=np.full((n,width),4,dtype=np.uint8)
    starts=(core_width-lengths)//2;ends=starts+lengths
    context_start=np.empty(n,dtype=int);context_end=np.empty(n,dtype=int)
    lookup=np.full(256,4,dtype=np.uint8)
    lookup[np.frombuffer(b'ACGT',dtype=np.uint8)]=np.arange(4)
    fasta=IndexedFasta(fasta_path)
    lower=np.zeros(n,dtype=np.int64);upper=np.zeros(n,dtype=np.int64)
    for chrom,part in coordinates.groupby('chrom',sort=False):
        ids=np.flatnonzero(coordinates.chrom.eq(chrom));chrom_name=f'chr{chrom}'
        upper[ids]=fasta.entries[chrom_name][0]
        # Cuts in gaps between adjacent occupied genomic blocks preserve all ACRs.
        ordered=ids[np.argsort(coordinates.start.to_numpy()[ids])]
        block=coordinates.block.to_numpy()
        cuts=np.flatnonzero(block[ordered[1:]]!=block[ordered[:-1]])
        for j in cuts:
            left,right=ordered[j],ordered[j+1]
            end,start=int(coordinates.end.iloc[left]),int(coordinates.start.iloc[right])
            if end>start:raise ValueError('ACRs overlap across frozen genomic blocks')
            cut=(end+start)//2
            upper[ids[block[ids]==block[left]]]=cut
            lower[ids[block[ids]==block[right]]]=cut
    clipped=0;ambiguous=0
    for i,sequence in enumerate(data['sequences'].sequence):
        raw=sequence.encode();core[i,starts[i]:ends[i]]=lookup[np.frombuffer(raw,dtype=np.uint8)]
        start,end=int(coordinates.start.iloc[i]),int(coordinates.end.iloc[i])
        desired=(start+end)//2-width//2
        first=max(desired,int(lower[i]));last=min(desired+width,int(upper[i]))
        if first>start or last<end:raise ValueError('Context clipping would truncate an ACR')
        text=fasta.fetch(f'chr{coordinates.chrom.iloc[i]}',first,last)
        if text[start-first:end-first]!=raw:raise ValueError('B73 FASTA and original ACR sequence disagree')
        encoded=lookup[np.frombuffer(text,dtype=np.uint8)]
        context[i,first-desired:last-desired]=encoded
        context_start[i]=start-desired;context_end[i]=end-desired
        clipped+=int(first!=desired or last!=desired+width)
        ambiguous+=int(np.sum(encoded==4))
    fasta.handle.close()
    return dict(core=core,context=context,core_start=starts,core_end=ends,
        context_start=context_start,context_end=context_end,
        audit=dict(context_width=width,core_padded_width=core_width,clipped_windows=clipped,
            ambiguous_context_bases=ambiguous,all_acr_sequences_match_fasta=True,
            flank_rule='Clip at sequence-only inter-ACR gaps separating occupied frozen 5Mb blocks; '
                      'pad with unknown bases; preserve every ACR; no shared flank DNA across blocks'))
