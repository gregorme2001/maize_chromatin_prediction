#!/usr/bin/env python
"""Fit the frequency-ridge comparator using the exact CNN context sequences."""
from pathlib import Path
import argparse
from maize_regulatory.cnn_context_baseline import fit_context_baseline

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-core',action='store_true')
    args=parser.parse_args()
    fit_context_baseline(root/'data/intermediate/sequence_cnn',
                         root/'results/sequence_cnn_benchmark',root,include_core=args.include_core)
