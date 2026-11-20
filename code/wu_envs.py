import json

import numpy as np

ENV1D = ['data/gridsearch/experiment1D/kernel1.json',
         'data/gridsearch/experiment1D/kernel2.json']
ENV2D = ['data/gridsearch/experiment2D/kernel1.json',
         'data/gridsearch/experiment2D/kernel2.json']


def load_envs(paths, ndim, gridside):
    from config import ROOT
    out = {}
    for kidx, p in enumerate(paths):
        d = json.load(open(ROOT / p))
        envs = {}
        for ei, cells in d.items():
            arr = np.zeros(gridside ** ndim)
            for c in cells.values():
                if ndim == 1:
                    arr[int(c['x'])] = c['y']
                else:
                    arr[int(c['x2']) * gridside + int(c['x1'])] = c['y']
            envs[int(ei)] = arr
        out[kidx] = envs
    return out
