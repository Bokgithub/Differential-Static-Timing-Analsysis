import pickle as pk
import numpy as np
import torch

techdatafile = 'data/libdump_bilin.bin'
#other
inf = 1000000

#load precomputed technology library
def load_lib(filename):
    # Use a breakpoint in the code line below to debug your script.
    f1 = open(filename, 'rb')
    dmp = pk.load(f1)
    f1.close()
    return dmp

#why insert inf array?
def preprocess_arcs(a):
    shape = list(a.shape)
    b = a.reshape(-1, shape[-1] >> 3)
    inf_vector = np.full((b.shape[0], 1), inf)
    b = np.hstack((-inf_vector, b[:, 0:5], inf_vector, -inf_vector, b[:, 5:10], inf_vector, b[:, 10:]))
    shape[-1] = -1
    b = b.reshape(shape)
    return b
libdump = load_lib(techdatafile)
arc_emb = torch.tensor(preprocess_arcs(libdump['arc_emb']), dtype=torch.float32)
