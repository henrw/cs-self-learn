import numpy as np
dims=[4096]*7
dims=np.arange(7)
for Din, Dout in zip(dims[:-1],dims[1:]):
    print(Din,Dout)