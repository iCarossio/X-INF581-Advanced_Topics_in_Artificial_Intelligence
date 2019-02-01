from utils import *
from map_generator import gen_map
import numpy as np

M = np.ones((6,6), dtype=int) * -1
M[0:2,0:2] = np.array([[1,1],[0,0]])
M[-2:,-2:] = 0
print(M)
M = gen_map(M)
draw_save_map_png("M.png",M)
save_map_dat('M.dat',M)
