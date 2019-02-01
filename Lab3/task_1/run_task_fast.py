#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import itertools
import operator
from random import shuffle, choice
from copy import deepcopy
import sys
import time

import pygame
import numpy as np
import time

'''
    Functions to draw a map
'''

def code2num(code):
    ''' convert 2*2 terrain grid to tile index '''
    if np.sum(code < 0) > 0:
        return -1
    b = 1 * int(code[0,0]) + 2 * int(code[0,1]) + 4 * int(code[1,0]) + 8 * int(code[1,1])
    return 15 - b

def load_map(fname):
    return np.genfromtxt(fname, delimiter = 1, dtype=int)

def save_map_dat(fname, M): 
    print(M)
    np.savetxt(fname, M, fmt='%d', delimiter='')

def draw_save_map_png(fname, tile_codes, tile_size=16):
    '''
        Build and draw the map defined by tile_codes. 
        Draw it with a tile size of tile_size.
        Save it as a png to filename fname.

        Note: A tile image covers an array of 4 characters/numbers.
    '''
    
    N_ROWS = int(tile_codes.shape[0] / 2)
    N_COLS = int(tile_codes.shape[1] / 2)
    WIDTH = N_COLS * tile_size
    HEIGHT = N_ROWS * tile_size
    size = [WIDTH, HEIGHT]

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Init.
    background = pygame.Surface(size)

    # Load.
    sheet = pygame.image.load('RPGTiles.png').convert_alpha()

    # Draw
    for j in range(0,N_COLS*2,2):
        for k in range(0,N_ROWS*2,2):
            c = tile_codes[k:k+2,j:j+2]
            n = code2num(c)
            image = None
            if n < 0:
                image = pygame.Surface((tile_size,tile_size))
                image.fill((0,0,0))
            else:
                image = sheet.subsurface((n*tile_size,0,tile_size,tile_size))
            background.blit(image, (j/2*tile_size, k/2*tile_size))

    screen.blit(background, [0, 0])
    pygame.display.flip()
    #pygame.image.save(background, fname)


#M = load_map('tileset.dat')
#draw_save_map_png('tileset.png', M)
#M = load_map('test.dat')
#draw_save_map_png('test.png', M)
#M = load_map('invalid.dat')
#draw_save_map_png('invalid.png', M)
#save_map_dat('test2.dat',M)
#M = load_map('test2.dat')

#M = np.ones((10,10), dtype=int) * -1
#M[0:2,0:2] = 1
#M[-2:,-2:] = 0
#draw_save_map_png('test2.png',M)



np.set_printoptions(threshold=np.inf)


# In[3]:


TILES = [[[1,1],[1,1]], [[0,0],[0,0]], [[0,1],[1,1]], [[1,0],[1,1]], [[0,0],[1,1]], 
         [[1,1],[0,1]], [[0,1],[0,1]], [[0,0],[0,1]], [[1,1],[1,0]], [[1,0],[1,0]], 
         [[0,0],[1,0]], [[1,1],[0,0]], [[0,1],[0,0]], [[1,0],[0,0]]]

TILES = np.array([np.array(tile) for tile in TILES])
idx = list(range(len(TILES)))
#shuffle(idx)


# In[4]:


def intersection(a,b):
    """ Return intersection between lists A & B """
    return list(set(a) & set(b))

def union(a,b):
    """ Return union between lists A & B """
    return list(set(a+b))

def even(num):
    """ Check if a number is even """
    return num % 2 == 0


# In[5]:


def check_up(M, i, j):
    return M[i-1,j] == M[i-1,j+1] == -1 or M[i-1,j] == M[i,j] and M[i-1,j+1] == M[i,j+1]

def check_down(M, i, j):
    return M[i+2,j] == M[i+2,j+1] == -1 or M[i+2,j] == M[i+1,j] and M[i+2,j+1] == M[i+1,j+1]

def check_left(M, i, j):
    return M[i,j-1] == M[i+1,j-1] == -1 or M[i,j-1] == M[i,j] and M[i+1,j-1] == M[i+1,j]

def check_right(M, i, j):
    return M[i,j+2] == M[i+1,j+2] == -1 or M[i,j+2] == M[i,j+1] and M[i+1,j+2] == M[i+1,j+1]

def check_up_left(M, i, j):
    return M[i-1,j-1] == -1 or M[i-1,j-1] == M[i,j]

def check_up_right(M, i, j):
    return M[i-1,j+2] == -1 or M[i-1,j+2] == M[i,j+1]

def check_down_left(M, i, j):
    return M[i+2,j-1] == -1 or M[i+2,j-1] == M[i+1,j]

def check_down_right(M, i, j):
    return M[i+2,j+2] == -1 or M[i+2,j+2] == M[i+1,j+1]


def check_neighbor_up(M, i, j):
    if M[i-1,j] == M[i-1,j+1] == -1:
        for tile in TILES:
            M_new = deepcopy(M)
            M_new[i-2:i,j:j+2] = tile
            if is_valid_tile(M_new,i-2,j, lite=True):
                return True
        return False
    return True

def check_neighbor_down(M, i, j):
    if M[i+2,j] == M[i+2,j+1] == -1:
        for tile in TILES:
            M_new = deepcopy(M)
            M_new[i+2:i+4,j:j+2] = tile
            if is_valid_tile(M_new,i+2,j, lite=True):
                return True
        return False
    return True

def check_neighbor_left(M, i, j):
    if M[i,j-1] == M[i+1,j-1] == -1:
        for tile in TILES:
            M_new = deepcopy(M)
            M_new[i:i+2,j-2:j] = tile
            if is_valid_tile(M_new,i,j-2, lite=True):
                return True
        return False
    return True

def check_neighbor_right(M, i, j):
    if M[i,j+2] == M[i+1,j+2] == -1:
        for tile in TILES:
            M_new = deepcopy(M)
            M_new[i:i+2,j+2:j+4] = tile
            if is_valid_tile(M_new,i,j+2, lite=True):
                return True
        return False
    return True


# In[6]:


def is_valid_tile(M, i, j, lite=False):
    """ 
        Check if tile (i,j) is valid in V(k) := M, using auxiliary functions above
        Returns in fact P(V(k+1)|M,V(1),…,V(k)))>0 (cf. README) 
    """

    valid_up, valid_down, valid_left, valid_right, valid_up_left, valid_up_right, valid_down_left, valid_down_right, valid_neighbor_up, valid_neighbor_down, valid_neighbor_left, valid_neighbor_right = True, True, True, True, True, True, True, True, True, True, True, True

    if i != 0: # Not top
        valid_up = check_up(M, i, j)
        if not lite:
            valid_neighbor_up = check_neighbor_up(M, i, j)
    if i+1 != M.shape[0]-1: # Not down
        valid_down = check_down(M, i, j)
        if not lite:
            valid_neighbor_down = check_neighbor_down(M, i, j)
    if j != 0: # Not left
        valid_left = check_left(M, i, j)
        if not lite:
            valid_neighbor_left = check_neighbor_left(M, i, j)
    if j+1 != M.shape[1]-1: # Not right
        valid_right = check_right(M, i, j)
        if not lite:
            valid_neighbor_right = check_neighbor_right(M, i, j)

    if i != 0 and j != 0: # Not top left
        valid_up_left = check_up_left(M, i, j)
    if i != 0 and j+1 != M.shape[1]-1: # Not top right
        valid_up_right = check_up_right(M, i, j)
    if i+1 != M.shape[0]-1 and  j != 0: # Not down left
        valid_down_left = check_down_left(M, i, j)
    if i+1 != M.shape[0]-1 and j+1 != M.shape[1]-1: # Not down right
        valid_down_right = check_down_right(M, i, j)
        
    return valid_up and valid_down and valid_left and valid_right and valid_up_left and valid_up_right and valid_down_left and valid_down_right and valid_neighbor_up and valid_neighbor_down and valid_neighbor_left and valid_neighbor_right


# In[7]:


def is_valid(M):
    """ Check if the vertex M is valid """

    if not (even(M.shape[0]) and even(M.shape[1])):
        return False

    non_empty_tiles = [(i,j) for i,j in np.argwhere(M != -1) if even(i) and even(j)]

    for i,j in non_empty_tiles:
        if not is_valid_tile(M, i, j):
            return False
    return True


# In[21]:


def get_emtpy_tiles(M, up=True, left=True):
    """ Return all most upper/lower - left/right empty tiles that are contiguous with current non-empty tiles"""

    empty_tiles = [(i,j) for i,j in np.argwhere(M == -1) if even(i) and even(j)]

    return empty_tiles[0]


# In[9]:


def get_emtpy_tiles_old(M, up=True, left=True):
    """ Return all most upper/lower - left/right empty tiles that are contiguous with current non-empty tiles"""

    empty_tiles = [(i,j) for i,j in np.argwhere(M == -1) if even(i) and even(j)]

    right = not left
    down  = not up

    if up and right:
        return intersection([max(group) for key,group in itertools.groupby(empty_tiles,operator.itemgetter(0))], [min(group) for key,group in itertools.groupby(sorted(empty_tiles, key=lambda tup: tup[1]),operator.itemgetter(1))])
    if down and left:
        return intersection([min(group) for key,group in itertools.groupby(empty_tiles,operator.itemgetter(0))], [max(group) for key,group in itertools.groupby(sorted(empty_tiles, key=lambda tup: tup[1]),operator.itemgetter(1))])
    if down and right:
        return intersection([max(group) for key,group in itertools.groupby(empty_tiles,operator.itemgetter(0))], [max(group) for key,group in itertools.groupby(sorted(empty_tiles, key=lambda tup: tup[1]),operator.itemgetter(1))])
    else: # default: up and left
        return intersection([min(group) for key,group in itertools.groupby(empty_tiles,operator.itemgetter(0))], [min(group) for key,group in itertools.groupby(sorted(empty_tiles, key=lambda tup: tup[1]),operator.itemgetter(1))])


# In[10]:


def get_neighbors_old(M, up=True, left=True):
    """ Return the valid neighboors (cf. README) of the current vertex M in the tree """

    i,j = choice(get_emtpy_tiles_old(M, up=up, left=left)) # Insert randomness to create different versions of the map at each execution

    #shuffle(idx)
    neighbors = []
    for tile in TILES[idx]:
        M_new = deepcopy(M)
        M_new[i:i+2,j:j+2] = tile

        if is_valid_tile(M_new,i,j):
            neighbors.append(M_new)
        else:
            del M_new

    return neighbors


# In[11]:


def get_neighbors(M, up=True, left=True):
    """ Return the valid neighboors (cf. README) of the current vertex M in the tree """

    i,j = get_emtpy_tiles(M, up=up, left=left) # Insert randomness to create different versions of the map at each execution

    shuffle(idx)
    for tile in TILES[idx]:
        M_new = deepcopy(M)
        M_new[i:i+2,j:j+2] = tile

        if is_valid_tile(M_new,i,j):
            yield M_new
        else:
            del M_new


# In[12]:


def density(M):
    """ Count the number of elements completed in the vertex M """
    return (M != -1).sum()


# In[13]:


def is_full(M):
    """ Check if the vertex M is fully completed """
    return density(M) == M.size


# In[14]:


def print_eta_old(eta, i, stack_len, timer, visited):
    """ Print an estimated time of arrival of the search algorithm, based of the % of completion """
    sys.stdout.write('\r')            
    sys.stdout.write("[{:30s}] {:2.0%} | Step n°{:6d} | Stack: {:6d} | Visited: {:6d} | Time: {:3d}s".format('='*int(eta*30), eta, i, stack_len, visited, timer))
    sys.stdout.flush()


def print_eta(eta, timer):
    sys.stdout.write('\r')
    sys.stdout.write("[{:30s}] {:2.0%} | Time: {:3d}s".format('='*int(eta*30), eta, timer))
    sys.stdout.flush()

# In[15]:


def print_array(M):   
    """ Pretty print a vertex (usefull for DEBUG) """
    print("\n"+"*"*(M.shape[1]*2-1))
    print(np.array_str(M, max_line_width=np.inf).replace("-1", ".").replace("[ ","").replace("[","").replace(" ]","").replace("]","").replace("  ", " ").replace("1","1").replace("\n ","\n"))
    print("*"*(M.shape[1]*2-1))


# In[16]:


def set_direction(M):
    """ Set the optimal direction to perform the search in the graph (cf. README) """
    m, n = M.shape
    i, j = int(m/2), int(n/2)
    M11  = M[:i,:j]
    M10  = M[:i,j:]
    M01  = M[i:,:j]
    M00  = M[i:,j:]

    max_density = max([density(M11), density(M10), density(M01), density(M00)])

    if density(M11) == max_density:
        return (True, True)
    if density(M10) == max_density:
        return (True, False)
    if density(M01) == max_density:
        return (False, True)
    if density(M00) == max_density:
        return (False, False)


# In[17]:


def dfs(M, up, left, t0):
    
    d = density(M)

    if d == M.size:
        return M
   
    for N in get_neighbors(M, up=up, left=left):
        eta = d/M.size
        timer = int(time.time()-t0)
        draw_save_map_png("test",N)
        res = dfs(N, up, left, t0)
        #print_eta(eta, timer)
        if res is not None:
            return res

    return None


# In[18]:


def gen_map(M):

    assert is_valid(M), "Unvalid starting vertex"
    print_array(M)
    
    t0       = time.time()
    visited  = 0
    eta      = 0
    i        = 0
    up, left = set_direction(M)

    res = dfs(M, up, left, t0)
    if res is not None:
        return res
    print("There is no path to complete this matrix.")
    return M


# In[19]:


def gen_map_old(M):
    '''
        This function takes a partially-completed map and returns a fully-generated one which is valid. 
        
        Implementation
        ----------
          Depth-First Search on valid neihboors: The goal is to reduce as much as possible the number of neihboors in order to make the search as fast as possible
            - Trick 1: The algorithm begins by choosing a preferred direction (set_direction) by splitting the matrix in 4 same-sized sub-matrixes and finding the one which is already the most completed (density()). Then the algorithm will always proceed from this corner [let's say for example upper left] to the opposite one [e.g. lower right].
            - Trick 2: For each vertex V(k), a neihboor is defined as vertex V(k+1) which differs by only one tile from V(k). In order to avoid visiting multiple times the same vertex, only one of the of the tile position returned by *get_emtpy_tiles()* with *trick 1* is randomly choosen by the *choice()* made in get_neighbors. Thus there is no need for a hash map of visited vertexes.
            - Trick 3: Given this tile position, each of the 14 avalaible *TILES* are considered in a random order (shuffle()) so that completely different versions of the map are created at each execution of the algorithm. A neihboor V(k+1) is valid (i.e. is put in queue) iff P(V(k+1)|M,V(1),…,V(k)))>0. The function is_valid() returns in fact P(V(k+1)|M,V(1),…,V(k)))>0 (True or False).
            - Trick 4: Search is early-stopped when ONE path is found, that is when the current vertex is a completed and valid version of the map.

        Parameters
        ----------
          M: Array-like (n*d) of integers (i.e., a 2D binary matrix) representing a map (each grid of 4 bits represents a map tile). 

        Returns
        -------
          vertex: Array-like (n*d): A completed (valid) version of the map. 

    '''

    assert is_valid(M), "Unvalid starting vertex"
    
    print_array(M)
    
    t0       = time.time()
    visited  = 0
    eta      = 0
    i        = 0
    up, left = set_direction(M)

    stack = [M]
    vertex = stack.pop()
    while not is_full(vertex):
        
        neighbors = get_neighbors_old(vertex, up=up, left=left)
        visited += len(neighbors)
        stack.extend(neighbors)
        try:
            vertex = stack.pop()
        except Exception as e:
            print("There is no path to complete this matrix.")
            return M

        if i%1 == 0:
            #eta = (vertex != -1).sum()/vertex.size
            #print_eta(eta, i, len(stack), int(time.time()-t0), visited)
            draw_save_map_png("Hello.png", vertex)

        i+=1

    print_eta(1, i, len(stack), int(time.time()-t0), visited)
    print_array(vertex)
    return vertex



# In[23]:


M = np.ones((70,70), dtype=int) * -1
#M[0:2,0:2]   = np.array([[1,1],[0,1]])
M[2:4,2:4]   = np.array([[1,1],[0,0]])
M[2:4,4:6]   = np.array([[1,0],[0,1]])
M[4:6,4:6]   = np.array([[0,1],[0,0]])
M[0:2,6:8]  = np.array([[1,0],[0,1]])
M[8:10,8:10] = np.array([[0,1],[1,0]])

M[4:6,0:2] = np.array([[1,0],[1,1]])

M[-4:-2,-4:-2]   = np.array([[0,1],[0,0]])
M[-4:-2,-8:-6]   = np.array([[1,0],[0,1]])
M[-6:-4,-6:-4]   = np.array([[0,1],[0,0]])
M[-10:-8,-8:-6]  = np.array([[1,0],[0,1]])

M[-24:-22,-24:-22]   = np.array([[0,1],[0,0]])
M[-24:-22,-28:-26]   = np.array([[1,0],[0,1]])
M[-16:-14,-16:-14]   = np.array([[0,1],[0,0]])
M[-20:-18,-18:-16]  = np.array([[1,0],[0,1]])
#
M[2:4,-4:-2]   = np.array([[1,1],[0,0]])
M[2:4,-8:-6]   = np.array([[1,1],[1,1]])

M[-6:-4,4:6]   = np.array([[0,1],[0,0]])
#M[-2:-2]  = np.array([[1,1],[1,1]])
#print_array(M)
N = gen_map_old(M)


# In[ ]:




