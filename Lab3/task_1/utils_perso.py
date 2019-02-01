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

def draw_save_map_png(fname, tile_codes, tile_size=32):
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

    quit = False
    while not quit:
        time.sleep(0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True


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
