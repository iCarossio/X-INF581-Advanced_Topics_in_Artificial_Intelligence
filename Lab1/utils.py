from numpy import * 

pairs = array([(i, j) for i in range(5) for j in range(5)])

def tile2pos(i):
    '''
        Return position coordinates (as a numpy array) given tile number.
    '''
    if i < 0:
        return None
    return pairs[int(i)]

def pos2tile(y,width=5):
    '''
        Return tile ID given coordinates in 2D numpy array 'y'.
    '''
    i = y[0] * width + y[1]
    if i < (width * width):
        return i
    return -1


if __name__ == "__main__":
    # Test
    for n in range(100):
        i = random.choice(25)
        print(i,pos2tile(tile2pos(i)))
