from numpy import *

class Environment:

    # pi ( e.g., p(y = 1) = 0.1 )
    pi = array([0.9, 0.1])

    K = len(pi)

    # phi ( e.g., p(x = 1 | y = 0) = g[x,y] = g[1,0] =  0.1 ) 
    g = array([[0.8,0.2],
               [0.1,0.9]])

    # theta ( e.g., p(y' = 1 | y = 0) = f[y',y] = f[1,0] =  0.3 )
    f = array([[0.6, 0.4], 
               [0.3, 0.7]])


    def __init__(self):
        pass



def viterbi(x, env):
    """
        Viterbi Algorithm (MPE / MAP)
        -----------------------------

        x : observation
        env : environment (models)

        Returns
        -------

        both 
            - the most likely sequence argmax_y p(y|x) and 
            - the value max_y p(y|x).
    """
    T = len(x)      # infer T

    # TODO

    return y_argmax, y_max


env = Environment()
x = array([0,0,0,0,1,1,1,1,0])
y_max, y_prob = viterbi(x, env)

print("-----")
print(y_max, y_prob)
