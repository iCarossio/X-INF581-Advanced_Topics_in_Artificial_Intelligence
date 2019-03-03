from numpy import *
random.seed(1)

######################################################################################################
## Set up an Environment to work on
######################################################################################################

class Environment:

    nS = 3
    nA = 2

    def __init__(self):

        self.P = random.rand(self.nS,self.nA,self.nS)
        self.P = self.P * (self.P > 0.9)

        self.R = zeros((self.nS,self.nA,self.nS)) 
        self.R[:,0,self.nS-1] = 1
        self.R[:,1,self.nS-1] = 1

        for s in range(self.nS):
            for a in range(self.nA):
                self.P[s,a,:] = random.rand(self.nS) 
                self.P[s,a,:] = self.P[s,a,:] / sum(self.P[s,a,:])

    def draw_graph(self, fname):
        ''' Draw the graph.
        '''
        with open(fname, "w") as text_file:
            text_file.write("digraph MDP {\n")
            for s in range(self.nS):
                text_file.write("\ts_%s [style=filled shape=circle fillcolor=lightblue] ;\n" % (s+1)) 
                for a in range(self.nA):
                    text_file.write("\ta_%s%s [label=\"a_%d\", style=filled, shape=diamond, fillcolor=indianred1, fontsize=10, fixedsize=true, width=0.5, height=0.5] ;\n" % (s+1,a+1,a+1)) 
                    text_file.write("\ts_%s -> a_%s%s ;\n" % (s+1,a+1,a+1)) 
                    for s_ in range(self.nS):
                        if self.R[s,a,s_] > 0:
                            text_file.write("\ta_%s%s -> s_%s [label=\"%2.1f (r=%2.1f)\" color=green] ;\n" % (s+1,a+1,s_+1,self.P[s,a,s_],self.R[s,a,s_])) 
                        else:
                            text_file.write("\ta_%s%s -> s_%s [label=\"%2.1f\"] ;\n" % (s+1,a+1,s_+1,self.P[s,a,s_])) 
            text_file.write("}")


env = Environment()
#env.draw_graph("markov_decision_process.dot")

######################################################################################################
## Task 1
######################################################################################################

# TODO 

######################################################################################################
## Task 2
######################################################################################################

# TODO 

