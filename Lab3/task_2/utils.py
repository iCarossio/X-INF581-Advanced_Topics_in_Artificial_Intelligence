from numpy import *

def print_ptree(branches,filename=None,y_best=None,trans_bg=False):
    '''
        Prints the 'branches' of the tree as a series of strings that can be used to draw a probability tree using graphviz.

        Parameters
        ----------

        branches : a list of tuples the form (A,B,P_branch,P_path) where:
            A : label of the parent node (a list or array of integers)
            B : label of the child node (a list or array of integers)
            P_branch : is a score associated with this branch
            P_path : is a score associated with the full path accumulated until now

        filename : string
            write to this filename, or if None then print to screen

        y_best : an array of integers
            the path to a goal node which gives the best payoff

        trans_bg : boolean
            set to True if you want a transparent background (e.g., for use in slides)

    '''
    graph_properties = "node [shape=\"box\"];"
    if trans_bg:
        graph_properties = "bgcolor=\"#ffffff00\"\nnode [shape=\"box\"];"
    graph_string = []
    for (y_parent,y_child,prob_branch,prob_path) in branches:
        edge_properties = ""
        #if prob_path == True:
        #    edge_properties = "[color=blue]"
        graph_string.append("\t\"%s\" -> { \"%s\" [label=\"%s\"] } %s [label=\"%3.2f\"]" % (str(y_parent),str(y_child),str(y_child),edge_properties,prob_branch))
        if len(y_child) == len(y_best):
            color = "blue"
            if (y_best == y_child).all():
                color = "red"
            graph_string.append("\t\"%s\" -> { \"%s\" [label=\"%3.2f\", shape=oval, color=%s] } %s [dir=none, style=dotted]" % (str(y_child),str(y_child)+'P',prob_path,color,edge_properties))
    graph_string = list(set(graph_string))
    graph_string = ["digraph PTree {\n"] + [graph_properties] + graph_string + ["}"]
    graph_string = '\n'.join(graph_string)
    if filename is None:
        print(graph_string)
    else:
        with open(filename, "w") as text_file:
            text_file.write(graph_string)


if __name__ == "__main__":
    # Test
    branches = [
            ([],[0],0.5,True),
            ([],[1],0.5,False),
            ([0],[0,1],0.2,False),
            ([0],[0,0],0.8,True),
            ([0,0],[0,0,1],0.3,False),
            ([0,1],[0,0,1],0.3,True),
    ]
    print_ptree(branches, y_best=array([0,0,1]))

