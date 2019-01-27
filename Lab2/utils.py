from numpy import *
from sklearn.model_selection import KFold

def get_errors(h,X,Y,k=2):
    '''
        Get the Error Matrix
        --------------------

        Returns the error matrix resulting from applying model h on dataset X,y
         under k-fold cross validation. It should have the same dimension as Y.
    '''
    N,L = Y.shape
    E = zeros((N,L))
    kf = KFold(n_splits=k, shuffle=False)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        h.fit(X_train,y_train)
        y_pred = h.predict(X_test)

        for i in range(y_pred.shape[0]):
            for j in range(L):
                E[test_index[i],j] = (y_pred[i,j] == y_test[i,j])

    return E

def count_combinations(Y):
    '''
        Combination Frequency
        ---------------------

        Returns a dictionary with each label *combination* and its frequency in
         Y (a number between 0 and 1), such that comb2count[array_str(y)] = n 
        where n is the frequency of y ocurring in Y (y is a label vector).
    ''' 
    comb2count = {}
    
    # TODO 
    # HINT: best to cast numpy arrays as, e.g., strings, before using them as 
    #       the key in a dictionary, e.g., array_str(y).
    label_unique_with_count = unique(Y, axis=0, return_counts=True)
    label_unique = label_unique_with_count[0]
    combinations_freq = label_unique_with_count[1]/Y.shape[0]
    
    order_idx = argsort(-combinations_freq)
    combinations_freq = sorted(combinations_freq, reverse=True)
    label_unique = label_unique[order_idx,:]
    for i, label in enumerate(label_unique):
        comb2count[array_str(label).replace("[","").replace("]","")] = combinations_freq[i]
    return comb2count


from matplotlib.pyplot import *

def make_heatmap(S,labels):
    '''
        Return a heatmap plot of a given matrix (S) with its labels (labels).
    '''
    L = len(labels)
    S = S.copy()
    fill_diagonal(S,0.)
    S[triu_indices(L)] = 0
    fig, ax = subplots()
    ax.pcolor(S,cmap=cm.Blues,vmin=-1,vmax=1)
    ax.set_yticks(arange(L)+0.5, minor=False)
    ax.set_xticks(arange(L)+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, minor=False, rotation=45)
    ax.set_yticklabels(labels, minor=False, rotation=45)
    fig.tight_layout()
    return fig

import networkx as nx

def make_graph(A,col_names,min_weight=0.1):
    '''
        Builds a graph given adjacency matrix 'A' and corresponding column names 'col_names'. 
        Only connections of greater weight than threshold weight 'min_weight' are drawn.
    '''

    max_val = A.max()

    G = nx.Graph()

    L,K = A.shape

    widths = []
    for j in range(L):
        G.add_node(col_names[j])
        for k in range(j+1,K):
            if abs(A[j,k]) > min_weight:
                G.add_edge(col_names[j],col_names[k],weight=(abs(A[j,k]) / max_val * 30.),edge_color='r')
                widths.append(abs(A[j,k]) / max_val * 30.)

    pos=nx.spring_layout(G,iterations=10) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos,alpha=0.1)

    # edges
    nx.draw_networkx_edges(G,pos, width=widths,alpha=0.5,edge_color='b',style='dashed')

    nx.draw_networkx_labels(G, pos)

    return G




