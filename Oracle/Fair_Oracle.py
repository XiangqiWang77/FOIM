import numpy as np
import sys
sys.path.append('./fair_influmax_code')
sys.path.append('./Oracle')
from degreeDiscount import degreeDiscountIC, degreeDiscountIC2, degreeDiscountIAC, degreeDiscountIAC2, degreeDiscountStar, degreeDiscountIAC3
from generalGreedy import generalGreedy
from utils import greedy
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from algorithms import algo, maxmin_algo, make_normalized, indicator, make_welfare

from copy import deepcopy
from random import random
import numpy as np

attributes=["key3"]
#key5 for NBA

alpha=-2

def runICmodel (G, S, P):
    ''' Runs independent cascade model.
    Iput: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''

    T = deepcopy(S) # copy already selected nodes
    E = {}

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                #print("w is",w)
                if random() <= 1 - (1-P[T[i]][v]['weight'])**w: # if at least one of edges propagate influence
                    # print T[i], 'influences', v
                    T.append(v)
                    E[(T[i], v)] = 1
        i += 1

    return len(T), T, E


group_size = {}
def Fair_IM_oracle(G,K,currentPg):

    for attribute in attributes:
        nvalues = len(np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()]))
        group_size[attribute] = np.zeros((1, nvalues))

    live_graphs = sample_live_icm(G, 10)
    """
    estimation_error=0
    for graph in live_graphs:
        for (u,v) in graph.edges():
            estimation_error=max(np.abs(G[u][v]['weight']-graph[u][v]['p']),estimation_error)

    print("estimation error is",estimation_error)
    """
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def multi_to_set(f, n = None):
        if n == None:
            n = len(G)
        def f_set(S):
            return f(indicator(S, n))
        return f_set


    def f_multi(x):
        return val_oracle(x, 10).sum()
    
    fair_vals_attr = np.zeros((1, len(attributes)))
    greedy_vals_attr = np.zeros((1, len(attributes)))
    pof = np.zeros((1, len(attributes)))
    include_total = False
    #live_graphs = sample_live_icm(G, 40)
    
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def f_multi(x):
        return val_oracle(x, 10).sum()
        
    #f_multi = make_multilinear_objective_samples(live_graphs, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
    f_set = multi_to_set(f_multi)

    def valoracle_to_single(f, i):
        def f_single(x):
            return f(x, 10)[i]
        return f_single
    #print("about to run greedy")
    #This is the case that the 
    S_opop= degreeDiscountIAC3(G, K, currentPg)
    print("S_opop is",S_opop)
    #print("successfully output S_opop")
    set_to_fair=[]
    for attr_idx, attribute in enumerate(attributes):
        print("attribute",attribute)    
        values = np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()])
        print(values)
        nodes_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
            group_size[attribute][0, vidx] = len(nodes_attr[val])
        #print(nodes_attr[val])

        group_indicator = np.zeros((len(G.nodes()), len(values)))
        for val_idx, val in enumerate(values):
                group_indicator[nodes_attr[val], val_idx] = 1

        val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))

        f_attr = {}
        f_multi_attr = {}
        for vidx, val in enumerate(values):
            nodes_attr[val] = [v for v in G.nodes() if G.node[v]['node_type'][attribute] == val]
            f_multi_attr[val] = valoracle_to_single(val_oracle, vidx)
            f_attr[val] = multi_to_set(f_multi_attr[val])

        S_attr = {}
        opt_attr = {}

        print("another greedy?")
        print(len(values))
        for val in values:
            #S_attr=generalGreedy(G,K,0.1)
            print("value is",int(len(nodes_attr[val])/len(G) * K))
            S_attr[val]= degreeDiscountIAC3(G, int(len(nodes_attr[val])/len(G) * K), currentPg)
            opt_attr[val],_,_=runICmodel(G,S_attr[val], currentPg)
            #S_attr[val], opt_attr[val] = greedy(list(range(len(G))), int(len(nodes_attr[val])/len(G) * K), f_attr[val])

        all_opt = np.array([opt_attr[val] for val in values])

        solver="md"
        threshold = 140
        #Increase the threshold would make the output more optimal, decrease it would make the output more general.
        #threshold 115 for NBA datasets, 2 for alpha-0.5 case, 0.1 for alpha=2 case, -7 for alpha=-2 case, 1 for maxmin case
        #threshold 140 for german dataset, -30 for negative alpha, 4.3 for alpha=0.5, 0.1 for alpha=2
        #threshold 1300 for pokec dataset, 0.2 for maxmin case, 0.05 for alpha=2 case, 1.8 for alpha=0.5, -100 for alpha=-2
        targets = [opt_attr[val] for val in values]
        #print("S_att is",S_attr)
        targets = np.array(targets)
        print('ready to run fair algorithm')
        #The algo overhere is to to run two different algorithm, one focused on Diversity Fairness, another on Maximin fairness

        
        #The first one is on diversity constraint
        fair_x = algo(grad_oracle, val_oracle, threshold, K, group_indicator, np.array(targets), 5, solver)[1:]
        #print("fair_x output")
        #print(fair_x)
        fair_x = fair_x.mean(axis=0)
        set_to_fair=[i for i, x in enumerate(fair_x) if x == 1]
        print("set_to_fair",set_to_fair)
        
        # The second is on the maximin constraint
        """
        print("ready to run maximin oracle")
        grad_oracle_normalized = make_normalized(grad_oracle, group_size[attribute][0])
        val_oracle_normalized = make_normalized(val_oracle, group_size[attribute][0])
        minmax_x = maxmin_algo(grad_oracle_normalized, val_oracle_normalized, threshold, K, group_indicator, 10, 5, 0.1, solver)
        minmax_x = minmax_x.mean(axis=0)
        set_to_fair=[i for i, x in enumerate(minmax_x) if x == 0]
        print("set_to_fair",set_to_fair)
        """
        #The third algorithm is on the welfare function submodular maximization:
        """
        grad_oracle=make_welfare(grad_oracle, group_size[attribute][0], alpha)
        val_oracle=make_welfare(val_oracle,group_size[attribute][0], alpha)
        wel_x = algo(grad_oracle, val_oracle, threshold, K, group_indicator, np.array(targets), 4, solver)[1:]
        wel_x = wel_x.mean(axis=0)
        #set_to_fair=[i for i, x in enumerate(wel_x) if x == 0]
        #1 if alpha<0
        #0 if alpha>0
        set_to_fair=[i for i, x in enumerate(wel_x) if x == 1]
        #Later is for alpha<0
        print("set_to_fair",set_to_fair)
        """
        
        
    set_to_fair=np.unique(set_to_fair)
    #set_to_fair.drop(2000)
    #print("set_to_fair is",set_to_fair)
    #We have parameter to the number of graph
    set_to_fair = [item for item in set_to_fair if 0 <= item <= len(G.nodes())]
    #set_to_fair=set_to_fair.tolist()
    set_to_fair.extend([item for item in S_opop if item not in set_to_fair])
    set_to_fair=set_to_fair[:K]
    #print("set_to_fair is",set_to_fair)
    print("set_to_fair is",set_to_fair)
    return set_to_fair

def test_fair_im_oracle(G,K,currentPg):
    set_to_fair=[107, 603, 932, 983, 1066, 1078, 1140, 1409, 1412, 1430, 1451, 1533, 1544, 2027, 2278, 2286, 2505, 2742, 2747, 2881, 3204, 3226, 3300, 3311, 3507, 3568, 3651, 3660, 3762, 3807, 3820, 3914, 4114, 4242, 4275, 4376, 4406, 4418, 4458, 4632, 4738, 4773, 4775, 4807, 4824, 5089, 5307, 5315, 5335, 5523, 5534, 5569, 5576, 5603, 5699, 5714, 5742, 5761, 5836, 5853, 5867, 6000, 6016, 6020, 6289, 6313, 6328, 6364, 6583, 6992, 7015, 7143, 7182, 7239, 7242, 7250, 7607, 7623, 7687, 7721, 7795, 7929, 8058, 8115, 8155, 8192, 8414, 8484, 8588, 8723, 8803, 9366, 9715, 9981, 10162, 10249, 10582, 10748, 10749, 10777, 11360, 11561, 11571, 11711, 12058, 12084, 12140, 12217, 12410, 13035, 13101, 13415, 13812, 13902, 14301, 15037, 15431, 15827, 16808, 16816, 16849, 17039, 17494, 18025, 18154, 18383, 18530, 18580, 18642]
    #set_to_fair=[18865, 18676, 2859, 9973, 7928, 7402, 3895, 8882, 3023, 5478, 2798, 5948, 4084, 8691, 5941, 6927, 16358, 14530, 7372, 18261, 8386, 9061, 4773, 8190, 6711, 6713, 5487, 18026, 18162, 14578, 17789, 17035, 16154, 18586, 17822, 13799, 13952, 12538, 15291, 5554, 10185, 7678, 10312, 13615, 13623, 15040, 1611, 17963, 3743, 8103, 9224, 4473, 14692, 14506, 4756, 15849, 9741, 566, 7677, 445, 11454, 18406, 12118, 12121, 16266, 8393, 9637, 8880, 6549, 7117, 11131, 8068, 7378, 1829, 2276, 2282, 4430, 3946, 15772, 4592, 2222, 1165, 7820, 8314, 12528, 413, 2791, 14510, 18022, 17212, 1713, 9183, 18506, 18273, 3679, 3626, 3672, 17655, 4097, 4121, 4621, 15736, 13761, 14175, 11701, 14489, 10885, 15981, 13267, 15427, 16797, 3710, 8660, 457, 5384, 2048, 4633, 3502, 7871, 1462, 87, 14539, 6158, 3887, 18558, 14117, 11113, 12531, 12523, 12517, 16126, 5284, 17498, 17645, 13900, 15331, 17931, 10652, 18351, 14284, 14875, 18460, 8705, 14291, 15876, 4099, 18322, 14064, 16096, 10510, 15498, 1181, 12718, 10307, 481, 8758, 16909, 18660, 16584, 15670, 15673, 14952, 16607, 15311, 11764, 12214, 17610, 15007, 1454, 13286, 6869, 12533, 13504, 5359, 5333, 14683, 11957, 1219, 8133, 11538, 9639, 13827, 16363, 15441, 11407, 6738, 2645, 17327, 80, 11750, 16298, 12285, 1619, 13926, 16426, 17885, 12446, 5022, 3569, 14382, 4497, 13972, 14038, 15723, 11961, 6574, 5637, 14944, 14289, 3915, 97, 1865, 15656, 17557, 9563, 12814, 6729, 9871, 6929, 17033, 16259, 14987, 17993, 3214, 3208, 13712, 2597, 5957, 15382, 13886, 16372, 2646, 18606, 14333, 15168, 8377, 2426, 10018, 864, 11486, 15897, 8683, 7683, 7697, 5193, 7454, 9624, 3096, 2988, 4453, 9771, 13544, 18418, 717, 2184, 15059, 934, 7623, 7379, 12698, 14009, 1491, 1795, 18023, 4724, 16811, 2529, 17658, 4214, 12810, 16277, 8372, 12650, 9976, 8898, 12205, 499, 6892, 13935, 14281, 307, 16889, 9522, 16606, 156, 12813, 1233, 4063, 4073, 12495, 2956, 14107, 11276, 16614, 11118, 11137, 11749, 8513, 2607, 2589]
    return set_to_fair

group_size = {}
def optimal_gred(G,K,currentPg):
    for attribute in attributes:
        nvalues = len(np.unique([G.node[v]['node_type'][attribute] for v in G.nodes()]))
        group_size[attribute] = np.zeros((1, nvalues))

    live_graphs = sample_live_icm(G, 10)
    """
    estimation_error=0
    for graph in live_graphs:
        for (u,v) in graph.edges():
            estimation_error=max(np.abs(G[u][v]['weight']-graph[u][v]['p']),estimation_error)

    print("estimation error is",estimation_error)
    """
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def multi_to_set(f, n = None):
        if n == None:
            n = len(G)
        def f_set(S):
            return f(indicator(S, n))
        return f_set


    def f_multi(x):
        return val_oracle(x, 10).sum()
    
    fair_vals_attr = np.zeros((1, len(attributes)))
    greedy_vals_attr = np.zeros((1, len(attributes)))
    pof = np.zeros((1, len(attributes)))
    include_total = False
    #live_graphs = sample_live_icm(G, 40)
    
    group_indicator = np.ones((len(G.nodes()), 1))
        
    val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
    grad_oracle = make_multilinear_gradient_group(live_graphs, group_indicator,  list(G.nodes()), list(G.nodes()), np.ones(len(G)))
        
    def f_multi(x):
        return val_oracle(x, 10).sum()
        
    #f_multi = make_multilinear_objective_samples(live_graphs, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
    f_set = multi_to_set(f_multi)

    def valoracle_to_single(f, i):
        def f_single(x):
            return f(x, 10)[i]
        return f_single
    #print("about to run greedy")
    #This is the case that the 
    S_opop, obj = greedy(list(range(len(G))), K, f_set)
    return list(S_opop)
