import numpy as np
from utils import greedy
from scipy.linalg import fractional_matrix_power
from numba import jit
import random


def indicator(S, n):
    x = np.zeros(n)
    x[list(S)] = 1
    return x

def multi_to_set(f, n):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def make_weighted(f, weights, *args):
    def weighted(x):
        return np.dot(weights, f(x, *args))
    return weighted

def make_normalized(f, targets):
    def normalized(x, *args):
        return f(x, *args)@np.diag(1./targets)
    return normalized

stor_TO_B=[]
def make_welfare(f, targets,alpha=-2):
    def wel(x, *args):
        #portion=f(x, *args)@np.diag(1./targets)
        #portion=np.mean(portion)
        TO_B=f(x, *args)
        #print("targets",targets)
        #print(TO_B)
        global stor_TO_B
        if TO_B is not None:
            matrix=TO_B@np.diag(1./targets)
            stor_TO_B=TO_B
        else:
            #TO_B=stor_TO_B
            matrix=stor_TO_B@np.diag(1./targets)
        #print("matrix",matrix)
        max_dim = max(matrix.shape)
        #print("max_dim is",max_dim)
        #A=np.linalg.inv(matrix)
        #padded_matrix = np.pad(matrix, ((0,abs(max_dim - matrix.shape[0])), (0, abs(max_dim - matrix.shape[1]))), mode="linear_ramp")
        #padded_matrix=np.linalg.pinv(padded_matrix)
        #print("padded_matrix is",padded_matrix)
         #padded_matrix = np.pad(matrix, ((0, max_dim - matrix.shape[0]), (0, max_dim - matrix.shape[1])))
        temp=len(targets)*np.power(matrix,alpha)/alpha
        #result = temp[:matrix.shape[0], :matrix.shape[1]]
        return temp
    return wel

def make_contracted_function(f, S):
    '''
    For any function defined on the multilinear extension which takes x as its
    first argument, return a function which conditions on all items in S being
    chosen. 
    '''
    def contracted(x, *args):
        x = x.copy()
        x[list(S)] = 1
        return f(x, *args)
    return contracted

def is_2d_list(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return isinstance(lst[0], list)
    return False

def mirror_sp(x, grad_oracle, k, group_indicator, group_targets, num_iter, step_size = 5, batch_size = 20, verbose=False):
    '''
    Uses stochastic saddle point mirror descent to solve the inner maxmin linear
    optimization problem.
    '''
    #random initialization
    startstep = step_size
    m = group_indicator.shape[1]
    v = x + 0.01*np.random.rand(*x.shape)
    v[v > 1] = 1
    v = k*v/v.sum()
    y = (1./m) * np.ones(m)
    group_weights = 1./group_targets
    group_weights[group_targets <= 0 ] = 0
    #historical mixed strategy for defender
    for t in range(num_iter):
        step_size = startstep/np.sqrt(t+1)
        #get a stochastic estimate of the gradient for each player
        #print("x before grad_oracle",x)
        g = grad_oracle(x, batch_size)
        #gh=[]
        #for i in range(v.shape[0]):
        #    gh.append(g)
        temp_we=[0.00202429]*v.shape[0]
        #print(temp_we)
        if g is None:
            g=grad_oracle(x, 10)
            print("tuned None")
        if is_2d_list(g)==False:
            g=np.diag(temp_we)[:v.shape[0],:y.shape[0]]
            #g[np.isneginf(g)] = -1
        #print("g",g)
        else:
            for i in range(len(g)):
                g[i][np.isneginf(g[i])] = -1
        #print("g is",g.shape)
        #print("group weights",np.diag(group_weights))
        group_grad = g @ np.diag(group_weights)
        #print("shape of group_grad",group_grad.shape)
        #print("shape of y",y.shape)
        #print(v.shape[0])
        #Pad the group_grad matrix:
        #print("original matrix",group_grad)
        #padded_matrix = np.pad(group_grad, ((0,v.shape[1] - group_grad.shape[0]), (0, y.shape[0] - group_grad.shape[1])), mode="linear_ramp")
        #group_grad=padded_matrix
        #print("group_grad matrix",group_grad)
        #print("group_grad matrix",group_grad.shape)
        #print("v matrix",v.shape)
        #group_grad=np.array(group_grad)[:v.shape[1],y.shape[0]]
        #print("new matrix",group_grad)
        grad_v = group_grad@y
        grad_y = v@group_grad
        #Use broadcast
        #gradient step
        v = v * np.exp(step_size*grad_v)
        y = y*np.exp(-step_size*grad_y)
        #bregman projection
        v[v > 1] = 1
        v = k*v/v.sum()
        y = y/y.sum()
    if(np.any(np.isnan(v))):
        v = x + 0.01*np.random.rand(*x.shape)
        v[v > 1] = 1
        v = k*v/v.sum()
    else:
        return v

def lp_minmax(x, grad_oracle, k, group_indicator, group_targets):
    import gurobipy as gp
    m = gp.Model()
    m.setParam( 'OutputFlag', False )
    v = m.addVars(range(len(x)), lb = 0, ub = 1)
    obj = m.addVar()
    m.update()
    m.addConstr(gp.quicksum(v) <= k)
    g = grad_oracle(x, 5000)
    for i in range(len(group_targets)):
        if group_targets[i] > 0:
            m.addConstr(obj <= gp.quicksum(v[j]*g[j, i] for j in range(len(v)))/group_targets[i])
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return np.array([v[i].x for i in range(len(v))])

def rounding(x):
    '''
    Rounding algorithm that does not require decomposition of x into bases
    of the matroid polytope
    '''
    import random
    i = 0
    j = 1
    x = x.copy()
    for t in range(len(x)-1):
        if x[i] == 0 and x[j] == 0:
            i = max((i,j)) + 1
        elif x[i] + x[j] < 1:
            if random.random() < x[i]/(x[i] + x[j]):
                x[i] = x[i] + x[j]
                x[j] = 0
                j = max((i,j)) + 1
            else:
                x[j] = x[i] + x[j]
                x[i] = 0
                i = max((i,j)) + 1
        else:
            if random.random() < (1 - x[j])/(2 - x[i] - x[j]):
                x[j] = x[i] + x[j] - 1
                x[i] = 1
                i = max((i,j)) + 1

            else:
                x[i] = x[i] + x[j] - 1
                x[j] = 1
                j = max((i,j)) + 1
    return x

def multiobjective_fw(grad_oracle, val_oracle, k, group_indicator, group_targets, num_iter, solver = 'md'):
    '''
    Uses FW updates to find a fractional point meeting a threshold value for each 
    of the objectives
    '''
#    startstep = stepsize
    x = np.zeros((num_iter, group_indicator.shape[0]))
    #print("shape of x",x.shape)
    #print("shape of group_targets",group_targets.shape)
    for t in range(num_iter-1):
        #how far each group currently is from the target
        #print("shape of temp",val_oracle((1/num_iter)*x[0:t].sum(axis=0), 5) )
        iter_targets = group_targets - val_oracle((1/num_iter)*x[0:t].sum(axis=0), 10) 
        if np.all(iter_targets < 0):
            print('all targets met')
            iter_targets = np.ones(len(group_targets))
#        iter_targets = group_targets - val_oracle(x[t], 100) 
#        print(iter_targets)
#        iter_targets[iter_targets < 0] = 0
#        print(iter_targets.min(), iter_targets.max())
        #Frank-Wolfe update
#        stepsize = startstep/np.sqrt(2*t+1)
        if solver == 'md':
            #print(x.shape)
            #print("x_",x[(t)])
            #print("before mirror_sp input x is",(1./(t+1))*x[0:(t+1)].sum(axis=0))
            
            m = mirror_sp((1./(t+1))*x[0:(t+1)].sum(axis=0), grad_oracle, k, group_indicator, iter_targets, num_iter=20)
            #m=m.astype('float')
            if isinstance(m, list):
                x[t+1]=m
            else:
                m=(1./(t+1))*x[0:(t+1)].sum(axis=0)
                x[t+1]=m
        elif solver == 'gurobi':
            x[t+1] = lp_minmax((1./(t+1))*x[0:(t+1)].sum(axis=0), grad_oracle, k, group_indicator, iter_targets)
        else:
            raise Exception('solver must be either md or gurobi')
    return x


def greedy_top_k(grad, elements, budget):
    '''
    Greedily select budget number of elements with highest weight according to
    grad
    ''' 
    import numpy as np
    inx = np.argpartition(grad, -budget)[-budget:]
    indicator = np.zeros(len(grad))
    indicator[inx[:budget]] = 1
    return indicator


def fw(grad_oracle, val_oracle, threshold, k, group_indicator, group_targets, num_iter, stepsize= 1):
    '''
    Run the normal frank-wolf algorithm to maximize a single submodular function
    '''
    x = np.zeros((num_iter, group_indicator.shape[0]))
    for t in range(num_iter-1):
        grad = grad_oracle((1./num_iter)*x[0:t].sum(axis=0), 10).sum(axis=1)
        x[t+1] = greedy_top_k(grad, None, k)

    return x


def threshold_include(n_items, val_oracle, threshold):
    '''
    Makes a single pass through the items and returns a set including every
    item whose singleton value is at least threshold for some objective
    '''
    to_include = []
    x = np.zeros((n_items))
    for i in range(n_items):
        #print("i",i)
        x[i] = 1
        if random.choice([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]):
            #print("chosen")
            vals = val_oracle(x, 2)
            #print(vals.max(),threshold)
            if vals.max() >= threshold:
                to_include.append(i)
        x[i] = 0
        """
        vals = val_oracle(x, 2)
        #print(vals.max(),threshold)
        if vals.max() >= threshold:
            to_include.append(i)
        x[i] = 0
        """
    return to_include


def algo(grad_oracle, val_oracle, threshold, k, group_indicator, group_targets, num_iter, solver):
    '''
    Combine algorithm that runs the first thresholding stage and then FW
    '''
    S = threshold_include(group_indicator.shape[0], val_oracle, threshold)
    print("S is ",S)
    grad_oracle = make_contracted_function(grad_oracle, S)
    val_oracle = make_contracted_function(val_oracle, S)
    #print("successfully build oracles")
    x = multiobjective_fw(grad_oracle, val_oracle, k - len(S), group_indicator, group_targets, num_iter, solver)
    #print("x length is",len(x))
    #print("shape of x is",x.shape())
    #print("S is ",S)
    x[:, list(S)] = 1
    #x[:, list(S)] = 1
    
    return x


def maxmin_algo(grad_oracle, val_oracle, threshold, k, group_indicator, num_iter, num_bin_iter, eps, solver):
    S = threshold_include(group_indicator.shape[0], val_oracle, threshold)
    grad_oracle = make_contracted_function(grad_oracle, S)
    val_oracle = make_contracted_function(val_oracle, S)
    ub = 1.
    lb = 0
    iternum = 0 
    while ub - lb > eps and iternum < num_bin_iter:
        target = (ub + lb)/2
#        print(target)
        group_targets = np.zeros(group_indicator.shape[1])
        group_targets[:] = target
        x = algo(grad_oracle, val_oracle, threshold, k, group_indicator, group_targets, num_iter, solver)[1:]
        vals = val_oracle(x.mean(axis=0), 10)
        if vals.min() > target:
            lb = target
        else:
            ub = target
        iternum += 1
    return x



