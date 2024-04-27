#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
from gurobipy import Model, GRB, quicksum, LinExpr
import networkx as nx
import pickle
import numpy, random


# In[2]:


def OPT(model_name, main_graph, samples, budget ,m):
    model = Model(model_name)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 1800)
    
    mvars = []
    #active nodes
    avars = []
    #seed nodes
    svars = []
    var_seed_dict = {}
    var_active_dict = {}


    for node in main_graph.nodes():
        s = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
        svars.append(s)
        var_seed_dict[node] = s

    for sample_index, sample in enumerate(samples):
        for node in main_graph.nodes():
            a = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
            avars.append(a)
            var_active_dict[(sample_index,node)] = a    

    mvars.append(avars)
    mvars.append(svars)

    obj_expr = quicksum(avars)
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.addConstr(quicksum(svars), GRB.LESS_EQUAL, budget)

    for sample_index, sample in enumerate(samples):
        for i in main_graph.nodes():
            if (sample.has_node(i)):
                neighbors = nx.ancestors(sample, i) 
                e = len(neighbors)
                ai = var_active_dict[(sample_index,i)]
                si = var_seed_dict[i]
                neighbors_active_vars = []
                neighbors_seed_vars = []
                neighbors_active_vars.append(((e+1), ai))
                neighbors_seed_vars.append(si)
                for neighbor in neighbors:
                    neighbors_active_vars.append(((e+1), var_active_dict[(sample_index,neighbor)]))
                    neighbors_seed_vars.append(var_seed_dict[neighbor])
                seed_neighbors = quicksum(neighbors_seed_vars)
                model.addConstr(ai, GRB.LESS_EQUAL, seed_neighbors)
                model.addConstr(seed_neighbors, GRB.LESS_EQUAL, LinExpr(neighbors_active_vars))
    try:
        model.optimize()
    except e:
        print(e)       
            
    objective = model.ObjVal     
    return objective



# In[3]:


def stage_1_MIP(main_graph, label_dict, budget ,m, index):
    opt_dict = {}
    sub_graphs = induce_sub_graphs(main_graph, label_dict)
    
    for label, sub_graph in sub_graphs.items():
        sub_samples = Mont_Carlo_Samplig(sub_graph, m)
        estimation_error=0
    #print(sub_graphs)
        """
        average_degrees = {}
        for sub_graph in sub_samples:
            degrees = dict(sub_graph.degree())
            for node, degree in degrees.items():
                if node not in average_degrees:
                    average_degrees[node] = []
                average_degrees[node].append(degree)
        for node, degrees in average_degrees.items():
            average_degrees[node] = sum(degrees) / len(degrees)
        for graph in sub_samples:
            degree = dict(graph.degree())
            for (u,v) in graph.edges():
                estimation_error=max(numpy.abs(main_graph[u][v]['weight']-1 / average_degrees[u]),estimation_error)
        """
        #print("estimation error is",estimation_error)
        #print("sub_samples",sub_samples)
        model_name = 'DC_stage_1'+str(label)+'_'+str(index)
        sample_budget = m*(len(label_dict[label])/len(main_graph.nodes()))
        objective = OPT(model_name, sub_graph, sub_samples, sample_budget ,m)
        opt_dict[label] = objective
    print("stage_1_MIP finished")
    return opt_dict


# In[4]:


def stage_2_MIP(main_graph, opt_dict, attribute, label_dict, budget ,m, index):
    samples =  Mont_Carlo_Samplig(main_graph, m)
    model_name = 'DC_stage_2'+str(attribute)+'_'+str(index)
    model = Model(model_name)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 1800)
        
    mvars = []
    #active nodes
    avars = []
    #seed nodes
    svars = []
    var_seed_dict = {}
    var_active_dict = {}

    for j in main_graph.nodes():
        s = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
        svars.append(s)
        var_seed_dict[j] = s

    for sample_index, sample in enumerate(samples):
        for j in main_graph.nodes():
            a = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
            avars.append(a)
            var_active_dict[(sample_index,j)] = a    

    mvars.append(avars)
    mvars.append(svars)

    obj_expr = quicksum(avars)
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.addConstr(quicksum(svars), GRB.LESS_EQUAL, budget)

    for sample_index, sample in enumerate(samples):
        for i in main_graph.nodes():
            if (sample.has_node(i)):
                neighbors = nx.ancestors(sample, i) 
                e = len(neighbors)
                ai = var_active_dict[(sample_index,i)]
                si = var_seed_dict[i]
                neighbors_active_vars = []
                neighbors_seed_vars = []
                neighbors_active_vars.append(((e+1), ai))
                neighbors_seed_vars.append(si)
                for neighbor in neighbors:
                    neighbors_active_vars.append(((e+1), var_active_dict[(sample_index,neighbor)]))
                    neighbors_seed_vars.append(var_seed_dict[neighbor])
                seed_neighbors = quicksum(neighbors_seed_vars)
                model.addConstr(ai, GRB.LESS_EQUAL, seed_neighbors)
                model.addConstr(seed_neighbors, GRB.LESS_EQUAL, LinExpr(neighbors_active_vars))
            
    for label,node_labels in label_dict.items():
        label_vars = []
        for sample_index, sample in enumerate(samples): 
            for node in node_labels:
                label_vars.append(var_active_dict[(sample_index,neighbor)])
        expr = quicksum(label_vars)
        model.addConstr(expr, GRB.GREATER_EQUAL , opt_dict[label])    
    
    with open('../../../../Git/influence_maximization/experiments/im500/results/fairmip/dc/base100/'+attribute+'/output_'+str(index)+'.txt', "w") as of:    
        for key,value in var_seed_dict.items():
            if(value.x > 0):
                print(key, file=of)
    
    
    try:
        model.optimize()
    except e:
        print(e) 
    
    


# In[5]:


def induce_sub_graphs(main_graph, label_dict):
    sub_graphs = {}
    for label, label_nodes in label_dict.items():
        sub_graphs[label] = main_graph.subgraph(label_nodes)
    return sub_graphs


# In[6]:


def Mont_Carlo_Samplig(main_graph, m):
    samples = []
    for j in range(m):
        G = nx.DiGraph()
        for u in main_graph.nodes():
            G.add_node(u)
        for u,v in main_graph.edges():
            if main_graph[u][v]['weight']> random.random():
                G.add_edge(u, v)
        samples.append(G)
    return samples


# In[7]:


def MIP_IM():
    attributes = ['gender', 'age', 'region', 'ethnicity']
    budget = 25
    m = 100
    for attribute in attributes:
        for index in range(20):
            with open('../MIP/data/networks_prob/graph_spa_500_'+str(index)+'.pickle', "rb") as f:
                main_graph = pickle.load(f)
                labels = nx.get_node_attributes(main_graph, attribute)
                label_dict = {}
                for i in range(len(main_graph.nodes())):
                    label = labels[i].encode('utf-8')
                    if label in label_dict.keys():
                        label_dict[label].append(i)
                    else:
                        label_dict[label] = [i] 
                opt_dict = stage_1_MIP(main_graph, label_dict, budget ,m, index)
                stage_2_MIP(main_graph, opt_dict, attribute, label_dict, budget ,m, index)


# In[8]:


#MIP_IM()

