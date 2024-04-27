
# We first give the MIP oracle for maximin case.
# We additionally give the MIP oracle for diversity case.
#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
import numpy as np
import sys
from gurobipy import Model, GRB, quicksum, LinExpr
import networkx as nx
import pickle
import numpy, random
import argparse
from MIPDC import Mont_Carlo_Samplig, stage_1_MIP, induce_sub_graphs

attribute=["key5"]

def MIP_IM(input_graph,budget, currentPg):
    m=10
    main_graph = input_graph

    samples = []
    for j in range(m):
        G = nx.DiGraph()
        for u in main_graph.nodes():
            G.add_node(u)
        for u,v in main_graph.edges():
            if main_graph[u][v]['weight']> random.random():
                G.add_edge(u, v)
        samples.append(G)


    model = Model('group_maximin_'+'Youtube')
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 60)
    
    min_value = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    
    mvars = []
    #active nodes
    avars = []
    #seed nodes
    svars = []
    var_seed_dict = {}
    var_active_dict = {}
    var_mean_dict = {}

    for j in range(len(main_graph.nodes())):
        s = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
        svars.append(s)
        var_seed_dict[j] = s

    for sample_index, sample in enumerate(samples):
        for j in range(len(main_graph.nodes())):
            a = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
            avars.append(a)
            var_active_dict[(sample_index,j)] = a    

    mvars.append(avars)
    mvars.append(svars)

    obj_expr = quicksum(avars)
    
    model.setObjective(min_value, GRB.MAXIMIZE)
    model.addConstr(quicksum(svars), GRB.LESS_EQUAL, budget)

    for sample_index, sample in enumerate(samples):
        for i in range(len(main_graph.nodes())):
            neighbors = nx.ancestors(sample, i) 
            e = len(neighbors)
            ai = var_active_dict[(sample_index,i)]
            si = var_seed_dict[i]
            if i in  var_mean_dict.keys():
                var_mean_dict[i]+= ai
            else:
                var_mean_dict[i]= ai
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


    labels = nx.get_node_attributes(main_graph, "node_type")
    temp_label={}
    for keys,dictions in labels.items():
        #print(type(dictions))
        temp_label[keys]=dictions["key2"]
    labels=temp_label
    label_dict = {}
    for i in range(len(main_graph.nodes())):
        label = labels[i]
        if label in label_dict.keys():
            label_dict[label].append(i)
        else:
            label_dict[label] = [i]     

    mean_label_dict = {}
    for label in label_dict.keys():
        for node in label_dict[label]:
            if label in mean_label_dict.keys():
                mean_label_dict[label].append(var_mean_dict[node])
            else:
                mean_label_dict[label]= [var_mean_dict[node]]
        label_size = len(label_dict[label])
        expr = quicksum(mean_label_dict[label])
        model.addConstr(expr, GRB.GREATER_EQUAL, label_size *m* min_value)


    try:
        model.optimize()
    except e:
        print(e)
    print("var_seed_dict", var_seed_dict)
    objective_value = -1
    return_list=[]
    if (model.solCount>0):
        objective_value = model.ObjVal
        #with open(output_file, "w") as of:    
        for key,value in var_seed_dict.items():
            if(value.x > 0):
                return_list.append(key)
    
    return return_list
    


def MIP_IM_DC( input_graph,budget, currentPg):
    #budget = 25
    m=10
    #with open(input_graph, "rb") as f:
    main_graph = input_graph
    labels = nx.get_node_attributes(main_graph, "node_type")
    temp_label={}
    for keys,dictions in labels.items():
        #print(type(dictions))
        temp_label[keys]=dictions["key5"]
    labels=temp_label
    #print(labels)
    label_dict = {}
    for i in range(len(main_graph.nodes())):
        label = labels[i]
        if label in label_dict.keys():
            label_dict[label].append(i)
        else:
            label_dict[label] = [i] 
    index = random.randint(0,100)
    opt_dict = stage_1_MIP(main_graph, label_dict, budget ,m, index)
    print(opt_dict)
    print(len(opt_dict))
    return_list=stage_2_MIP(main_graph, opt_dict, attribute, label_dict, budget ,m, index)
    print(return_list)
    print(len(return_list))
    return return_list



def stage_2_MIP(main_graph, opt_dict, attribute, label_dict, budget ,m, index):
    print("stage_2_MIP")
    samples =  Mont_Carlo_Samplig(main_graph, m)
    model_name = 'DC_stage_2'+str(attribute)+'_'+str(index)
    model = Model(model_name)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 5)
        
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
    
    try:
        model.optimize()
    except e:
        print(e) 
        
        
    objective_value = -1
    print(var_seed_dict)
    return_list=[]
    if (model.solCount>0):
        objective_value = model.ObjVal
        #with open(output_file, "w") as of:    
        for key,value in var_seed_dict.items():
            if(value.x > 0):
                return_list.append(key)
    
    return return_list

    


