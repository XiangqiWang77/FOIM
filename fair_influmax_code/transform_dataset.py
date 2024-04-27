# We need to change the dataset like NBA, NBA, or NetHEPT, so that it can be used in check_fariness
dataset="NBA"
import pickle
import ast

graph_address = '../datasets/NBA/nba_relationship.G'
prob_address = '../datasets/NBA/prob.dic'
param_address = '../datasets/NBA/parameter.dic'
edge_feature_address = '../datasets/NBA/edge_feature.dic'
## For simulation on Fair IM, we need to choose a different dataset which includes class information
# Choose the third column of edge feature as class feature
class_feature_address = '../datasets/NBA/country.dic'

G = pickle.load(open(graph_address, 'rb'), encoding='latin1')
prob = pickle.load(open(prob_address, 'rb'), encoding='latin1')
prob = ast.literal_eval(prob)
parameter = pickle.load(open(param_address, 'rb'), encoding='latin1')
parameter = ast.literal_eval(parameter)
feature_dic = pickle.load(open(edge_feature_address, 'rb'), encoding='latin1')
feature_dic= ast.literal_eval(feature_dic)
class_feature = pickle.load(open(class_feature_address, 'rb'),encoding='latin1')

Node_attr={}
for key in parameter:
    if key in G.nodes():
        value=parameter[key]
        merged_list = [val for sublist in value for val in sublist]
        my_dict = {f'key{i+1}': val for i, val in enumerate(merged_list)}
        my_dict = {key: my_dict}
        Node_attr.update(my_dict)

import networkx as nx
#print(Node_attr)
nx.set_node_attributes(G, values = Node_attr, name='node_type')

with open('./networks/NBA.pickle', 'wb') as f:
    pickle.dump(G, f)