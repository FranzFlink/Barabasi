from scipy.stats import linregress

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import networkx as nx
import numpy as np


def initialize_network(num_agents):
   return nx.scale_free_graph(num_agents)

def calculate_adjacency_matrix(G):
    return nx.adjacency_matrix(G).todense()

def form_new_connection(G):
    nodes = np.array(G.nodes())
    out_degrees = np.array([G.out_degree(n) for n in nodes])
    in_degrees = np.array([G.in_degree(n) for n in nodes])

    out_probs = out_degrees / (out_degrees.sum() + 1e-10)
    in_probs = in_degrees / (in_degrees.sum() + 1e-10)


    #dirty fix for probabilites not adding up to one. Just renomralized 
    # here, but where does the inital error in summing come from?
    for probs in [in_probs, out_probs]:
        if ~np.isclose(sum(probs), 1):
            probs = probs / sum(probs)

    # Select nodes for the new connection.
    source = np.random.choice(nodes, p=out_probs)
    target = np.random.choice(nodes, p=in_probs)

    # Add the new connection.
    G.add_edge(source, target)

def add_new_node(G):
    # Add a new node to the graph
    new_node = max(G.nodes()) + 1
    G.add_node(new_node)
    
    # Get existing nodes and their degrees
    nodes = np.array(G.nodes())
    out_degrees = np.array([G.out_degree(n) for n in nodes])
    in_degrees = np.array([G.in_degree(n) for n in nodes])

    # Compute probabilities for selecting nodes for connections
    out_probs = out_degrees / (out_degrees.sum() + 1e-10)
    in_probs = in_degrees / (in_degrees.sum() + 1e-10)

    for probs in [in_probs, out_probs]:
        if ~np.isclose(sum(probs), 1):
            probs = probs / sum(probs)


    for _ in range(5):
        # Select nodes for the new connections
        source = np.random.choice(nodes, p=out_probs)
        target = np.random.choice(nodes, p=in_probs)
        
        # Add outgoing connection from the new node
        G.add_edge(new_node, source)
        
        # Add incoming connection to the new node
        G.add_edge(target, new_node)


def measure_system(G):
    adjacency_matrix = calculate_adjacency_matrix(G)
    number_of_nodes = adjacency_matrix.shape[0]
    number_of_nodes = G.number_of_nodes()
    alpha_matrix = np.zeros(adjacency_matrix.shape)
    U_in = np.zeros(number_of_nodes)
    U_out = np.zeros(number_of_nodes)
    U_i = np.zeros(number_of_nodes)
    c = np.zeros(number_of_nodes)

    for i in range(G.number_of_nodes()):
        node = list(G.nodes)
        for j in range(G.number_of_nodes()):
            if adjacency_matrix[i,j] != 0:
                k_in, k_out = G.in_degree(node[i]), G.out_degree(node[j])
                alpha = 2 / (1 + np.exp(-(k_out - k_in)))
                alpha_matrix[j,i] = alpha
                
    for i in range(G.nodes().__len__()):
        U_in[i] = np.sum(1-alpha_matrix[i,:])
        U_out[i] = np.sum(alpha_matrix[:,i]-1)
        U_i[i] = U_in[i] + U_out[i]
        c[i] = U_i[i] / U_out[i]
    
    return U_i, c


# 3. Calculate energy balances, leverages, and handle bankruptcies.
def update_agents(G, cth, pos):
    nodes_to_remove = []
    U_i, c = measure_system(G)

    #for node in G.nodes:
    #
    #    _, _, ci = measure_system(node)

    for i, node in enumerate(G.nodes):
        ci = c[i]
        if ci < 0.05 or 0.99 < np.random.random():
            nodes_to_remove.append(node)  # Add the node to the list of nodes to be removed.
    for node in nodes_to_remove:
        G.remove_node(node)
        del pos[node]

    #measuring the system twice... very ellegant ;) 
    U_i, c = measure_system(G)

    # Recalculate leverage and energy balance for the remaining nodes.
    #U_in_sum_list, U_out_sum_list = [], []
    #leverage = []
    #for node in G.nodes:
    #    U_in_sum, U_out_sum, ci = measure_system(node)
    #    leverage.append(ci)
    #    U_in_sum_list.append(U_in_sum)
    #    U_out_sum_list.append(U_out_sum)

    return U_i, c, pos



# Now, we can initialize the network and simulate its evolution.
G = initialize_network(100)
np.random.seed(42)

cth = .3  # Set a threshold for bankruptcy.
pos = nx.random_layout(G)



fig, axs = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [4, 1]})

def update(num, G=G, pos_0=pos):
    global pos
    axs[0].cla()
    axs[1].cla()
    xmin, xmax = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()


    random_add = np.random.randint(1, 10)
    random_add = 20

    for _ in range(random_add):
        add_new_node(G)

    for node in G.nodes:
        if node not in pos:
            pos[node] = ((xmax - xmin) * random.random() + xmin, (ymax - ymin) * random.random() + ymin)

    if num == 0:
        U_i, leverage, pos = update_agents(G, cth, pos_0)
    else:
        U_i, leverage, pos = update_agents(G, cth, pos)

    for node in G.nodes:
        if node not in pos:
            pos[node] = ((xmax - xmin) * random.random() + xmin, (ymax - ymin) * random.random() + ymin)

    clusters = nx.algorithms.community.greedy_modularity_communities(G)

    # Calculate cluster centroids
    centroids = {i: np.mean([pos[node] for node in cluster], axis=0) for i, cluster in enumerate(clusters)}
    # Move nodes towards their cluster centroid
    for i, cluster in enumerate(clusters):
        for node in cluster:
            distance_to_centroid = np.linalg.norm(np.array(pos[node]) - np.array(centroids[i]))
            if distance_to_centroid > 0.25:
                pos[node] = tuple(0.99 * np.array(pos[node]) + 0.01 * np.array(centroids[i]))


    nx.draw_networkx_nodes(G, pos, node_color=leverage, node_size=leverage * 50, cmap=plt.cm.jet, ax=axs[0])
    for node in G.nodes:
        outgoing_edges = G.out_edges(node)
        incoming_edges = G.in_edges(node)
    
        nx.draw_networkx_edges(G, pos, edgelist=outgoing_edges, edge_color='green', alpha=0.4, arrowsize=2, arrowstyle='->', ax=axs[0])
        nx.draw_networkx_edges(G, pos, edgelist=incoming_edges, edge_color='red', alpha=0.4, arrowsize=2, arrowstyle='->', ax=axs[0])

    try:
        axs[1].hist(leverage, bins=10, color='blue', alpha=0.5)
    except:
        pass
    #axs[1].set_xscale('log')
    axs[1].set_title('Distribution of U_T')
    axs[1].set_xlabel('U_T')
    axs[1].set_ylabel('Frequency')


    if num == 0:
        print('Step \t Energy \t Leverage \t N_agents')
        print(f'{num} \t {np.mean(U_i):.2f}±{np.std(U_i):.2f} \t {np.mean(leverage):.2f}±{np.std(leverage):.2f} \t {len(G.nodes)}')
    else:
        print(f'{num} \t {np.mean(U_i):.2f}±{np.std(U_i):.2f} \t {np.mean(leverage):.2f}±{np.std(leverage):.2f} \t {len(G.nodes)}')

    return pos





#_ ,G = update(1, G)

ani = FuncAnimation(fig, update, frames=range(50), repeat=False)
ani.save("finance_graph_v0.1.gif", writer='ffmpeg', fps=2, dpi=300)
