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

# 1. Initialize the network with nodes and their connections.
def initialize_network(num_agents):
    #G = nx.DiGraph()  # Directed graph: connections are not symmetric.
    #G.add_nodes_from(range(num_agents))  # Add nodes.
    #G = nx.gnp_random_graph(num_agents, .3, directed=True)
    G = nx.scale_free_graph(num_agents)
    return G


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



def measure_system(node):
    k_out = G.out_degree(node)
    k_in = G.in_degree(node)    

    ci = k_out / (k_in + 1e-10) - 1
    
    print(ci)
    return 0, 0, ci


    # Compute αij for outgoing and incoming connections.

    alpha_out_list = [2/(1 + np.exp(-(k_out - G.in_degree(n)))) for n in G.successors(node)]
    alpha_in_list = [2/(1 + np.exp(-(G.out_degree(n) - k_in))) for n in G.predecessors(node)]

    for a_i, a_o in zip(alpha_out_list, alpha_in_list):
        if a_o < 0:
            a_o = 0
        if a_o > 2:
            a_o = 2
        if a_i < 0:
            a_i = 0
        if a_i > 2:
            a_i = 2      

    #alpha_out = {n: 2/(1 + np.exp(-(k_out - G.in_degree(n)))) for n in G.successors(node)}
    #alpha_in = {n: 2/(1 + np.exp(-(G.out_degree(n) - k_in))) for n in G.predecessors(node)}
    alpha_out = alpha_out_list
    alpha_in = alpha_in_list


    if alpha_out == {} or alpha_in == {}:
        return 0, 0, 0




    # Compute energy balances Uij and total energy balance Ui.
    #U_out = {n: 1 - alpha for n, alpha in alpha_out.items()}
    #U_in = {n: alpha - 1 for n, alpha in alpha_in.items()}
    #Ui = sum(U_out.values()) + sum(U_in.values())


    U_out_sum = sum([value for value in alpha_out])
    U_in_sum = sum([value for value in alpha_in])
    #if U_in_sum == 0:
    #    U_in_sum = 1e4

    ci = 1 +  U_out_sum / U_in_sum
    #print(ci)


    return U_out_sum, U_in_sum, ci, 


# 3. Calculate energy balances, leverages, and handle bankruptcies.
def update_agents(G, cth, pos):
    nodes_to_remove = []

    for node in G.nodes:

        _, _, ci = measure_system(node)

        if ci < cth or 0.99 < np.random.random():
            nodes_to_remove.append(node)  # Add the node to the list of nodes to be removed.
    
    for node in nodes_to_remove:
        G.remove_node(node)
        del pos[node]

    # Recalculate leverage and energy balance for the remaining nodes.
    U_in_sum_list, U_out_sum_list = [], []
    leverage = []
    for node in G.nodes:
        U_in_sum, U_out_sum, ci = measure_system(node)
        leverage.append(ci)
        U_in_sum_list.append(U_in_sum)
        U_out_sum_list.append(U_out_sum)

    return U_in_sum_list, U_out_sum_list, leverage, pos



# Now, we can initialize the network and simulate its evolution.
G = initialize_network(100)
np.random.seed(42)

cth = .5  # Set a threshold for bankruptcy.
pos = nx.random_layout(G)


#print(pos) 

fig, axs = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [4, 1]})

def update(num, pos_0=pos):
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
        U_in, U_out, leverage, pos = update_agents(G, cth, pos_0)
    else:
        U_in, U_out, leverage, pos = update_agents(G, cth, pos)

    for node in G.nodes:
        if node not in pos:
            pos[node] = ((xmax - xmin) * random.random() + xmin, (ymax - ymin) * random.random() + ymin)



    
    U_in, U_out, leverage = np.array(U_in), np.array(U_out), np.array(leverage)

    UT = U_in + U_out
    #energy = U_T
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.jet
    sm = ScalarMappable(norm=norm, cmap=cmap)

    clusters = nx.algorithms.community.greedy_modularity_communities(G)

    # Calculate cluster centroids
    centroids = {i: np.mean([pos[node] for node in cluster], axis=0) for i, cluster in enumerate(clusters)}
    # Move nodes towards their cluster centroid
    for i, cluster in enumerate(clusters):
        for node in cluster:
            distance_to_centroid = np.linalg.norm(np.array(pos[node]) - np.array(centroids[i]))
            if distance_to_centroid > 0.25:
                pos[node] = tuple(0.99 * np.array(pos[node]) + 0.01 * np.array(centroids[i]))


    # Move nodes towards their cluster centroid
    #for i, cluster in enumerate(clusters):
    #    for node in cluster:
    #            pos[node] = 0.8 * np.array(pos[node]) + 0.2 * np.array(centroids[i])
    #        #pos[node] = 0.8 * pos[node] + 0.2 * tuple(centroids[i])

    #lognorm_energy = np.log1p(np.abs(U_T))
    #lognorm_bins = np.logspace(np.log10(np.min(energy)), np.log10(np.max(energy)), 10)

    # Normalize the energy values to be between 0 and 1 for coloring
    #norm_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    

    nx.draw_networkx_nodes(G, pos, node_color=leverage, node_size=leverage * 50, cmap=plt.cm.jet, ax=axs[0])
    for node in G.nodes:
        outgoing_edges = G.out_edges(node)
        incoming_edges = G.in_edges(node)
    
        nx.draw_networkx_edges(G, pos, edgelist=outgoing_edges, edge_color='green', alpha=0.4, arrowsize=2, arrowstyle='->', ax=axs[0])
        nx.draw_networkx_edges(G, pos, edgelist=incoming_edges, edge_color='red', alpha=0.4, arrowsize=2, arrowstyle='->', ax=axs[0])
    avg_degree = np.mean(leverage)
    number_of_agents = len(G.nodes)
    #axs[0].set_title(f"Step {num} - Avg. leverage: {avg_degree:.2f} - U_T: {UT:.2f} - N agents: {number_of_agents}")
    
    try:
        axs[1].hist(leverage, bins=10, color='blue', alpha=0.5)
    except:
        pass
    #axs[1].set_xscale('log')
    axs[1].set_title('Distribution of U_T')
    axs[1].set_xlabel('U_T')
    axs[1].set_ylabel('Frequency')
    #axs[1].set_yscale('log')

    energy = U_in + U_out

    if num == 0:
        print('Step Energy Leverage UT N_agents')
        print(f'{num} | {np.mean(energy):.2f}±{np.std(energy):.2f} | {np.mean(leverage):.2f}±{np.std(leverage):.2f} | {sum(UT):.2f} | {len(G.nodes)}')

    else:
    
        print(f'{num} | {sum(U_in):.2f} | {sum(U_out):.2f} | {sum(UT):.2f} | {len(G.nodes)} | {np.mean(leverage)}')

        #print(f'{num} | {np.mean(energy):.2f}±{np.std(energy):.2f} | {np.mean(leverage):.2f}±{np.std(leverage):.2f} | {sum(UT):.2f} | {len(G.nodes)}')
    return pos


ani = FuncAnimation(fig, update, frames=range(50), repeat=False)
ani.save("finance_graph.gif", writer='ffmpeg', fps=2, dpi=300)
