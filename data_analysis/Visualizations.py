from common_helpers import *
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_percent_gesture_cluster_overlap(cluster_A, cluster_B):
    setA = set(cluster_A['gesture_ids'])
    setB = set(cluster_B['gesture_ids'])
    overlap = float(len(setA & setB)) / len(setA)
    return overlap


def get_all_clusters_for_gesture_ids(g_ids, clusters):
    ids = []
    for c in clusters.keys():
        for gid in clusters[c]['gesture_ids']:
            if gid in g_ids:
                ids.append(c)
    return list(set(ids))


def get_cluster_sizes(clusters, appender='R'):
    ids = []
    lens = []
    for c in clusters.keys():
        ids.append(str(c)+appender)      # TODO undo this hacky R bit
        s = len(clusters[c]['gesture_ids'])
        lens.append(s)
    return {'ids': ids, 'sizes': lens}


def create_network_from_two_clusterings(A, B):
    # create df for network diagram
    from_ = []
    to_ = []
    colors_ = []
    weights_ = []
    for k in A.keys():
        B_clusters = get_all_clusters_for_gesture_ids(A[k]['gesture_ids'], B)
        for b in B_clusters:
            from_.append(str(k)+'R')
            to_.append(str(b)+'M')
            weights_.append(get_percent_gesture_cluster_overlap(A[k], B[b]))

    df = pd.DataFrame({'from': from_, 'to': to_, 'weights': weights_})
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    node_sizes = pd.DataFrame(get_cluster_sizes(A))
    node_sizes = node_sizes.set_index('ids')
    node_sizes = node_sizes.reindex(G.nodes())
    print(node_sizes['sizes'])

    nx.draw(G, with_labels=True, edge_color=df['weights'], node_size=node_sizes['sizes'])
    plt.show()


def create_network_from_two_clusterings(A, B):
    # create df for network diagram
    G = nx.Graph()
    for k in A.keys():
        B_clusters = get_all_clusters_for_gesture_ids(A[k]['gesture_ids'], B)
        for b in B_clusters:
            G.add_edge(str(k)+'R', str(b)+'M', weight=get_percent_gesture_cluster_overlap(A[k], B[b]))

    node_sizes = pd.DataFrame(get_cluster_sizes(A, 'R'))
    m_node_sizes = pd.DataFrame(get_cluster_sizes(B, 'M'))

    node_sizes = node_sizes.append(m_node_sizes)
    node_sizes = node_sizes.set_index('ids')
    node_sizes = node_sizes.reindex(G.nodes())

    node_colors = list(map(lambda x: 'r' if 'R' in x else 'b', G.nodes()))

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.2]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.2]

    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes['sizes'], node_color=node_colors, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=1, edge_color='g')
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                           width=1, alpha=0.5, edge_color='b')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.axis('off')
    plt.show()
