from GeoMultiGraph import GeoMultiGraph
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community


def nx_community_to_pd(graph, nx_community):
    table = {'tazid': [],
             'community': [],
             'size': []}
    cid = 0
    for c in nx_community:
        for i in list(c):
            table['tazid'].append(graph.nodes[i]['tazid'])
            table['community'].append(cid)
            table['size'].append(len(c))
        cid += 1
    return pd.DataFrame.from_dict(table)


def nx_centrality_to_pd(graph, nx_centrality):
    table = {'tazid': [],
             'centrality': []}
    for k, i in nx_centrality.items():
        table['tazid'].append(graph.nodes[k]['tazid'])
        table['centrality'].append(i)
    return pd.DataFrame.from_dict(table)


if __name__ == '__main__':
    gmg = GeoMultiGraph()
    gmg.load('GeoMultiGraph_week')
    G = gmg.to_nx_graph(min_weight=5, max_weight=100)
    # gmg.mapbox_draw_background(color='black', opacity=0.9)
    # gmg.mapbox_draw_taz_unit()
    # for i in range(len(G)):
    #     gmg.mapbox_draw_single_network(width=0.7, index=i)
    # gmg.mapbox_show()
    # in_degree = pd.DataFrame.from_dict({'tazid': [G[0].nodes[i]['tazid'] for i in range(1371)],
    #                                     'in_degree': [G[0].in_degree(i, weight='weight') for i in range(1371)]})
    # gmg.draw_map(value_name='in_degree', cmap='summer', data=in_degree)
    # out_degree = pd.DataFrame.from_dict({'tazid': [G[0].nodes[i]['tazid'] for i in range(1371)],
    #                                     'out_degree': [G[0].out_degree(i, weight='weight') for i in range(1371)]})
    # gmg.draw_map(value_name='out_degree', cmap='summer', data=out_degree)
    # degree = pd.DataFrame.from_dict({'tazid': [G[0].nodes[i]['tazid'] for i in range(1371)],
    #                                 'degree': [G[0].degree(i, weight='weight') for i in range(1371)]})
    # gmg.draw_map(value_name='degree', cmap='summer', data=degree)
    # centrality = nx.closeness_centrality(G[4])
    # gmg.draw_map_centrality(nx_centrality_to_pd(G[4], centrality))
    m = community.greedy_modularity_communities(nx.Graph(G[0]), weight='weight')
    m_uw = community.greedy_modularity_communities(nx.Graph(G[0]))
    gmg.draw_map_community(nx_community_to_pd(G[0], m), title='week_2012.png')
    gmg.draw_map_community(nx_community_to_pd(G[0], m_uw), title='week_2012_uw.png')
