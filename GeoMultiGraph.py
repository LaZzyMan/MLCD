import numpy as np
import geopandas as gpd
from community import best_partition, modularity
import seaborn as sns
import networkx as nx
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from pywebplot import *
from palettable.colorbrewer.diverging import Spectral_10

NETWORK_LIST = ['2012',
                '2013',
                '2014',
                '2015',
                '2016',
                '2017']


class GeoMultiGraph:
    def __init__(self, geo_mapping=None, graph=None):
        self._geo_mapping = geo_mapping
        self._root_graph = graph
        self._graph = graph
        if graph is None:
            self._num_nodes = 0
            self._num_graph = 0
        else:
            self._num_nodes = len(graph[0])
            self._num_graph = len(graph)
        self._nx_graph = None

    def save(self, file_name):
        self._geo_mapping.to_file('src/data/' + file_name + '.geojson', driver='GeoJSON')
        np.save('src/data/' + file_name + '.npy', self._graph)

    def load(self, file_name):
        self._graph = np.load('src/data/' + file_name + '.npy')
        self._geo_mapping = gpd.read_file('src/data/' + file_name + '.geojson')
        self._num_nodes = len(self._graph[0])
        self._num_graph = len(self._graph)
        self.__update_nx_graph()

    @property
    def nx_graph(self):
        return self._nx_graph

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_graph(self):
        return self._num_graph

    @property
    def edges(self):
        for g, index in zip(self.nx_graph, range(self.num_graph)):
            connect_table = {'from_tazid': [],
                             'to_taziid': [],
                             'weight': [],
                             'network': []}

            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    try:
                        weight = g.adj[i][j]['weight']
                        if weight == 0:
                            continue
                        connect_table['network'].append(NETWORK_LIST[index])
                        connect_table['weight'].append(weight)
                        from_tazid = g.nodes[i]['tazid']
                        to_tazid = g.nodes[j]['tazid']
                        connect_table['to_taziid'].append(to_tazid)
                        connect_table['from_tazid'].append(from_tazid)
                    except KeyError as _:
                        continue
            connect_df = pd.DataFrame.from_dict(connect_table)
            return connect_df

    @property
    def edges_geo(self):
        for g, index in zip(self.nx_graph, range(self.num_graph)):
            connect_table = {'from_tazid': [],
                             'to_taziid': [],
                             'weight': [],
                             'network': [],
                             'geometry': []}
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    try:
                        weight = g.adj[i][j]['weight']
                        if weight == 0:
                            continue
                        connect_table['network'].append(NETWORK_LIST[index])
                        connect_table['weight'].append(weight)
                        from_tazid = g.nodes[i]['tazid']
                        to_tazid = g.nodes[j]['tazid']
                        connect_table['to_taziid'].append(to_tazid)
                        connect_table['from_tazid'].append(from_tazid)
                        from_point = self._geo_mapping[self._geo_mapping.tazid == from_tazid].geometry[i].centroid
                        to_point = self._geo_mapping[self._geo_mapping.tazid == to_tazid].geometry[j].centroid
                        connect_table['geometry'].append(LineString([from_point, to_point]))
                    except KeyError as _:
                        continue
            connect_df = gpd.GeoDataFrame.from_dict(connect_table)
            return connect_df

    def threshold(self, t_min=0, t_max=1000000):
        '''
        weight bigger than y_max will be set to t_max, lower than t_min will be set to 0.
        :param t_min:
        :param t_max:
        :return:
        '''
        def process(x, low, up):
            if x < low:
                return 0
            if x > up:
                return up
            return x

        expand = self._graph.reshape((self.num_nodes * self.num_nodes * self.num_graph, ))
        expand_t = np.array([process(i, t_min, t_max) for i in expand[0]])
        self._graph = expand_t.reshape((self.num_graph, self.num_nodes, self.num_nodes))
        self.__update_nx_graph()

    def community_detection_louvain(self, resolution=1.):
        table = {
            'tazid': [],
            'community': []
        }
        df_partiton = []
        for g in self.nx_graph:
            g = nx.Graph(g)
            p = best_partition(g, weight='weight', resolution=resolution)
            print('Network %s Modularity: %f.' % (g.graph['date'], modularity(p, g, weight='weight')))
            for key, item in p.items():
                table['tazid'].append(self.__get_tazid(key))
                table['community'].append(item)
            df_partiton.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['community'].clear()
        return df_partiton

    @property
    def closeness_centrality(self):
        table = {'tazid': [],
                 'closeness_centrality': []}
        closeness_centrality = []
        for g in self.nx_graph:
            cc = nx.closeness_centrality(g)
            for k, i in cc.items():
                table['tazid'].append(g.nodes[k]['tazid'])
                table['closeness_centrality'].append(i)
            closeness_centrality.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['closeness_centrality'].clear()
        return closeness_centrality

    @property
    def degree(self):
        table = {'tazid': [],
                 'in_degree': []}
        degree = []
        for g in self.nx_graph:
            for k in range(self.num_nodes):
                table['tazid'].append(g.nodes[k]['tazid'])
                table['degree'].append(g.degree(k, weight='weight'))
            degree.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['degree'].clear()
        return degree

    @property
    def in_degree(self):
        table = {'tazid': [],
                 'in_degree': []}
        in_degree = []
        for g in self.nx_graph:
            for k in range(self.num_nodes):
                table['tazid'].append(g.nodes[k]['tazid'])
                table['in_degree'].append(g.in_degree(k, weight='weight'))
            in_degree.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['in_degree'].clear()
        return in_degree

    @property
    def out_degree(self):
        table = {'tazid': [],
                'out_degree': []}
        out_degree = []
        for g in self.nx_graph:
            for k in range(self.num_nodes):
                table['tazid'].append(g.nodes[k]['tazid'])
                table['out_degree'].append(g.out_degree(k, weight='weight'))
            out_degree.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['out_degree'].clear()
        return out_degree

    def draw_dist(self, kde=True, rug=True):
        sns.set_style('ticks')
        graphs = sns.FacetGrid(self.edges, col='network')
        graphs.map(plt.hist, 'weight')

    def draw_multi_scale_community(self, map_view, community, title='community', cmap=Spectral_10):
        community = community[0]
        community_num = community['community'].max()

        def set_color(x):
            mpl_colormap = cmap.get_mpl_colormap(N=community_num)
            rgba = mpl_colormap(x.community)
            return rgb2hex(rgba[0], rgba[1], rgba[2])
        community_geo_map = self._geo_mapping.merge(community, on='tazid')
        community_geo_map = community_geo_map[['tazid', 'community', 'geometry']]
        community_geo_map['color'] = community_geo_map.apply(set_color)
        community_geo_map.to_file('src/data/%s.geojson' % title, driver='GeoJSON')
        source = GeojsonSource(id=title, data='data/%s.geojson')
        map_view.add_source(source)
        layer = FillLayer(id=title, source=title, p_fill_opacity=0.7, p_fill_color=['get', 'color'])
        map_view.add_layer(layer)
        map_view.update()

    def draw_choropleth_map(self, map_view, data, value='', title='Choropleth Map', cmap=Spectral_10):
        value_min = data[value].min()
        value_max = data[value].max()

        def set_color(x):
            mpl_colormap = cmap.get_mpl_colormap(N=value_max-value_min)
            rgba = mpl_colormap(x[value] + value_min)
            return rgb2hex(rgba[0], rgba[1], rgba[2])
        value_geo_map = self._geo_mapping.merge(data, on=value)
        value_geo_map = value_geo_map[['tazid', value, 'geometry']]
        value_geo_map['color'] = value_geo_map.apply(set_color)
        value_geo_map.to_file('src/data/%s.geojson' % title, driver='GeoJSON')
        source = GeojsonSource(id=value, data='data/%s.geojson')
        map_view.add_source(source)
        layer = FillLayer(id=value, source=value, p_fill_opacity=0.7, p_fill_color=['get', 'color'])
        map_view.add_layer(layer)
        map_view.update()

    def draw_single_network(self, map_view, network=NETWORK_LIST[0], color='white', width=1., value='weight', title='network', bk=True):
        connect_df = self.edges_geo[self.edges_geo['network'] == network]
        min_weight = connect_df['weight'].min()
        max_weight = connect_df['weight'].max()
        connect_df['opacity'] = connect_df['weight'].map(
            lambda x: (x - min_weight) / (max_weight - min_weight) * 0.8 + 0.00)
        draw_df = connect_df[['geometry', 'opacity']]
        draw_df.to_file('src/data/%s.geojson' % title, driver='GeoJSON')
        network_source = GeojsonSource(id=value, data='data/%s.geojson' % title)
        map_view.add_source(network_source)
        bk_layer = BackgroundLayer(id='bk',
                                   p_background_opacity=0.7,
                                   p_background_color='#000000')
        network_layer = LineLayer(id=title,
                                  source=title,
                                  p_line_color=color,
                                  p_line_width=width,
                                  p_line_opacity=['get', 'opacity'])
        if bk:
            map_view.add_layer(bk_layer)
        map_view.add_layer(network_layer)
        map_view.update()

    def draw_taz(self, map_view, color='#3BA1C3', fill_opacity=0.3):
        taz_unit_df = self._geo_mapping['geometry']
        taz_unit_df.to_file('src/data/taz.geojson', driver='GeoJSON')
        source = GeojsonSource(id='taz', data='data/taz.geojson')
        map_view.add_source(source)
        layer = FillLayer(id='taz', source='taz', p_fill_opacity=fill_opacity, p_fill_color=color)
        map_view.add_layer(layer)
        map_view.update()

    def __get_tazid(self, index):
        return self._geo_mapping['tazid'][index]

    def __update_nx_graph(self):
        G = []
        for l, time in zip(self._graph, NETWORK_LIST):
            print('Generating Network %s' % time)
            g = nx.DiGraph(date=time)
            for i in range(len(l)):
                g.add_node(i, tazid=self.__get_tazid(i))
            for i in range(len(l)):
                for j in range(len(l)):
                    if i == j:
                        continue
                    g.add_edge(i, j, weight=l[i, j])
            G.append(g)
        return G


if __name__ == '__main__':
    gmg = GeoMultiGraph()
    gmg.load('GeoMultiGraph_week')
    cc = gmg.closeness_centrality
    in_degree = gmg.in_degree
    out_degree = gmg.out_degree
    degree = gmg.degree
    cl = gmg.community_detection_louvain()
    plt = PlotView(column_num=2, row_num=2, title='Geo-Multi-Graph')
    plt[0, 0].name = 'Closeness-Centrality'
    plt[1, 0].name = 'in-degree'
    plt[1, 1].name = 'out-degree'
    plt[0, 1].name = 'degree'
    mb_0 = MapBox(name='map-0',
                  pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                  lon=116.37363,
                  lat=39.915606,
                  style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                  pitch=55,
                  bearing=0,
                  zoom=12,
                  viewport=plt[0, 0])
    mb_1 = MapBox(name='map-1',
                  pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                  lon=116.37363,
                  lat=39.915606,
                  style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                  pitch=55,
                  bearing=0,
                  zoom=12,
                  viewport=plt[1, 0])
    mb_2 = MapBox(name='map-2',
                  pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                  lon=116.37363,
                  lat=39.915606,
                  style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                  pitch=55,
                  bearing=0,
                  zoom=12,
                  viewport=plt[1, 1])
    mb_3 = MapBox(name='map-3',
                  pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                  lon=116.37363,
                  lat=39.915606,
                  style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                  pitch=55,
                  bearing=0,
                  zoom=12,
                  viewport=plt[0, 1])
    gmg.draw_choropleth_map(map_view=mb_0, data=cc[0], value='closeness_centrality', title='cc')
    gmg.draw_choropleth_map(map_view=mb_1, data=in_degree[0], value='in_degree', title='in')
    gmg.draw_choropleth_map(map_view=mb_2, data=out_degree[0], value='out_degree', title='out')
    gmg.draw_single_network(map_view=mb_3, network='2012')
