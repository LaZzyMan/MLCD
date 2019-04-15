from community import best_partition, modularity
import seaborn as sns
import time
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from pywebplot import *
from palettable.colorbrewer.diverging import Spectral_10
from scipy import stats
import powerlaw
import infomap
import tensorly as tl


def get_closeness_centrality(g):
    table = {'tazid': [],
             'closeness_centrality': []}
    cc = nx.closeness_centrality(g)
    for k, i in cc.items():
        table['tazid'].append(g.nodes[k]['tazid'])
        table['closeness_centrality'].append(i)
    return pd.DataFrame.from_dict(table)


def get_degree_centrality(g):
    table = {'tazid': [],
             'degree_centrality': []}
    cc = nx.degree_centrality(g)
    for k, i in cc.items():
        table['tazid'].append(g.nodes[k]['tazid'])
        table['degree_centrality'].append(i)
    return pd.DataFrame.from_dict(table)


def get_eigenvector_centrality(g):
    table = {'tazid': [],
             'eigenvector_centrality': []}
    cc = nx.eigenvector_centrality(g)
    for k, i in cc.items():
        table['tazid'].append(g.nodes[k]['tazid'])
        table['eigenvector_centrality'].append(i)
    return pd.DataFrame.from_dict(table)


def get_cluster_coefficient(g):
    table = {'tazid': [],
             'cluster_coefficient': []}
    cc = nx.clustering(g)
    for k, i in cc.items():
        table['tazid'].append(g.nodes[k]['tazid'])
        table['cluster_coefficient'].append(i)
    return pd.DataFrame.from_dict(table)


class GeoMultiGraph:
    def __init__(self, geo_mapping=None, graph=None, network_list=None):
        self._geo_mapping = geo_mapping
        self._root_graph = graph
        self._graph = graph
        self._network_list = network_list
        if graph is None:
            self._num_nodes = 0
            self._num_graph = 0
        else:
            self._num_nodes = len(graph[0])
            self._num_graph = len(graph)
        self._nx_graph = None
        mkdir()

    def save(self, file_name):
        self._geo_mapping.to_file(file_name + '.geojson', driver='GeoJSON')
        np.save(file_name + '.npy', self._graph)

    def load(self, file_name, generate_nx=False, network_list=None):
        self._graph = np.load(file_name + '.npy')
        self._geo_mapping = gpd.read_file(file_name + '.geojson')
        self._num_nodes = len(self._graph[0])
        self._num_graph = len(self._graph)
        if network_list is None:
            self._network_list = ['Network-%d' % i for i in range(self._num_graph)]
        else:
            self._network_list = network_list
        if generate_nx:
            self.__update_nx_graph()

    def recovery(self):
        '''
        recovery the origin data
        :return:
        '''
        self._graph = self._root_graph
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
        print('Generating Edges Data Frame...')
        connect_table = {'from_tazid': [],
                         'to_taziid': [],
                         'weight': [],
                         'network': []}
        for g, index in zip(self.nx_graph, range(self.num_graph)):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    try:
                        weight = g.adj[i][j]['weight']
                        if weight == 0:
                            continue
                        connect_table['network'].append(self._network_list[index])
                        connect_table['weight'].append(weight)
                        from_tazid = g.nodes[i]['tazid']
                        to_tazid = g.nodes[j]['tazid']
                        connect_table['to_taziid'].append(to_tazid)
                        connect_table['from_tazid'].append(from_tazid)
                    except KeyError as _:
                        continue
        connect_df = pd.DataFrame.from_dict(connect_table)
        print('Finished.')
        return connect_df

    @property
    def edges_geo(self):
        print('Generating Geo Edges Data Frame...')
        connect_table = {'from_tazid': [],
                         'to_taziid': [],
                         'weight': [],
                         'network': [],
                         'geometry': []}
        for g, index in zip(self.nx_graph, range(self.num_graph)):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    try:
                        weight = g.adj[i][j]['weight']
                        if weight == 0:
                            continue
                        connect_table['network'].append(self._network_list[index])
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
        print('Finished.')
        return connect_df

    @property
    def closeness_centrality(self):
        return [get_closeness_centrality(g) for g in self.nx_graph]

    @property
    def cluster_coefficient(self):
        return [get_cluster_coefficient(g) for g in self.nx_graph]

    @property
    def degree_centrality(self):
        return [get_degree_centrality(g) for g in self.nx_graph]

    @property
    def eigenvector_centrality(self):
        return [get_eigenvector_centrality(g) for g in self.nx_graph]

    @property
    def degree(self):
        table = {'tazid': [],
                 'in_degree': []}
        degree = []
        for g in self.nx_graph:
            print('Degree for %s...' % g.graph['date'])
            for k in range(self.num_nodes):
                table['tazid'].append(g.nodes[k]['tazid'])
                table['degree'].append(g.degree(k, weight='weight'))
            degree.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['degree'].clear()
        print('Finished.')
        return degree

    @property
    def in_degree(self):
        table = {'tazid': [],
                 'in_degree': []}
        in_degree = []
        for g in self.nx_graph:
            print('In Degree for %s...' % g.graph['date'])
            for k in range(self.num_nodes):
                table['tazid'].append(g.nodes[k]['tazid'])
                table['in_degree'].append(g.in_degree(k, weight='weight'))
            in_degree.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['in_degree'].clear()
        print('Finished.')
        return in_degree

    @property
    def out_degree(self):
        table = {'tazid': [],
                 'out_degree': []}
        out_degree = []
        for g in self.nx_graph:
            print('Out Degree for %s...' % g.graph['date'])
            for k in range(self.num_nodes):
                table['tazid'].append(g.nodes[k]['tazid'])
                table['out_degree'].append(g.out_degree(k, weight='weight'))
            out_degree.append(pd.DataFrame.from_dict(table))
            table['tazid'].clear()
            table['out_degree'].clear()
        print('Finished.')
        return out_degree

    def transform(self, func='log', generate_nx=False):
        '''
        call func for weight of the graph
        :param func:
        :param generate_nx:
        :return:
        '''
        def tf_ln(x):
            if x == 0:
                return 0
            else:
                return np.log(x)

        def tf_log2(x):
            if x == 0:
                return 0
            else:
                return np.log2(x)

        def tf_log10(x):
            if x == 0:
                return 0
            else:
                return np.log10(x)

        def tf_sqrt(x):
            return np.sqrt(x)
        func_dict = {
            'log': tf_ln,
            'log2': tf_log2,
            'log10': tf_log10,
            'sqrt': tf_sqrt
        }
        if func in func_dict.keys():
            self._graph = np.vectorize(func_dict[func])(self._graph)
        if func == 'cox':
            expand = [stats.boxcox(g.reshape((self.num_nodes * self.num_nodes, )) + 0.00001)[0] for g in self._graph]
            self._graph = np.array([g.reshape(self.num_nodes, self.num_nodes) for g in expand])
        if generate_nx:
            self.__update_nx_graph()

    def threshold(self, t_min=0, t_max=1000000, generate_nx=False):
        '''
        weight bigger than y_max will be set to t_max, lower than t_min will be set to 0.
        :param t_min:
        :param t_max:
        :param generate_nx:
        :return:
        '''
        def process(x):
            if x < t_min:
                return 0
            if x > t_max:
                return t_max
            return x

        self._graph = np.vectorize(process)(self._graph)
        if generate_nx:
            self.__update_nx_graph()

    def generate_tensor(self):
        tensor = tl.tensor(np.zeros([self.num_nodes, self.num_nodes, self.num_graph, self.num_graph], dtype=np.float64))
        for node_0 in range(self.num_nodes):
            for node_1 in range(self.num_nodes):
                for layer_0 in range(self.num_graph):
                    for layer_1 in range(self.num_graph):
                        if layer_0 == layer_1:
                            if node_0 == node_1:
                                continue
                            else:
                                tensor[node_0][node_1][layer_0][layer_1] = self._graph[layer_0][node_0][node_1]
                        else:
                            time_bin = abs(layer_0 - layer_1)
                            geo_bin = 0
        return tensor

    def community_detection_louvain(self, resolution=1., min_size=10):
        def louvain(g):
            table = {
                'tazid': [],
                'community': []
            }
            print('Louvain for network %s...' % g.graph['date'])
            g = nx.Graph(g)
            p = best_partition(g, weight='weight', resolution=resolution)
            print('Network %s Modularity: %f.' % (g.graph['date'], modularity(p, g, weight='weight')))
            for key, item in p.items():
                table['tazid'].append(self.__get_tazid(key))
                table['community'].append(item)
            community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(table), size=min_size)
            print('Finished %s, %d communities found, %d point unclassified.'
                  % (g.graph['date'], num_community, num_unclassify))
            return community
        df_partition = map(louvain, self.nx_graph)
        return list(df_partition)

    def community_detection_infomap(self, min_size=10):
        communities = []
        for g in self.nx_graph:
            table = {
                'tazid': [],
                'community': []
            }
            infomap_wrapper = infomap.Infomap('--two-level --directed')
            network = infomap_wrapper.network()
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    try:
                        weight = g.adj[i][j]['weight']
                        if weight == 0:
                            continue
                        network.addLink(i, j, float(weight))
                    except KeyError:
                        continue
            infomap_wrapper.run()
            print("Found %d top modules with codelength: %f" %
                  (infomap_wrapper.numTopModules(), infomap_wrapper.codelength()))
            for node in infomap_wrapper.iterTree():
                if node.isLeaf():
                    table['tazid'].append(self.__get_tazid(node.physicalId))
                    table['community'].append(node.moduleIndex())
            community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(table), size=min_size)
            print('Finished %s, %d communities found, %d point unclassified.'
                  % (g.graph['date'], num_community, num_unclassify))
            communities.append(community)
        return communities

    def community_detection_multi_infomap(self):
        pass

    def draw_dist(self, hist=True, kde=True, rug=True, bins=10):
        sns.set_style('ticks')
        sns.set(color_codes=True)
        graphs = sns.FacetGrid(self.edges, col='network')
        graphs.map(sns.distplot, 'weight', hist=hist, kde=kde, rug=rug, bins=bins)
        plt.show()

    def draw_cdf(self):
        def cdf(x, **kwargs):
            powerlaw.plot_cdf(x)

        sns.set_style('ticks')
        sns.set(color_codes=True)
        graphs = sns.FacetGrid(self.edges, col='network')
        graphs.map(cdf, 'weight')
        plt.show()

    def draw_ccdf(self):
        def ccdf(x, **kwargs):
            powerlaw.plot_ccdf(x)

        sns.set_style('ticks')
        sns.set(color_codes=True)
        graphs = sns.FacetGrid(self.edges, col='network')
        graphs.map(ccdf, 'weight')
        plt.show()

    def draw_qq_plot(self):
        def qqplot(x, **kwargs):
            stats.probplot(x, dist='norm', plot=plt, fit=False)

        sns.set_style('ticks')
        sns.set(color_codes=True)
        graphs = sns.FacetGrid(self.edges, col='network')
        graphs.map(qqplot, 'weight')
        plt.show()

    def draw_community_dist(self, community, hist=True, kde=False, rug=False, bins=10):
        for i, c in zip(self._network_list, community):
            c['network'] = i
        c_df = pd.concat(community, axis=0)
        sns.set_style('ticks')
        sns.set(color_codes=True)
        graphs = sns.FacetGrid(c_df, col='network')
        graphs.map(sns.distplot, 'community', hist=hist, kde=kde, rug=rug, bins=bins)
        plt.show()

    def draw_choropleth_map(self, map_view, data, value='', title='Choropleth Map', cmap=Spectral_10):
        timestamp = int(time.time())
        value_min = data[value].min()
        value_max = data[value].max()
        mpl_colormap = cmap.get_mpl_colormap(N=value_max - value_min)

        def set_color(x):
            rgba = mpl_colormap(x[value] + value_min)
            return rgb2hex(rgba[0], rgba[1], rgba[2])
        value_geo_map = self._geo_mapping.merge(data, on='tazid')
        value_geo_map = value_geo_map[['tazid', value, 'geometry']]
        value_geo_map['color'] = value_geo_map.apply(set_color, axis=1)
        value_geo_map.to_file('dist/data/%s.geojson' % (title + str(timestamp)), driver='GeoJSON')
        source = GeojsonSource(id=value, data='%s.geojson' % (title + str(timestamp)))
        map_view.add_source(source)
        layer = FillLayer(id=value, source=value, p_fill_opacity=0.7, p_fill_color=['get', 'color'])
        map_view.add_layer(layer)
        map_view.update()

    def draw_multi_scale_community(self, community, cmap=Spectral_10, column=2, row=3, inline=False, title='Geo-Multi-Graph'):
        view = PlotView(column_num=row, row_num=column, title=title)
        for subview, i in zip(view, range(self.num_graph)):
            subview.name = self._network_list[i]
        maps = [MapBox(name='map_%d' % i,
                       pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                       lon=116.37363,
                       lat=39.915606,
                       style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                       pitch=55,
                       bearing=0,
                       zoom=12,
                       viewport=view[i]) for i in range(self.num_graph)]
        for i in range(self.num_graph):
            self.draw_choropleth_map(map_view=maps[i], data=community[i], value='community', title='%scommunity' % self._network_list[i], cmap=cmap)
        view.plot(inline=inline)

    def draw_network(self, color='white', width=1., value='weight', bk=True, inline=False, row=3, column=2):
        view = PlotView(column_num=row, row_num=column, title='Network')
        for subview, i in zip(view, range(self.num_graph)):
            subview.name = self._network_list[i]
        maps = [MapBox(name='map_network_%d' % i,
                       pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                       lon=116.37363,
                       lat=39.915606,
                       style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                       pitch=55,
                       bearing=0,
                       zoom=12,
                       viewport=view[i]) for i in range(self.num_graph)]
        edge_df = self.edges_geo
        for i in range(self.num_graph):
            self.draw_single_network(map_view=maps[i],
                                     data=edge_df[edge_df['network'] == self._network_list[i]].copy(),
                                     color=color,
                                     width=width,
                                     value=value,
                                     title='Network-%s' % self._network_list[i],
                                     bk=bk)
        view.plot(inline=inline)

    def draw_single_network(self, map_view, data, color='white', width=1., value='weight', title='network', bk=True):
        timestamp = int(time.time())
        print('Drawing %s...' % title)
        min_weight = data[value].min()
        max_weight = data[value].max()
        data['opacity'] = data[value].map(
            lambda x: (x - min_weight) / (max_weight - min_weight) * 0.8 + 0.00)
        draw_df = data[['geometry', 'opacity', value]]
        draw_df.to_file('dist/data/%s.geojson' % (title + str(timestamp)), driver='GeoJSON')
        network_source = GeojsonSource(id=title, data='%s.geojson' % (title + str(timestamp)))
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
        print('Finished.')

    def draw_taz(self, map_view, color='#3BA1C3', fill_opacity=0.3):
        taz_unit_df = self._geo_mapping['geometry']
        taz_unit_df.to_file('dist/data/taz.geojson', driver='GeoJSON')
        source = GeojsonSource(id='taz', data='taz.geojson')
        map_view.add_source(source)
        layer = FillLayer(id='taz', source='taz', p_fill_opacity=fill_opacity, p_fill_color=color)
        map_view.add_layer(layer)
        map_view.update()

    def __get_tazid(self, index):
        return self._geo_mapping['tazid'][index]

    def __update_nx_graph(self):
        G = []
        for l, time in zip(self._graph, self._network_list):
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
        self._nx_graph = G

    @staticmethod
    def __simplify_community(community, size=10):
        c_list = {}
        r_list = []
        un_list = []
        for _, line in community.iterrows():
            if line['community'] in c_list.keys():
                c_list[line['community']].append(line['tazid'])
            else:
                c_list[line['community']] = [line['tazid']]
        for key, item in c_list.items():
            if len(item) <= size:
                for i in item:
                    un_list.append(i)
            else:
                r_list.append(item)
        r_list.append(un_list)
        table = {
            'tazid': [],
            'community': []
        }
        for r, i in zip(r_list, range(len(r_list))):
            for k in r:
                table['tazid'].append(k)
                table['community'].append(i)
        return pd.DataFrame.from_dict(table), len(r_list) - 1, len(un_list)

    def __cal_topology_distance(self):
        pass

    def __cal_2d_distance(self):
        pass


if __name__ == '__main__':
    gmg = GeoMultiGraph()
    gmg.load('GeoMultiGraph_week', generate_nx=True, network_list=['2012', '2013', '2014', '2015', '2016', '2017'])
    # gmg.threshold(t_min=3, generate_nx=True)
    # gmg.transform(func='sqrt', generate_nx=True)
    # gmg.draw_dist(hist=True, kde=False, rug=False, bins=20)
    # gmg.draw_qq_plot()
    cl = gmg.community_detection_louvain(min_size=20, resolution=2.)
    gmg.draw_multi_scale_community(community=cl, cmap=Spectral_10)
