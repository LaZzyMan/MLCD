from community import best_partition, modularity
import seaborn as sns
import time
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
from pywebplot import *
from palettable.colorbrewer.diverging import Spectral_10
from palettable.colorbrewer.sequential import OrRd_9
from scipy import stats
import powerlaw
import infomap
import tensorly as tl
from pysal.lib import weights
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import subprocess
import copy


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


def merge_layers(mc):
    table = {
        'tazid': [],
        'community': []
    }
    for taz in mc[0]['tazid'].unique():
        table['tazid'].append(taz)
        m = np.argmax(np.bincount([int(c[c['tazid'] == taz]['community']) for c in mc]))
        table['community'].append(m)
    return pd.DataFrame.from_dict(table)


class GeoMultiGraph:
    def __init__(self, geo_mapping=None, graph=None, network_list=None, generate_nx=False, infomap_url='../../Infomap/Infomap '):
        self._geo_mapping = geo_mapping
        self._root_graph = graph
        self._graph = graph
        self._network_list = network_list
        self._infomap_url = infomap_url
        if graph is None:
            self._num_nodes = 0
            self._num_graph = 0
        else:
            self._num_nodes = len(graph[0])
            self._num_graph = len(graph)
        self._nx_graph = None
        mkdir()
        if generate_nx:
            self.__update_nx_graph()

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

    def import_map_file(self, filename, min_size=10):
        tag = ''
        community_table = {
            'tazid': [],
            'community': [],
            'flow': []
        }
        module_table = {
            'community': [],
            'flow': []
        }
        link_table = {
            'from': [],
            'to': [],
            'flow': []
        }
        code_length = 0.
        with open(filename, 'r') as f:
            for line in f.readlines():
                if '# codelength' in line:
                    code_length = float(line.split(' ')[2])
                    continue
                if '*Modules' in line:
                    tag = 'module'
                    continue
                if '*Link' in line:
                    tag = 'link'
                    continue
                if '*Node' in line:
                    tag = 'node'
                    continue
                if tag == 'module':
                    module = line.split(' ')[0]
                    flow = line.split(' ')[2]
                    module_table['community'].append(int(module))
                    module_table['flow'].append(float(flow))
                if tag == 'link':
                    link_table['from'].append(int(line.split(' ')[0]))
                    link_table['to'].append(int(line.split(' ')[1]))
                    link_table['flow'].append(float(line.split(' ')[2]))
                if tag == 'node':
                    node = eval(line.split(' ')[1])
                    tree = line.split(' ')[0]
                    flow = float(line.split(' ')[2])
                    community_table['tazid'].append(self.__get_tazid(int(node)))
                    community_table['community'].append(int(tree.split(':')[0]))
                    community_table['flow'].append(flow)
            f.close()
        community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(community_table), size=min_size)

        def norm_flow(x):
            f_min = community['flow'][community['community'] == x['community']].min()
            f_max = community['flow'][community['community'] == x['community']].max()
            return (x['flow'] - f_min) / (f_max - f_min)
        community['flow_norm'] = community.apply(norm_flow, axis=1)

        def get_community_centroid(x):
            tazs = community['tazid'][community['community'] == x['community']].unique()
            cen_x = 0.
            cen_y = 0.
            total_flow = 0.
            for taz in tazs:
                centroid = list(self._geo_mapping['geometry'][self._geo_mapping['tazid'] == taz])[0].centroid
                flow = float(community['flow_norm'][community['tazid'] == taz])
                cen_x += centroid.x * flow
                cen_y += centroid.y * flow
                total_flow += flow
            return Point(cen_x / total_flow, cen_y / total_flow)
        module = gpd.GeoDataFrame.from_dict(module_table)
        module['enable'] = module['community'].isin(community['community'].unique())
        module = gpd.GeoDataFrame(module[['community', 'flow']][module['enable']])
        module['geometry'] = module.apply(get_community_centroid, axis=1)
        module.set_geometry('geometry')
        link = gpd.GeoDataFrame.from_dict(link_table)

        def get_community_link(x):
            from_point = list(module['geometry'][module['community'] == x['from']])[0]
            to_point = list(module['geometry'][module['community'] == x['to']])[0]
            return LineString([from_point, to_point])
        link['enable'] = link['from'].isin(community['community'].unique()) & link['to'].isin(community['community'].unique())
        link = gpd.GeoDataFrame(link[['from', 'to', 'flow']][link['enable']])
        link['geometry'] = link.apply(get_community_link, axis=1)
        link.set_geometry('geometry')
        print('Finished. %d communities found, %d point unclassified, code length %f.'
              % (num_community, num_unclassify, code_length))
        return community, module, link

    def import_expanded_map_file(self, filename, min_size=1):
        tag = ''
        community_table = {
            'tazid': [],
            'community': [],
            'layer': [],
            'flow': []
        }
        module_table = {
            'community': [],
            'flow': []
        }
        link_table = {
            'from': [],
            'to': [],
            'flow': []
        }
        code_length = 0.
        with open(filename, 'r') as f:
            for line in f.readlines():
                if '# codelength' in line:
                    code_length = float(line.split(' ')[2])
                    continue
                if '*Modules' in line:
                    tag = 'module'
                    continue
                if '*Link' in line:
                    tag = 'link'
                    continue
                if '*Node' in line:
                    tag = 'node'
                    continue
                if tag == 'module':
                    module = line.split(' ')[0]
                    flow = line.split(' ')[4]
                    module_table['community'].append(int(module))
                    module_table['flow'].append(float(flow))
                if tag == 'link':
                    link_table['from'].append(int(line.split(' ')[0]))
                    link_table['to'].append(int(line.split(' ')[1]))
                    link_table['flow'].append(float(line.split(' ')[2]))
                if tag == 'node':
                    node = line.split(' ')[1][1:]
                    layer = line.split(' ')[3][:-1]
                    tree = line.split(' ')[0]
                    flow = float(line.split(' ')[4])
                    community_table['tazid'].append(self.__get_tazid(int(node)))
                    community_table['layer'].append(int(layer))
                    community_table['community'].append(int(tree.split(':')[0]))
                    community_table['flow'].append(flow)
            f.close()
        community = pd.DataFrame.from_dict(community_table)
        module = pd.DataFrame.from_dict(module_table)
        link = pd.DataFrame.from_dict(link_table)
        full_community = []
        for i in range(self.num_graph):
            c = community[community['layer'] == i].copy()
            exist_list = c['tazid'].tolist()
            un_list = {
                'tazid': [],
                'community': [],
                'layer': [],
                'flow': []
            }
            for taz in self._geo_mapping['tazid']:
                if taz not in exist_list:
                    un_list['tazid'].append(taz)
                    un_list['community'].append(-1)
                    un_list['layer'].append(i)
                    un_list['flow'].append(0.)
            full_community.append(pd.concat([c, pd.DataFrame.from_dict(un_list)], axis=0))
        community = [self.__simplify_community(c, size=min_size)[0] for c in full_community]
        community = pd.concat(community, axis=0)

        def norm_flow(x):
            f_min = community['flow'][(community['community'] == x['community']) & (community['layer'] == x['layer'])].min()
            f_max = community['flow'][(community['community'] == x['community']) & (community['layer'] == x['layer'])].max()
            return (x['flow'] - f_min) / (f_max - f_min)

        community['flow_norm'] = community.apply(norm_flow, axis=1)
        print('Finished. %d communities found, code length %f.'
              % (community['community'].max(), code_length))
        return [community[community['layer'] == i].copy() for i in range(self.num_graph)], module, link

    def export(self, type, folder, filename, split=True, geo_weight='kde', connect='flow'):
        if type == 'MultiTensor':
            with open('%s/%s' % (folder, filename), 'w') as f:
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if not i == j:
                            w_layers = ''
                            for layer in range(self.num_graph):
                                w_layers += str(self._graph[layer][i][j]) + ' '
                            f.write('E %d %d %s\n' % (i, j, w_layers))
                f.close()
            return
        if type == 'multi_infomap':
            tensor = self.generate_tensor(geo_weight=geo_weight, connect=connect)
            if split:
                with open('%s/%s' % (folder, filename), 'w') as f:
                    f.write('*Intra\n')
                    for layer in range(self.num_graph):
                        for node_0 in range(self.num_nodes):
                            for node_1 in range(self.num_nodes):
                                weight = tensor[node_0][node_1][layer][layer]
                                if not weight == 0:
                                    f.write('%d %d %d %f\n' % (layer, node_0, node_1, weight))
                    f.write('*Inter\n')
                    for layer_0 in range(self.num_graph):
                        for layer_1 in range(self.num_graph):
                            for node_0 in range(self.num_nodes):
                                for node_1 in range(self.num_nodes):
                                    if not layer_0 == layer_1:
                                        weight = tensor[node_0][node_1][layer_0][layer_1]
                                        if not weight == 0:
                                            f.write('%d %d %d %d %f\n' % (layer_0, node_0, layer_1, node_1, weight))
                    f.close()
                return
            else:
                with open('%s/%s' % (folder, filename), 'w') as f:
                    for layer_0 in range(self.num_graph):
                        for layer_1 in range(self.num_graph):
                            for node_0 in range(self.num_nodes):
                                for node_1 in range(self.num_nodes):
                                        weight = tensor[node_0][node_1][layer_0][layer_1]
                                        if not weight == 0:
                                            f.write('%d %d %d %d %f\n' % (layer_0, node_0, layer_1, node_1, weight))
                    f.close()
                return
        if type == 'single_infomap':
            for i in range(self.num_graph):
                with open('%s/%s_%s' % (folder, self._network_list[i], filename), 'w') as f:
                    for j in range(self.num_nodes):
                        for k in range(self.num_nodes):
                            if not self._graph[i][j][k] == 0:
                                f.write('%d %d %d\n' % (j, k, self._graph[i][j][k]))
                    f.close()
            return
        raise AttributeError('Unknown type %s.' % type)

    def recovery(self):
        '''
        recovery the origin data
        :return:
        '''
        self._graph = self._root_graph
        self.__update_nx_graph()

    def sub_graph(self, nodes, network_list, layers=None):
        if layers is None:
            layers = range(self.num_graph)
        nodes = sorted(nodes)
        geo_mapping = self._geo_mapping[self._geo_mapping['tazid'].isin(nodes)].copy()
        geo_mapping = geo_mapping.reset_index(drop=True)
        graph = np.zeros((self.num_graph, len(nodes), len(nodes)), dtype=np.int)
        index_nodes = [self.__get_index_by_tazid(node) for node in nodes]
        for layer in layers:
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    graph[layer][i][j] = self._graph[layer][index_nodes[i]][index_nodes[j]]
        g = GeoMultiGraph(geo_mapping=geo_mapping, graph=graph, network_list=network_list, generate_nx=True)
        return g

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
                        line = LineString([from_point, to_point])
                        connect_table['geometry'].append(line)
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
                 'degree': []}
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

    def transform(self, func='log', generate_nx=False, **kwargs):
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
        if func == 'kde':
            kde = self.__kde_weight(**kwargs)
            self._graph = [graph * (1 + kde) for graph in self._graph]
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

    def get_local_relation(self, geo_weight='queen', **kwargs):
        geo_weight_dict = {
            'queen': self.__queen_neighbor_weight,
            'queen2': self.__queen_neighbor_weight_2,
            'knn': self.__knn_weight,
            'kde': self.__kde_weight,
            'none': lambda: np.eye(self.num_nodes, dtype=np.float)
        }
        return geo_weight_dict[geo_weight](**kwargs)

    def generate_tensor(self, geo_weight='queen', connect='flow', **kwargs):
        print('Generating tensor...')
        # flow = [[1]]
        flow = []
        for i in range(self.num_graph):
            if i == 0:
                flow.append([])
            else:
                flow.append([i - 1])
        # flow.append([self.num_graph - 2])
        all_connect = [list(range(self.num_graph)) for _ in range(self.num_graph)]
        for i in range(self.num_graph):
            all_connect[i].remove(i)
        connect_dict = {
            'flow': flow,
            'memory': [[j for j in range(i)] for i in range(self.num_graph)],
            'all_connect': all_connect,
            'none': [[] for _ in range(self.num_nodes)]
        }
        tensor = tl.tensor(np.zeros([self.num_nodes, self.num_nodes, self.num_graph, self.num_graph], dtype=np.float64))
        geo_affect = self.get_local_relation(geo_weight, **kwargs)
        for node_0 in range(self.num_nodes):
            for node_1 in range(self.num_nodes):
                for layer_0 in range(self.num_graph):
                    tensor[node_0][node_1][layer_0][layer_0] = self._graph[layer_0][node_0][node_1]
                    for layer_1 in connect_dict[connect][layer_0]:
                        time_affect = self.__logistic_func(layer_0 - layer_1)
                        tensor[node_0][node_1][layer_0][layer_1] = time_affect * geo_affect[node_0][node_1]
        print('Finished.')
        return tensor

    def local_community_detection_infomap(self, geo_weight='kde', min_size=10, **kwargs):
        table = {
            'tazid': [],
            'community': []
        }
        infomap_wrapper = infomap.Infomap('--two-level --directed')
        network = infomap_wrapper.network()
        w = self.get_local_relation(geo_weight, **kwargs)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if not w[i][j] == 0:
                    network.addLink(i, j, w[i][j])
        infomap_wrapper.run()
        print("Found %d top modules with codelength: %f" %
              (infomap_wrapper.numTopModules(), infomap_wrapper.codelength()))
        for node in infomap_wrapper.iterTree():
            if node.isLeaf():
                table['tazid'].append(self.__get_tazid(node.physicalId))
                table['community'].append(node.moduleIndex())
        community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(table),
                                                                             size=min_size)
        print('%d communities found, %d point unclassified.'
              % (num_community, num_unclassify))
        return community

    def community_detection_nmf(self, n_components=10, overlapping=False, threshold=0.5, min_size=10):
        communities = []
        for g, i in zip(self._graph, range(self.num_graph)):
            print('NMF for network %s...' % self._network_list[i])
            for k in range(self.num_nodes):
                g[k][k] = self.nx_graph[i].degree(k, weight='weight')
            model = NMF(n_components=n_components,
                        init='random',
                        solver='cd',
                        max_iter=500)
            w = model.fit_transform(g)
            if not overlapping:
                community = pd.DataFrame.from_dict({
                    'tazid': [self.__get_tazid(i) for i in range(self.num_nodes)],
                    'community': list(np.argmax(w, axis=1))})
                community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(community),
                                                                                     size=min_size)
                print('Finished %s, %d communities found, %d point unclassified.'
                      % (self._network_list[i], num_community, num_unclassify))
                communities.append(community)
            else:
                community = {
                    'tazid': [],
                    'community': []
                }
                for k in range(n_components):
                    for node in range(self.num_nodes):
                        if w[node][k] > threshold:
                            community['tazid'].append(self.__get_tazid(node))
                            community['community'].append(k)
                community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(community),
                                                                                     size=min_size)
                print('Finished %s, %d communities found, %d point unclassified.'
                      % (self._network_list[i], num_community, num_unclassify))
                communities.append(pd.DataFrame.from_dict(community))
        return communities

    def louvain(self, g, resolution=1., min_size=10):
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
        community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(table),
                                                                             size=min_size)
        print('Finished %s, %d communities found, %d point unclassified.'
              % (g.graph['date'], num_community, num_unclassify))
        return community

    def community_detection_louvain(self, resolution=1., min_size=10):
        df_partition = [self.louvain(g, resolution, min_size) for g in self.nx_graph]
        return df_partition

    def info_map_c(self, in_put, out_put, min_size=1, multi_layer=False, **kwargs):
        command = '../../Infomap/Infomap '
        for key, item in kwargs.items():
            if type(item) is bool:
                if item:
                    command += '--%s ' % key.replace('_', '-')
            else:
                command += '--%s %s ' % (key.replace('_', '-'), str(item))
        command += '%s %s' % (in_put, out_put)
        print(command)
        subprocess.call(command, shell=True)
        if multi_layer:
            return self.import_expanded_map_file(filename='data/' + kwargs['out_name'] + '_expanded.map', min_size=min_size)
        else:
            return self.import_map_file(filename='data/' + kwargs['out_name'] + '.map', min_size=min_size)

    def community_detection_c(self, k=True, p=False, y=False, m=False, num_trials=False, silent=True, min_size=1):
        self.export(type='single_infomap', filename='infomap.dat', folder='data')
        result = [self.info_map_c(in_put='data/%s_infomap.dat' % network,
                                  out_put='data',
                                  out_name='%s_y%sp%sk%s' % (network, str(y), str(p), str(k)),
                                  input_format='link-list',
                                  directed=True,
                                  zero_based_numbering=True,
                                  include_self_links=k,
                                  map=True,
                                  two_level=True,
                                  self_link_teleportation_probability=y,
                                  teleportation_probability=p,
                                  markov_time=m,
                                  num_trials=num_trials,
                                  silent=silent,
                                  min_size=min_size) for network in self._network_list]
        communities = [i[0] for i in result]
        modules = [i[1] for i in result]
        links = [i[2] for i in result]
        return communities, modules, links

    def community_detection_multi_c(self, k=True, p=False, y=False, m=False, num_trials=False, silent=True, min_size=1,
                                    split=False, geo_weight='kde', connect='memory', rebuild=True, filename='infomap'):
        if split:
            filename = 'split_%s.dat' % filename
        else:
            filename = '%s.dat' % filename
        if rebuild:
            self.export(type='multi_infomap',
                        filename=filename,
                        folder='data',
                        split=split,
                        geo_weight=geo_weight,
                        connect=connect)
        communities, modules, links = self.info_map_c(in_put='data/' + filename,
                                                      out_put='data',
                                                      out_name='y%sp%sk%s_%s' % (str(y), str(p), str(k), filename),
                                                      input_format='multilayer',
                                                      directed=True,
                                                      expanded=True,
                                                      two_level=True,
                                                      zero_based_numbering=True,
                                                      include_self_links=k,
                                                      # multilayer_relax_rate=0.15,
                                                      # multilayer_relax_limit=1,
                                                      map=True,
                                                      to_nodes=True,
                                                      self_link_teleportation_probability=y,
                                                      teleportation_probability=p,
                                                      markov_time=m,
                                                      num_trials=num_trials,
                                                      silent=silent,
                                                      min_size=min_size,
                                                      multi_layer=True,)
        return communities, modules, links

    def info_map(self, g, min_size=10):
        table = {
            'tazid': [],
            'community': []
        }
        infomap_wrapper = infomap.Infomap('--two-level --directed -y 0.2')
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
        community, num_community, num_unclassify = self.__simplify_community(pd.DataFrame.from_dict(table),
                                                                             size=min_size)
        print('Finished %s, %d communities found, %d point unclassified.'
              % (g.graph['date'], num_community, num_unclassify))
        return community

    def community_detection_infomap(self, min_size=10):
        communities = []
        for g in self.nx_graph:
            communities.append(self.info_map(g, min_size))
        return communities

    def community_detection_multi_infomap(self, geo_weight='queen', connect='flow', only_self_transition=False):
        print('Multi-layer community detection by Infomap...')
        table = {
            'tazid': [],
            'layer_id': [],
            'community': []
        }
        tensor = self.generate_tensor(geo_weight=geo_weight, connect=connect)
        if geo_weight == 'none' and connect == 'none':
            infomap_wrapper = infomap.Infomap('--two-level --directed')
        else:
            infomap_wrapper = infomap.Infomap('--two-level --directed --multilayer-relax-rate 0 --multilayer-relax-limit 1')
        network = infomap_wrapper.network()
        if not only_self_transition:
            for node_0 in range(self.num_nodes):
                for node_1 in range(self.num_nodes):
                    for layer_0 in range(self.num_graph):
                        for layer_1 in range(self.num_graph):
                            weight = tensor[node_0][node_1][layer_0][layer_1]
                            if not weight == 0:
                                network.addMultilayerLink(layer_0, node_0, layer_1, node_1, weight)
        else:
            for node_0 in range(self.num_nodes):
                for node_1 in range(self.num_nodes):
                    for layer_0 in range(self.num_graph):
                        for layer_1 in range(self.num_graph):
                            if node_0 == node_1:
                                weight = tensor[node_0][node_1][layer_0][layer_1]
                                if not weight == 0:
                                    network.addMultilayerLink(layer_0, node_0, layer_1, node_1, weight)
            w = self.get_local_relation(geo_weight)
            for node_0 in range(self.num_nodes):
                for node_1 in range(self.num_nodes):
                    if not w[node_0][node_1] == 0:
                        network.addMultilayerLink(self.num_graph, node_0, self.num_graph, node_1, w[node_0][node_1])
            for node in range(self.num_nodes):
                for layer in range(self.num_graph):
                    network.addMultilayerLink(self.num_graph, node, layer, node, 0.1)
                    network.addMultilayerLink(layer, node, self.num_graph, node, 0.1)
        infomap_wrapper.run()
        print("Found %d top modules with codelength: %f" %
              (infomap_wrapper.numTopModules(), infomap_wrapper.codelength()))
        for node in infomap_wrapper.iterTree():
            if node.isLeaf():
                table['tazid'].append(self.__get_tazid(node.physicalId))
                table['community'].append(node.moduleIndex())
                table['layer_id'].append(node.layerId)
        df = pd.DataFrame.from_dict(table)
        community = [self.__simplify_community(i, size=10)[0] for i in [df[df['layer_id'] == i] for i in range(self.num_graph)]]
        return community

    def community_detection_twice(self, community, method='cinfomap', **kwargs):
        cdt = []
        for i, cl in zip(range(self.num_graph), community):
            sub_graphs = [self.sub_graph(cl[cl['community'] == m]['tazid'].unique(), network_list=['%s_sub_%d' % (n, m) for n in self._network_list]) for m in cl['community'].unique()]
            smcs = []
            num_community = 0
            for sub_graph in sub_graphs:
                try:
                    if method == 'infomap':
                        smc = sub_graph.info_map(sub_graph.nx_graph[i], **kwargs)
                    if method == 'louvain':
                        smc = sub_graph.louvain(sub_graph.nx_graph[i], **kwargs)
                    if method == 'cinfomap':
                        sub_graph.export(type='single_infomap', filename='infomap.dat', folder='data')
                        params = copy.deepcopy(kwargs)
                        out_name = params.pop('out_name')
                        smc, _, _ = sub_graph.info_map_c(in_put='data/%s_infomap.dat' % sub_graph._network_list[i],
                                                         out_put='data',
                                                         out_name='%s_%s' % (sub_graph._network_list[i], out_name),
                                                         input_format='link-list',
                                                         directed=True,
                                                         zero_based_numbering=True,
                                                         map=True,
                                                         silent=True,
                                                         **params)
                except:
                    smc = sub_graph._geo_mapping[['tazid']].copy()
                    smc['community'] = 0
                smc['community'] = smc['community'] + num_community
                num_community += len(smc['community'].unique())
                smcs.append(smc)
            cdt.append(pd.concat(smcs, axis=0))
        return cdt

    def community_detection_twice_multi(self, community, k=True, p=False, y=False, m=False, num_trials=False, silent=True, min_size=1, split=False, geo_weight='kde', connect='memory'):
        cl = merge_layers(community)
        num_community = [0 for _ in range(self.num_graph)]
        smcs = []
        sub_graphs = [self.sub_graph(cl[cl['community'] == m]['tazid'].unique(), network_list=['%s_sub_%d' % (n, m) for n in self._network_list]) for m in cl['community'].unique()]
        for sub_graph, i in zip(sub_graphs, range(len(sub_graphs))):
            smc, _, _ = sub_graph.community_detection_multi_c(k=k,
                                                              p=p,
                                                              y=y,
                                                              m=m,
                                                              num_trials=num_trials,
                                                              silent=silent,
                                                              min_size=min_size,
                                                              split=split,
                                                              geo_weight=geo_weight,
                                                              connect=connect,
                                                              filename='sub_%d' % i)
            for j in range(self.num_graph):
                smc[j]['community'] = smc[j]['community'] + num_community[j]
                num_community[j] += len(smc[j]['community'].unique())
            smcs.append(smc)
        return [pd.concat([smc[i] for smc in smcs], axis=0) for i in range(self.num_graph)]

    def draw_tensor(self, geo_weight='kde', connect='flow', cmap=OrRd_9.mpl_colormap):
        plt.rcParams['figure.figsize'] = (1200, 1200)
        plt.rcParams['figure.dpi'] = 300
        tensor = self.generate_tensor(geo_weight=geo_weight, connect=connect)
        unfolding = np.zeros((self.num_graph * self.num_nodes, self.num_graph * self.num_nodes), dtype=np.float)
        for layer_0 in range(self.num_graph):
            for layer_1 in range(self.num_graph):
                for node_0 in range(self.num_nodes):
                    for node_1 in range(self.num_nodes):
                        if layer_0 == layer_1:
                            continue
                        else:
                            unfolding[self.num_nodes * layer_0 + node_0][self.num_nodes * layer_1 + node_1] = tensor[node_0][node_1][layer_0][layer_1]
        plt.matshow(unfolding, cmap=cmap)
        plt.savefig('dist/image/%s_%s.png' % (geo_weight, connect))
        # plt.matshow(self.get_local_relation(geo_weight=geo_weight), cmap=cmap)
        # plt.savefig('dist/image/%s.png' % geo_weight)

    def draw_dist(self, hist=True, kde=True, rug=True, bins=10):
        sns.set_style('ticks')
        sns.set(color_codes=True)
        graphs = sns.FacetGrid(self.edges, col='network')
        graphs.map(sns.distplot, 'weight', hist=hist, kde=kde, rug=rug, bins=bins)
        plt.show()

    def draw_distance_dist(self, hist=True, kde=True, rug=True, bins=10):
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

    def draw_choropleth_map(self, map_view, data, value='', integer=True, title='Choropleth Map', c_map=Spectral_10.mpl_colormap):
        timestamp = int(time.time())
        value_min = data[value].min()
        value_max = data[value].max()
        if integer:
            cmap = IntegerColorMap(value_max - value_min + 1)

            def func(x):
                return cmap.get_hex_color(int(x[value] + value_min))
            set_color = func
        else:
            def func(x):
                rgba = c_map((x[value]))
                return rgb2hex(rgba[0], rgba[1], rgba[2])
            set_color = func
        value_geo_map = gpd.GeoDataFrame(data.merge(self._geo_mapping, on='tazid'))
        value_geo_map = value_geo_map[['tazid', value, 'geometry']]
        value_geo_map['color'] = value_geo_map.apply(set_color, axis=1)
        value_geo_map.to_file('dist/data/%s.geojson' % (title + str(timestamp)), driver='GeoJSON')
        source = GeojsonSource(id=value.replace('_', '') + title, data='%s.geojson' % (title + str(timestamp)))
        map_view.add_source(source)
        layer = FillLayer(id=value.replace('_', '') + title, source=value.replace('_', '') + title, p_fill_opacity=0.7, p_fill_color=['get', 'color'])
        map_view.add_layer(layer)
        map_view.update()

    def draw_extrusion_choropleth_map(self, map_view, data, value='', integer=True, title='Choropleth Map', c_map=Spectral_10.mpl_colormap):
        timestamp = int(time.time())
        value_min = data[value].min()
        value_max = data[value].max()
        if integer:
            cmap = IntegerColorMap(value_max - value_min + 1)

            def func(x):
                return cmap.get_hex_color(int(x[value] + value_min))
            set_color = func
        else:
            def func(x):
                rgba = c_map((x[value]))
                return rgb2hex(rgba[0], rgba[1], rgba[2])
            set_color = func
        value_geo_map = gpd.GeoDataFrame(data.merge(self._geo_mapping, on='tazid'))
        value_geo_map = value_geo_map[['tazid', value, 'geometry']]
        value_geo_map['color'] = value_geo_map.apply(set_color, axis=1)
        value_geo_map.to_file('dist/data/%s_extrusion.geojson' % (title + str(timestamp)), driver='GeoJSON')
        source = GeojsonSource(id=value.replace('_', '') + title, data='%s_extrusion.geojson' % (title + str(timestamp)))
        map_view.add_source(source)
        layer = FillExtrusionLayer(id=value.replace('_', '') + title,
                                   source=value.replace('_', '') + title,
                                   p_fill_extrusion_opacity=0.2,
                                   p_fill_extrusion_color=['get', 'color'],
                                   p_fill_extrusion_height=300)
        map_view.add_layer(layer)
        map_view.update()

    def draw_community_link(self, map_view, module, link, title='ml'):
        timestamp = int(time.time())
        cmap = IntegerColorMap(module['community'].max())
        module['color'] = module.apply(lambda x: cmap.get_hex_color(x['community']), axis=1)
        f_max = module['flow'].max()
        f_min = module['flow'].min()
        module['flow_norm'] = module.apply(lambda x: (x['flow'] - f_min)/(f_max - f_min), axis=1)
        module['radius'] = module.apply(lambda x: x['flow_norm'] * 75 + 5, axis=1)
        module.to_file('dist/data/m_%s.geojson' % (title + str(timestamp)), driver='GeoJSON')
        m_source = GeojsonSource(id=title + 'm', data='m_%s.geojson' % (title + str(timestamp)))
        map_view.add_source(m_source)
        m_layer = CircleLayer(id=title + 'm', source=title + 'm', p_circle_radius=['get', 'radius'], p_circle_opacity=1., p_circle_color=['get', 'color'])
        # link['color'] = link.apply(lambda x: 'white' if x['from'] > x['to'] else 'white', axis=1)
        f_max = link['flow'].max()
        f_min = link['flow'].min()
        link['flow_norm'] = link.apply(lambda x: (x['flow'] - f_min) / (f_max - f_min), axis=1)
        link['opacity'] = link.apply(lambda x: x['flow_norm'] * 0.5 + 0.3, axis=1)
        link['width'] = link.apply(lambda x: x['flow_norm'] * 10 + 0.1, axis=1)
        # link['offset'] = link.apply(lambda x: x['width'] / 2 if x['from'] > x['to'] else - x['width'] / 2, axis=1)
        link.to_file('dist/data/l_%s.geojson' % (title + str(timestamp)), driver='GeoJSON')
        l_source = GeojsonSource(id=title + 'l', data='l_%s.geojson' % (title + str(timestamp)))
        map_view.add_source(l_source)
        l_layer = LineLayer(id=title + 'l',
                            source=title + 'l',
                            p_line_color='#E7FDA5',
                            p_line_width=['get', 'width'],
                            p_line_opacity=['get', 'opacity'],
                            l_line_join='round',
                            l_line_cap='round')
        map_view.add_layer(l_layer)
        map_view.add_layer(m_layer)
        map_view.update()

    def draw_gradient_community(self, community, column=2, row=3, community_list=None, inline=False, title='Gradient-Community'):
        for c in community:
            c['num_nodes'] = c.apply(lambda x: len(c[c['community'] == x['community']]), axis=1)
            c = c.sort_values(by='num_nodes')
        if community_list is None:
            communities = [[c[c['community'] == i] for i in c['community'].unique()[:5]] for c in community]
        else:
            communities = [[c[c['community'] == i] for i in community_list] for c in community]
        view = PlotView(column_num=row, row_num=column, title=title)
        for subview, i in zip(view, range(self.num_graph)):
            subview.name = self._network_list[i]
        maps = [MapBox(name='map_%d' % i,
                       pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                       lon=116.37363,
                       lat=39.915606,
                       style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                       pitch=0,
                       bearing=0,
                       zoom=9,
                       viewport=view[i]) for i in range(self.num_graph)]
        for i in range(self.num_graph):
            c_map = MultiGradColorMap(num_color=len(communities[i]), grad_value=1.)
            for c, j in zip(communities[i], range(len(communities[i]))):
                self.draw_choropleth_map(map_view=maps[i], data=c, value='flow_norm', integer=False,
                                         title='%sc%d' % (self._network_list[i], j), c_map=c_map.get_c_map(j))
        view.plot(inline=inline)

    def draw_multi_scale_community_link(self, module, link, column=2, row=3, inline=False, title='Multi-Link'):
        view = PlotView(column_num=row, row_num=column, title=title)
        for subview, i in zip(view, range(self.num_graph)):
            subview.name = self._network_list[i]
        maps = [MapBox(name='map_%d' % i,
                       pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                       lon=116.37363,
                       lat=39.915606,
                       style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                       pitch=0,
                       bearing=0,
                       zoom=9,
                       viewport=view[i]) for i in range(self.num_graph)]
        for i in range(self.num_graph):
            self.draw_community_link(map_view=maps[i], module=module[i], link=link[i],
                                     title='%sml' % self._network_list[i])
        view.plot(inline=inline)

    def draw_multi_scale_community(self, community, cmap=Spectral_10, column=2, row=3, inline=False, title='Geo-Multi-Graph'):
        view = PlotView(column_num=row, row_num=column, title=title)
        for subview, i in zip(view, range(self.num_graph)):
            subview.name = self._network_list[i]
        maps = [MapBox(name='map_%d' % i,
                       pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                       lon=116.37363,
                       lat=39.915606,
                       style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                       pitch=0,
                       bearing=0,
                       zoom=9,
                       viewport=view[i]) for i in range(self.num_graph)]
        for i in range(self.num_graph):
            self.draw_choropleth_map(map_view=maps[i], data=community[i], value='community', title='%scommunity' % self._network_list[i], c_map=cmap)
        view.plot(inline=inline)

    def draw_network(self, color='white', width=1., value='weight', bk=True, inline=False, row=3, column=2, percent=100):
        view = PlotView(column_num=row, row_num=column, title='Network')
        for subview, i in zip(view, range(self.num_graph)):
            subview.name = self._network_list[i]
        maps = [MapBox(name='map_network_%d' % i,
                       pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                       lon=116.37363,
                       lat=39.915606,
                       style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                       pitch=0,
                       bearing=0,
                       zoom=10,
                       viewport=view[i]) for i in range(self.num_graph)]
        edge_df = self.edges_geo
        for i in range(self.num_graph):
            self.draw_single_network(map_view=maps[i],
                                     data=edge_df[edge_df['network'] == self._network_list[i]].copy(),
                                     color=color,
                                     width=width,
                                     value=value,
                                     title='Network-%s' % self._network_list[i],
                                     bk=bk,
                                     percent=percent)
        view.plot(inline=inline)

    def draw_single_network(self, map_view, data, color='white', width=1., value='weight', title='network', bk=True, percent=100):
        data = data.sort_values(value)
        data = data[:int(len(data) * percent / 100)].copy()
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

    def draw_taz(self, map_view, fill_color='white', fill_opacity=0.3, fill_outline_color='white'):
        taz_unit_df = self._geo_mapping['geometry']
        taz_unit_df.to_file('dist/data/taz.geojson', driver='GeoJSON')
        source = GeojsonSource(id='taz', data='taz.geojson')
        map_view.add_source(source)
        layer = FillLayer(id='taz',
                          source='taz',
                          p_fill_opacity=fill_opacity,
                          p_fill_color=fill_color,
                          p_fill_outline_color=fill_outline_color)
        map_view.add_layer(layer)
        map_view.update()

    def draw_multi_community_extrusion(self, map_view, communities, cmap=Spectral_10, title='Multi-Community-Extrusion'):
        timestamp = int(time.time())
        value_min = min([community['community'].min() for community in communities])
        value_max = max([community['community'].max() for community in communities])
        mpl_colormap = cmap.get_mpl_colormap(N=value_max - value_min + 1)

        def set_color(x):
            rgba = mpl_colormap((x + value_min))
            return rgb2hex(rgba[0], rgba[1], rgba[2])
        _ = Legend(view=map_view.viewport,
                        colors=[('Community_%d' % i, set_color(i)) for i in range(value_min, value_max + 1)],
                        title='Community Detection')
        for community, i in zip(communities, range(self.num_graph)):
            geo_map = self._geo_mapping.merge(community, on='tazid')
            geo_map = geo_map[['tazid', 'community', 'geometry']]
            geo_map['color'] = geo_map.apply(lambda x: set_color(x['community']), axis=1)
            geo_map.to_file('dist/data/%s_%d_%s.geojson' % (title, i, str(timestamp)), driver='GeoJSON')
            source = GeojsonSource(id='%s%d' % (title, i),
                                   data='%s_%d_%s.geojson' % (title, i, str(timestamp)))
            map_view.add_source(source)
            layer = FillExtrusionLayer(id='%s%d' % (title, i),
                                       source='%s%d' % (title, i),
                                       p_fill_extrusion_opacity=0.3,
                                       p_fill_extrusion_color=['get', 'color'],
                                       p_fill_extrusion_height=1000 * (i + 1),
                                       p_fill_extrusion_base=1000 * i)
            map_view.add_layer(layer)
            map_view.update()

    def draw_mix_result(self, sc, mc, title='Mix-result', column=2, row=3, inline=False):
        view = PlotView(column_num=row, row_num=column, title=title)
        for subview, i in zip(view, range(self.num_graph)):
            subview.name = self._network_list[i]
        maps = [MapBox(name='map_%d' % i,
                       pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                       lon=116.37363,
                       lat=39.915606,
                       style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                       pitch=30,
                       bearing=0,
                       zoom=9,
                       viewport=view[i]) for i in range(self.num_graph)]
        for i in range(self.num_graph):
            self.draw_choropleth_map(map_view=maps[i], data=sc[i], value='community', title='%scommunity' % self._network_list[i])
        view.plot(inline=inline)

    def read_multi_tensor_result(self, filename, vec='u', min_size=10):
        u_file = 'u_' + filename
        v_file = 'v_' + filename
        u = []
        v = []
        with open(u_file, 'r') as f:
            for line in f.readlines()[1:]:
                u.append([float(i) for i in line.split(' ')[1:]])
            f.close()
        u = np.array(u)
        u = normalize(u, axis=1, norm='l1')
        with open(v_file, 'r') as f:
            for line in f.readlines()[1:]:
                v.append([float(i) for i in line.split(' ')[1:]])
            f.close()
        v = np.array(v)
        v = normalize(v, axis=1, norm='l1')
        if vec == 'u':
            community = pd.DataFrame.from_dict({
                'tazid': [self.__get_tazid(i) for i in range(self.num_nodes)],
                'community': list(np.argmax(u, axis=1))})
            community, num_community, num_unclassify = self.__simplify_community(community, size=min_size)
            print('Finished %d communities found, %d point unclassified.' % (num_community, num_unclassify))
            return community
        if vec == 'v':
            community = pd.DataFrame.from_dict({
                'tazid': [self.__get_tazid(i) for i in range(self.num_nodes)],
                'community': list(np.argmax(v, axis=1))})
            community, num_community, num_unclassify = self.__simplify_community(community, size=min_size)
            print('Finished %d communities found, %d point unclassified.' % (num_community, num_unclassify))
            return community
        if vec == 'uv':
            community = pd.DataFrame.from_dict({
                'tazid': [self.__get_tazid(i) for i in range(self.num_nodes)],
                'community': list(np.argmax(v, axis=1))})
            community, num_community, num_unclassify = self.__simplify_community(community, size=min_size)
            print('Finished %d communities found, %d point unclassified.' % (num_community, num_unclassify))
            return community

    def __get_tazid(self, index):
        return self._geo_mapping['tazid'][index]

    def __get_index_by_tazid(self, tazid):
        return self._geo_mapping[self._geo_mapping['tazid'] == tazid].index.tolist()[0]

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

    def __simplify_community(self, community, size=10):
        c_list = {}
        r_list = []
        un_list = []
        for _, line in community.iterrows():
            if line['community'] in c_list.keys():
                c_list[line['community']].append(line['tazid'])
            else:
                c_list[line['community']] = [line['tazid']]
        for key, item in c_list.items():
            if (len(item) <= size) or key == -1:
                for i in item:
                    un_list.append(i)
            else:
                r_list.append(item)
        exist_list = community['tazid'].tolist()
        for taz in self._geo_mapping['tazid']:
            if taz not in exist_list:
                un_list.append(taz)
        table = {
            'tazid': [],
            'community': []
        }
        for r, i in zip(r_list, range(len(r_list))):
            for k in r:
                table['tazid'].append(k)
                table['community'].append(i)
        # classify single points by knn
        try:
            w = weights.KNN.from_dataframe(self._geo_mapping, geom_col='geometry', k=len(un_list) + 1)
            for i in un_list:
                neighbors = w.neighbors[self.__get_index_by_tazid(i)]
                neighbor_community = []
                for neighbor in neighbors:
                    tazid = self.__get_tazid(neighbor)
                    if tazid not in un_list:
                        neighbor_community.append(int(community[community['tazid'] == tazid]['community']))
                table['tazid'].append(i)
                table['community'].append(np.argmax(np.bincount(neighbor_community)))
        except IndexError:
            print('Too many unclassified node.')
            for i in un_list:
                table['tazid'].append(i)
                table['community'].append(len(r_list))
        result = pd.DataFrame.from_dict(table)

        queen = weights.Queen.from_dataframe(self._geo_mapping, geom_col='geometry')
        for index, row in result.iterrows():
            neighbor = queen.neighbors[self.__get_index_by_tazid(row['tazid'])]
            if len(neighbor) == 0:
                continue
            c_neighbor = [int(result['community'][result['tazid'] == self.__get_tazid(i)]) for i in neighbor]
            if not row['community'] in c_neighbor:
                result.loc[index, 'community'] = np.argmax(np.bincount(c_neighbor))
        community = community.drop(columns=['community'])
        result = result.merge(community, on='tazid')
        return result, len(r_list), len(un_list)

    def __queen_neighbor_weight(self):
        w = weights.Queen.from_dataframe(self._geo_mapping, geom_col='geometry')
        td = np.zeros([self._num_nodes, self._num_nodes], dtype=np.float)
        for i in range(self.num_nodes):
            td[i][i] = 1.
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if j in w.neighbors[i]:
                    td[i][j] = 0.5
                    td[j][i] = 0.5
        return td

    def __queen_neighbor_weight_2(self):
        w = weights.Queen.from_dataframe(self._geo_mapping, geom_col='geometry')
        td = np.zeros([self._num_nodes, self._num_nodes], dtype=np.float)
        for i in range(self.num_nodes):
            td[i][i] = 1.
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if j in w.neighbors[i]:
                    td[i][j] = .5
                    td[j][i] = .5
                else:
                    for k in w.neighbors[i]:
                        if j in w.neighbors[k]:
                            td[i][j] = 0.2
                            td[j][i] = 0.2
                            break
        return td

    def __knn_weight(self, k=10):
        w = weights.KNN.from_dataframe(self._geo_mapping, geom_col='geometry', k=k)
        td = np.zeros([self._num_nodes, self._num_nodes], dtype=np.float)
        for i in range(self.num_nodes):
            td[i][i] = 1.
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if j in w.neighbors[i]:
                    td[i][j] = 0.5
                    td[j][i] = 0.5
        return td

    def __kde_weight(self, bandwidth=None, function='gaussian'):
        w = weights.Kernel.from_dataframe(self._geo_mapping, geom_col='geometry', bandwidth=bandwidth, function=function)
        td = np.zeros([self._num_nodes, self._num_nodes], dtype=np.float)
        for i in range(self.num_nodes):
            td[i][i] = 1.
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if j in w.neighbors[i]:
                    td[i][j] = w.weights[i][w.neighbors[i].index(j)]
                    td[j][i] = w.weights[i][w.neighbors[i].index(j)]
        return td

    @staticmethod
    def __logistic_func(x):
        # return 1. - (1./(1. + pow(np.e, -1 * abs(x))))
        return (1. - (1./(1. + pow(np.e, -1 * abs(x))))) * 0.02


if __name__ == '__main__':
    gmg = GeoMultiGraph()
    gmg.load('GeoMultiGraph_week', generate_nx=True, network_list=['2012', '2013', '2014', '2015', '2016', '2017'])
    # gmg.threshold(t_min=3, generate_nx=True)
    # gmg.transform(func='sqrt', generate_nx=True)
    # gmg.draw_dist(hist=True, kde=False, rug=False, bins=20)
    # gmg.draw_qq_plot()
    cl = gmg.community_detection_louvain(min_size=20, resolution=2.)
    gmg.draw_multi_scale_community(community=cl, cmap=Spectral_10)
