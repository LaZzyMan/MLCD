import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from pywebplot.MapBox import MapBox

NETWORK_LIST = ['2012',
                '2013',
                '2014',
                '2015',
                '2016',
                '2017']


class GeoMultiGraph:
    def __init__(self, geo_mapping=None, graph=None):
        self.geo_mapping = geo_mapping
        self.graph = graph
        self.nx_graph = None
        self.mb = None
        if graph is None:
            self.num_nodes = 0
        else:
            self.num_nodes = len(graph[0])

    def save(self, file_name):
        self.geo_mapping.to_file('src/data/' + file_name + '.geojson', driver='GeoJSON')
        np.save('src/data/' + file_name + '.npy', self.graph)

    def load(self, file_name):
        self.graph = np.load('src/data/' + file_name + '.npy')
        self.geo_mapping = gpd.read_file('src/data/' + file_name + '.geojson')
        self.num_nodes = len(self.graph[0])

    def to_nx_graph(self, min_weight=1, max_weight=10000000):
        G = []
        for l, time in zip(self.graph, NETWORK_LIST):
            print('Generating Network %s' % time)
            g = nx.DiGraph(date=time)
            for i in range(len(l)):
                g.add_node(i, tazid=self.geo_mapping['tazid'][i])
            for i in range(len(l)):
                for j in range(len(l)):
                    if i == j:
                        continue
                    if min_weight <= l[i, j]:
                        if l[i, j] > max_weight:
                            g.add_edge(i, j, weight=max_weight)
                        else:
                            g.add_edge(i, j, weight=l[i, j])
            G.append(g)
            self.nx_graph = G
        return G

    def draw_map_community(self, community=None, title='Community.png'):
        community_geo_map = self.geo_mapping.merge(community, on='tazid')
        filtered_community = community_geo_map[community_geo_map['size'] >= 5]
        filtered_community.plot(column='community', cmap='Dark2')
        plt.savefig(title)
        plt.show()

    def draw_map_centrality(self, community=None):
        community_geo_map = self.geo_mapping.merge(community, on='tazid')
        community_geo_map.plot(column='centrality', cmap='hot')
        plt.savefig('hot_centrality.png')
        plt.show()

    def draw_map(self, value_name, cmap, data):
        community_geo_map = self.geo_mapping.merge(data, on='tazid')
        community_geo_map.plot(column=value_name, cmap=cmap)
        plt.savefig('%s_%s.png' % (value_name, cmap))
        plt.show()

    def draw_network_on_map(self, cmap):
        for g in self.nx_graph:
            print('Drawing network %s' % g.graph['date'])
            nx_graph = nx.Graph(g)
            connect_table = {'from_tazid': [],
                             'to_taziid': [],
                             'weight': [],
                             'geometry': []}
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    try:
                        weight = nx_graph.adj[i][j]['weight']
                        if weight == 0:
                            continue
                        connect_table['weight'].append(weight)
                        from_tazid = nx_graph.nodes[i]['tazid']
                        to_tazid = nx_graph.nodes[j]['tazid']
                        connect_table['to_taziid'].append(to_tazid)
                        connect_table['from_tazid'].append(from_tazid)
                        from_point = self.geo_mapping[self.geo_mapping.tazid == from_tazid].geometry[i].centroid
                        to_point = self.geo_mapping[self.geo_mapping.tazid == to_tazid].geometry[j].centroid
                        connect_table['geometry'].append(LineString([from_point, to_point]))
                    except KeyError as _:
                        continue
            connect_df = gpd.GeoDataFrame.from_dict(connect_table)
            connect_df.plot(column='weight', cmap=cmap)
            plt.savefig('network_on_map_3.png')
            plt.show()

    def create_mapbox(self, title='Mapbox', style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87', lon=116.37363, lat=39.915606, pitch=45, bearing=0, zoom=12):
        print('Creating mapbox map...')
        self.mb = MapBox(title=title,
                         pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                         lon=lon,
                         lat=lat,
                         style=style,
                         pitch=pitch,
                         bearing=bearing,
                         zoom=zoom)

    def mapbox_show(self):
        print('Open your web browser.')
        self.mb.show()

    def mapbox_draw_single_network(self, color='white', index=0, width=0.3):
        print('Drawing single layer network on map...')
        if self.mb is None:
            self.create_mapbox()
        nx_graph = nx.Graph(self.nx_graph[index])
        connect_table = {'from_tazid': [],
                         'to_taziid': [],
                         'weight': [],
                         'geometry': []}
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                try:
                    weight = nx_graph.adj[i][j]['weight']
                    # if weight <= 2:
                    #     continue
                    connect_table['weight'].append(weight)
                    from_tazid = nx_graph.nodes[i]['tazid']
                    to_tazid = nx_graph.nodes[j]['tazid']
                    connect_table['to_taziid'].append(to_tazid)
                    connect_table['from_tazid'].append(from_tazid)
                    from_point = self.geo_mapping[self.geo_mapping.tazid == from_tazid].geometry[i].centroid
                    to_point = self.geo_mapping[self.geo_mapping.tazid == to_tazid].geometry[j].centroid
                    connect_table['geometry'].append(LineString([from_point, to_point]))
                except KeyError as _:
                    continue
        connect_df = gpd.GeoDataFrame.from_dict(connect_table)
        min_weight = connect_df['weight'].min()
        max_weight = connect_df['weight'].max()
        connect_df['weight'].to_csv('connect_2012.csv')
        connect_df['opacity'] = connect_df['weight'].map(lambda x: (x-min_weight)/(max_weight-min_weight)*0.8+0.00)
        draw_df = connect_df[['geometry', 'opacity']]
        draw_df.to_file('src/network_%d.geojson' % index, driver='GeoJSON')
        self.mb.add_geojson_source(geojson='network_%d.geojson' % index, name='network_%d' % index)
        self.mb.add_layer(name='network_%d' % index, source='network_%d' % index, type='line',
                          paint={'line-color': color, 'line-width': width, 'line-opacity': ['get', 'opacity']})

    def mapbox_draw_taz_unit(self, color='#3BA1C3', fill_opacity=0.3):
        print('Drawing taz unit on map...')
        if self.mb is None:
            self.create_mapbox()
        taz_unit_df = self.geo_mapping['geometry']
        taz_unit_df.to_file('src/taz.geojson', driver='GeoJSON')
        self.mb.add_geojson_source(geojson='taz.geojson', name='taz')
        self.mb.add_layer(name='taz', source='taz', type='fill',
                          paint={'fill-color': color, 'fill-opacity': fill_opacity})

    def mapbox_draw_background(self, color='black', opacity=1):
        print('Drawing background on map...')
        if self.mb is None:
            self.create_mapbox()
        self.mb.add_layer(name='bk', type='background',
                          paint={'background-color': color, 'background-opacity': opacity})


if __name__ == '__main__':
    gmg = GeoMultiGraph()
    gmg.load('GeoMultiGraph')
    gmg.to_nx_graph()
    gmg.draw_map_community()
