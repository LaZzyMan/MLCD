from pywebplot import *
from palettable.cartocolors.qualitative import Antique_10, Bold_10, Pastel_10, Prism_10, Safe_10, Vivid_10
from GeoMultiGraph import *
import subprocess
from functools import reduce
import numpy as np
import copy
import json
from palettable.colorbrewer.sequential import Reds_9, GnBu_9, BuPu_9, Blues_9


def run_infomap_command(in_put, out_put, **kwargs):
    command = '../../Infomap/Infomap '
    for key, item in kwargs.items():
        if type(item) is bool:
            if item:
                command += '--%s ' % key.replace('_', '-')
        else:
            command += '--%s %s ' % (key.replace('_', '-'), str(item))
    command += '%s %s' % (in_put, out_put)
    subprocess.call(command, shell=True)
    return in_put.split('.')[0] + '.map'


def get_code_length(filename):
    code_length = 0.
    with open(filename, 'r') as f:
        for line in f.readlines():
            if '# codelength' in line:
                code_length = float(line.split(' ')[2])
                break
        f.close()
    return code_length


def grid_search(func, score, param_grid, **kwargs):
    params = []
    param_names = []
    for key, item in param_grid.items():
        param_names.append(key)
        params.append(item)
    params.insert(0, [[]])

    def grid(x, y):
        r = []
        for i in x:
            for j in y:
                tmp = copy.deepcopy(i)
                tmp.append(j)
                r.append(tmp)
        return r
    pg = reduce(grid, params)
    scores = []
    for i in pg:
        display = ''
        for p, v in zip(param_names, i):
            kwargs[p] = v
            display += '%s: %s, ' % (p, str(v))
        r = func(**kwargs)
        s = score(r)
        scores.append(s)
        print(display + 'score: %s.' % s)
    best = pg[scores.index(min(scores))]
    display = ''
    for p, v in zip(param_names, best):
        display += '%s: %s \n' % (p, str(v))
    print('Best params: \n' + display)
    return best, [i.append(j) for i, j in zip(pg, scores)]


if __name__ == '__main__':
    WEB_SERVER.run()
    networks = ['2012', '2013', '2014', '2015', '2016', '2017']
    gmg = GeoMultiGraph()
    gmg.load('../src/data/GeoMultiGraph_week', network_list=networks, generate_nx=True)
    mkdir()
    cl = gmg.community_detection_c(k=False, p=False, y=False, num_trials=False, silent=False)
    gmg.draw_multi_scale_community(community=cl, inline=True, title='c-infomap-single-default')
    # ommunity_detection_twice(community=cl, method='infomap', min_size=5)
    # gmg.draw_multi_scale_community(community=cll, cmap=Prism_10, inline=True, title='c-infomap-single-infomap')
    # cl, _ = gmg.import_map_file('data/2012_infomap.map', min_size=5)
    # sub_graphs = [gmg.sub_graph(cl[cl['community'] == i]['tazid'].unique()) for i in cl['community'].unique()]
    # for i in range(len(sub_graphs)):
    #     sub_graphs[i].export(type='single_infomap', filename='infomap_sub_%d.dat' % i)
    # sub_graphs = [gmg.sub_graph(cl[cl['community'] == i]['tazid'].unique()) for i in cl['community'].unique()]
    # sub_graphs[0].export(type='single_infomap', filename='test.dat')
    # grid_result = {}
    # for network in networks:
    #     best, result = grid_search(func=run_infomap_command,
    #                           score=get_code_length,
    #                           param_grid={
    #                               'teleportation_probability': list(np.arange(0, 0.3, 0.01)),
    #                               'self_link_teleportation_probability': list(np.arange(0, 0.3, 0.01)),
    #                               'include_self_links': [True, False]
    #                           },
    #                           in_put='data/%s_infomap.dat' % network,
    #                           out_put='data',
    #                           input_format='link-list',
    #                           directed=True,
    #                           zero_based_numbering=True,
    #                           include_self_links=True,
    #                           map=True,
    #                           num_trials=20,
    #                           silent=True)
    #     grid_result[network] = {
    #         'best': best,
    #         'result': result
    #     }
    # with open('single_grid.json', 'w') as f:
    #     json.dump(grid_result, f)
    #     f.close()
    # for i in networks:
    #     run_infomap_command(in_put='data/%s_infomap.dat' % i,
    #                         out_put='data',
    #                         input_format='link-list',
    #                         directed=True,
    #                         zero_based_numbering=True,
    #                         include_self_links=True,
    #                         map=True,
    #                         teleportation_probability=0.2,
    #                         self_link_teleportation_probability=0.15,
    #                         num_trials=20)

    # gmg.draw_multi_scale_community(community=cl, cmap=Prism_10, inline=True, title='c-infomap-single-p020-y015-k')
    # gmg.export(type='single_infomap', filename='infomap.dat')
    # gmg.export(type='multi_infomap',
    #            filename='split_infomap.dat',
    #            split=True,
    #            geo_weight='kde',
    #            connect='memory')
    # gmg.export(type='multi_infomap',
    #            filename='infomap.dat',
    #            split=False,
    #            geo_weight='kde',
    #            connect='memory')
    # cll = gmg.community_detection_louvain(resolution=1., min_size=10)
    # gmg.draw_multi_scale_community(community=cll, cmap=Prism_10, inline=True, title='louvain')
    # scll = gmg.community_detection_twice(cll, method='louvain', min_size=10)
    # gmg.draw_multi_scale_community(community=scll, cmap=Prism_10, inline=True, title='s-louvain')
    # c_u = gmg.read_multi_tensor_result(filename='K10_result.dat', vec='u')
    # c_v = gmg.read_multi_tensor_result(filename='K10_result.dat', vec='v')
    # view = PlotView(column_num=2, row_num=1, title='multi-tensor')
    # view[0].name = 'u'
    # view[1].name = 'v'
    # maps = [MapBox(name='map_%d' % i,
    #                pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
    #                lon=116.37363,
    #                lat=39.915606,
    #                style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
    #                pitch=55,
    #                bearing=0,
    #                zoom=12,
    #                viewport=view[i]) for i in range(2)]
    # gmg.draw_choropleth_map(map_view=maps[0], data=c_u, value='community',
    #                              title='ucommunity', cmap=Vivid_10)
    # gmg.draw_choropleth_map(map_view=maps[1], data=c_v, value='community',
    #                         title='vcommunity', cmap=Vivid_10)
    # view.plot(inline=True)
    # cl = gmg.community_detection_infomap(min_size=0)
    # gmg.draw_multi_scale_community(community=cl, cmap=Prism_10, inline=True, title='infomap-m0')
    # scl = gmg.community_detection_twice(cl, method='infomap', min_size=0)
    # gmg.draw_multi_scale_community(community=scl, cmap=Prism_10, inline=True, title='s-infomap-m0')
    # sscl = gmg.community_detection_twice(scl, method='infomap', min_size=0)
    # gmg.draw_multi_scale_community(community=sscl, cmap=Prism_10, inline=True, title='ss-infomap-m0')
    # cll = gmg.community_detection_louvain(resolution=1., min_size=10)
    # gmg.draw_multi_scale_community(community=cll, cmap=Prism_10, inline=True, title='louvain')
    # scll = gmg.community_detection_twice(cll, method='louvain', min_size=10)
    # gmg.draw_multi_scale_community(community=scll, cmap=Prism_10, inline=True, title='s-louvain')
    # mc = gmg.community_detection_multi_infomap(geo_weight='kde', connect='memory', only_self_transition=True)
    # gmg.draw_multi_scale_community(community=mc, cmap=Pastel_10, inline=True, title='memory-kde-gc')
    # gmg.export(type='MultiTensor', filename='adj.dat')
    # gc = gmg.local_community_detection_infomap(geo_weight='kde', min_size=10)
    # plt = PlotView(column_num=1, row_num=1, title='gc-kde')
    # plt[0].name = 'gc'
    # map = MapBox(name='map_gc',
    #              pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
    #              lon=116.37363,
    #              lat=39.915606,
    #              style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
    #              pitch=55,
    #              bearing=0,
    #              zoom=12,
    #              viewport=plt[0])
    # gmg.draw_choropleth_map(map_view=map, data=gc, value='community', title='gc-kde', cmap=Antique_10)
    # plt.plot(inline=True)
    # connect_dict = ['flow', 'memory', 'all_connect', 'none']
    # geo_weight_dict = ['queen', 'queen2', 'knn', 'kde', 'none']
    # for con in connect_dict:
    #     for geo in geo_weight_dict:
    #         mc = gmg.community_detection_multi_infomap(geo_weight=geo, connect=con, only_self_transition=False)
    #         gmg.draw_multi_scale_community(community=mc, cmap=Pastel_10, inline=True, title='%s-%s' % (con, geo))
    #         plt = PlotView(column_num=1, row_num=1, title='extrusion-%s-%s' % (con, geo))
    #         plt[0].name = '%s-%s' % (con, geo)
    #         map = MapBox(name='map_infomap',
    #                      pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
    #                      lon=116.37363,
    #                      lat=39.915606,
    #                      style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
    #                      pitch=55,
    #                      bearing=0,
    #                      zoom=12,
    #                      viewport=plt[0])
    #         gmg.draw_taz(map_view=map)
    #         gmg.draw_multi_community_extrusion(map_view=map,
    #                                            communities=mc,
    #                                            cmap=Pastel_10,
    #                                            title='%s%s' % (con, geo))
    #         plt.plot(inline=True)
    # closeness_centrality = get_closeness_centrality(gmg.nx_graph[0])
    # plt = PlotView(column_num=1, row_num=1, title='Statistics-Information')
    # plt[0].name = 'closeness_centrality'
    # map = MapBox(name='map_si_0',
    #              pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
    #              lon=116.37363,
    #              lat=39.915606,
    #              style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
    #              pitch=55,
    #              bearing=0,
    #              zoom=12,
    #              viewport=plt[0])
    # gmg.draw_choropleth_map(map_view=map, data=closeness_centrality, value='closeness_centrality',
    #                         title='closeness-centrality', cmap=Reds_9)
    # plt.plot(inline=False)
    # gmg.threshold(t_min=20, generate_nx=True)
    # gmg.transform(func='sqrt', generate_nx=True)
    # gmg.draw_dist(hist=True, kde=False, rug=False, bins=20)
    # gmg.draw_qq_plot()
    # gmg.draw_network(color='white', width=1., value='weight', bk=True, inline=False, row=3, column=2)
