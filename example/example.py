from pywebplot import *
from palettable.cartocolors.qualitative import Antique_10, Bold_10, Pastel_10, Prism_10, Safe_10, Vivid_10
from GeoMultiGraph import *
from palettable.colorbrewer.sequential import Reds_9, GnBu_9, BuPu_9, Blues_9


if __name__ == '__main__':
    WEB_SERVER.run()
    mkdir()
    gmg = GeoMultiGraph()
    gmg.load('../src/data/GeoMultiGraph_week', network_list=['2012', '2013', '2014', '2015', '2016', '2017'], generate_nx=True)
    # cl = gmg.community_detection_nmf(n_components=5, overlapping=False)
    # gmg.draw_multi_scale_community(community=cl, cmap=Prism_10, inline=False, title='nmf-10')
    # gmg.export(type='MultiTensor', filename='adj.dat')
    # gc = gmg.local_community_detection_infomap(geo_weight='kde', min_size=10)
    # plt = PlotView(column_num=1, row_num=1, title='gc')
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
    # gmg.draw_choropleth_map(map_view=map, data=gc, value='community', title='gc', cmap=Antique_10)
    # plt.plot(inline=True)
    # mc = gmg.community_detection_multi_infomap(geo_weight='none', connect='none', only_self_transition=False)
    # mmc = merge_layers(mc)
    # sub_graphs = [gmg.sub_graph(mmc[mmc['community'] == i]['tazid'].unique()) for i in mmc['community'].unique()]
    # for i in range(len(sub_graphs)):
    #     # smc = sub_graphs[0].community_detection_infomap(min_size=10)
    #     # smc = sub_graphs[i].community_detection_multi_infomap(geo_weight='kde', connect='memory', only_self_transition=False)
    #     smc = sub_graphs[0].community_detection_louvain(resolution=1., min_size=10)
    #     sub_graphs[i].draw_multi_scale_community(community=smc, cmap=Pastel_10, inline=True, title='s%d-kde-memory-louvain' % i)
    # gmg.draw_multi_scale_community(community=mc, cmap=Pastel_10, inline=True, title='none-none-r0')
    # plt = PlotView(column_num=1, row_num=1, title='extrusion-kde-memory')
    # plt[0].name = 'none-none-r0'
    # map = MapBox(name='map_infomap',
    #              pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
    #              lon=116.37363,
    #              lat=39.915606,
    #              style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
    #              pitch=55,
    #              bearing=0,
    #              zoom=12,
    #              viewport=plt[0])
    # gmg.draw_taz(map_view=map)
    # gmg.draw_multi_community_extrusion(map_view=map,
    #                                    communities=mc,
    #                                    cmap=Pastel_10,
    #                                    title='nnr0')
    # plt.plot(inline=True)
    # mc = gmg.community_detection_multi_infomap(geo_weight='queen', connect='all_connect')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='queen-all')
    # mc = gmg.community_detection_multi_infomap(geo_weight='kde', connect='flow')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='kde-flow')
    # mc = gmg.community_detection_multi_infomap(geo_weight='kde', connect='memory')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='kde-memory')
    # mc = gmg.community_detection_multi_infomap(geo_weight='kde', connect='all_connect')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='kde-all')
    # mc = gmg.community_detection_multi_infomap(geo_weight='queen2', connect='flow')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='queen2-flow')
    # mc = gmg.community_detection_multi_infomap(geo_weight='queen2', connect='memory')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='queen2-memory')
    # mc = gmg.community_detection_multi_infomap(geo_weight='queen2', connect='all_connect')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='queen2-all')
    # mc = gmg.community_detection_multi_infomap(geo_weight='knn', connect='flow')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='knn-flow')
    # mc = gmg.community_detection_multi_infomap(geo_weight='knn', connect='memory')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='knn-memory')
    # mc = gmg.community_detection_multi_infomap(geo_weight='knn', connect='all_connect')
    # gmg.draw_multi_scale_community(community=[mc[mc['layer_id'] == i] for i in range(6)], cmap=Pastel_10, inline=True,
    #                                title='knn-all')
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
    # cl = gmg.community_detection_louvain(min_size=20, resolution=1.)
    # gmg.draw_multi_scale_community(community=cl, cmap=Set3_12, inline=False, title='Louvain')
    # cl = gmg.community_detection_infomap(min_size=10)
    # gmg.draw_multi_scale_community(community=cl, cmap=Pastel_10, inline=False, title='Info-Map')
    # sub_graphs = [gmg.sub_graph(cl[cl['community'] == i]['tazid'].unique()) for i in cl['community'].unique()]
    # for i in range(len(sub_graphs)):
    #     # smc = sub_graphs[0].community_detection_infomap(min_size=10)
    #     # smc = sub_graphs[i].community_detection_multi_infomap(geo_weight='kde', connect='memory', only_self_transition=False)
    #     smc = sub_graphs[0].community_detection_louvain(resolution=1., min_size=10)
    #     sub_graphs[i].draw_multi_scale_community(community=smc, cmap=Pastel_10, inline=True, title='s%d-kde-memory-louvain' % i)
    # gmg.draw_multi_scale_community(community=cl, cmap=Pastel_10, inline=False)
    # gmg.draw_network(color='white', width=1., value='weight', bk=True, inline=False, row=3, column=2)
