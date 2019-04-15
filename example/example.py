from pywebplot import *
from palettable.colorbrewer.qualitative import Set3_12
from GeoMultiGraph import *
from palettable.colorbrewer.sequential import Reds_9, GnBu_9, BuPu_9, Blues_9


if __name__ == '__main__':
    mkdir()
    gmg = GeoMultiGraph()
    gmg.load('../src/data/GeoMultiGraph_week', network_list=['2012', '2013', '2014', '2015', '2016', '2017'], generate_nx=True)
    closeness_centrality = get_closeness_centrality(gmg.nx_graph[0])
    plt = PlotView(column_num=1, row_num=1, title='Statistics-Information')
    plt[0].name = 'closeness_centrality'
    map = MapBox(name='map_si_0',
                 pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                 lon=116.37363,
                 lat=39.915606,
                 style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                 pitch=55,
                 bearing=0,
                 zoom=12,
                 viewport=plt[0])
    gmg.draw_choropleth_map(map_view=map, data=closeness_centrality, value='closeness_centrality',
                            title='closeness-centrality', cmap=Reds_9)
    plt.plot(inline=False)
    # gmg.threshold(t_min=20, generate_nx=True)
    # gmg.transform(func='sqrt', generate_nx=True)
    # gmg.draw_dist(hist=True, kde=False, rug=False, bins=20)
    # gmg.draw_qq_plot()
    # cl = gmg.community_detection_louvain(min_size=20, resolution=1.)
    # gmg.draw_multi_scale_community(community=cl, cmap=Set3_12, inline=False, title='Louvain')
    # cl = gmg.community_detection_infomap(min_size=20)
    # gmg.draw_multi_scale_community(community=cl, cmap=Set3_12, inline=False, title='Info-Map')
    # gmg.draw_multi_scale_community(community=cl, cmap=Set3_12, inline=False)
    # gmg.draw_network(color='white', width=1., value='weight', bk=True, inline=False, row=3, column=2)
