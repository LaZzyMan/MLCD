from pywebplot import *
from palettable.colorbrewer.qualitative import Set3_12
from GeoMultiGraph import GeoMultiGraph

if __name__ == '__main__':
    mkdir()
    gmg = GeoMultiGraph()
    gmg.load('../src/data/GeoMultiGraph_week', network_list=['2012', '2013', '2014', '2015', '2016', '2017'])
    gmg.threshold(t_min=10, generate_nx=True)
    # gmg.transform(func='sqrt', generate_nx=True)
    # gmg.draw_dist(hist=True, kde=False, rug=False, bins=20)
    # gmg.draw_qq_plot()
    # cl = gmg.community_detection_louvain(min_size=20, resolution=1.)
    # gmg.draw_multi_scale_community(community=cl, cmap=Set3_12, inline=False)
    gmg.draw_network(color='white', width=1., value='weight', bk=True, inline=False, row=3, column=2)
