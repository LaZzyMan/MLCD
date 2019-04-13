from pywebplot import *
from palettable.colorbrewer.diverging import Spectral_10
from GeoMultiGraph import GeoMultiGraph

if __name__ == '__main__':
    mkdir()
    gmg = GeoMultiGraph()
    gmg.load('GeoMultiGraph_week', generate_nx=True)
    # gmg.threshold(t_min=3, generate_nx=True)
    # gmg.transform(func='sqrt', generate_nx=True)
    # gmg.draw_dist(hist=True, kde=False, rug=False, bins=20)
    # gmg.draw_qq_plot()
    cl = gmg.community_detection_louvain(min_size=20, resolution=2.)
    gmg.draw_multi_scale_community(community=cl, cmap=Spectral_10, inline=False)
