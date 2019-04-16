
# Example for Geo-Multilayer Network

GeoMultiGraph is a model for multilayer graph with geo-reference for each node. Some analysis tools and plot interface have been provided so far and there can be more functions in the future.

The data here is in 'pywebplot/src/data/GeoMultiGraph.npy' which is an OD taxi trajectory in Beijing for six years and the region segement data can be found in 'pywebplot/src/data/taz.geojson'. That create a multilayer graph with 1371 nodes and 6 layers.


```python
import sys
sys.path.append("..")
from pywebplot import *
from GeoMultiGraph import *
from palettable.colorbrewer.qualitative import Set3_12
from IPython.display import HTML
mkdir()
gmg = GeoMultiGraph()
gmg.load('../src/data/GeoMultiGraph_week', generate_nx=True, network_list=['2012', '2013', '2014', '2015', '2016', '2017'])
```

    Using numpy backend.
    /Users/zz/anaconda3/lib/python3.7/site-packages/pysal/__init__.py:65: VisibleDeprecationWarning: PySAL's API will be changed on 2018-12-31. The last release made with this API is version 1.14.4. A preview of the next API version is provided in the `pysal` 2.0 prelease candidate. The API changes and a guide on how to change imports is provided at https://pysal.org/about
      ), VisibleDeprecationWarning)


    Generating Network 2012
    Generating Network 2013
    Generating Network 2014
    Generating Network 2015
    Generating Network 2016
    Generating Network 2017


The distribution of weight on edges can be displayed by dist-plot, qq-plot, cdf and ccdf.

In general, the distribution subordinates to power law distribution for a reality dataset, which can be seen in the following figures.


```python
gmg.draw_dist(hist=True, kde=True, bins=20)
```

    Generating Edges Data Frame...
    Finished.


    /Users/zz/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](https://github.com/LaZzyMan/pywebplot/tree/master/src/example/output_3_2.png)



```python
gmg.draw_qq_plot()
```

    Generating Edges Data Frame...
    Finished.



![png](https://github.com/LaZzyMan/pywebplot/tree/master/src/example/output_4_1.png)



```python
gmg.draw_ccdf()
```

    Generating Edges Data Frame...
    Finished.



![png](https://github.com/LaZzyMan/pywebplot/tree/master/src/example/output_5_1.png)



```python
gmg.draw_cdf()
```

    Generating Edges Data Frame...
    Finished.



![png](https://github.com/LaZzyMan/pywebplot/tree/master/src/example/output_6_1.png)


If the data does not meet expectations or needs to be adjusted, you can use 'threshold' and 'transform' to fix it.


```python
gmg.threshold(t_min=3, generate_nx=False)
gmg.transform(func='log', generate_nx=True)
gmg.draw_dist(hist=True, kde=True, bins=20)
```

    Generating Network 2012
    Generating Network 2013
    Generating Network 2014
    Generating Network 2015
    Generating Network 2016
    Generating Network 2017
    Generating Edges Data Frame...
    Finished.


    /Users/zz/anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](https://github.com/LaZzyMan/pywebplot/tree/master/src/example/output_8_2.png)


The basic distribution can be displayed by network visualization and some low-weight trajectories are removed to ensure image clarity.


```python
gmg.threshold(t_min=20, generate_nx=True)
gmg.draw_network(color='white', width=2., value='weight', bk=True, inline=True, row=3, column=2)
```

    Generating Network 2012
    Generating Network 2013
    Generating Network 2014
    Generating Network 2015
    Generating Network 2016
    Generating Network 2017
    Generating Geo Edges Data Frame...
    Finished.
    Drawing Network-2012...
    Finished.
    Drawing Network-2013...
    Finished.
    Drawing Network-2014...
    Finished.
    Drawing Network-2015...
    Finished.
    Drawing Network-2016...
    Finished.
    Drawing Network-2017...
    Finished.
    Network: http://localhost:4397/network.html
    Starting server, listen at: http://localhost:4397



```python
HTML('<iframe src="http://localhost:4396/network.html", width=1000, height=600></iframe>')
```




<iframe src="http://localhost:4396/network.html", width=1000, height=600></iframe>



The hotspot of the city can be displyed by calculate the centrality(for example, closeness centrality), degree or cluster coefficient. And they can be easily plotted on map.


```python
from palettable.colorbrewer.sequential import Reds_9
degree = gmg.in_degree[0]
plt = PlotView(column_num=1, row_num=1, title='In-Degree')
plt[0].name = 'In-Degree'
map = MapBox(name='map_si_0',
                  pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                  lon=116.37363,
                  lat=39.915606,
                  style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                  pitch=55,
                  bearing=0,
                  zoom=12,
                  viewport=plt[0])
gmg.draw_choropleth_map(map_view=map, data=degree, value='in_degree', title='indegree', cmap=Reds_9)
plt.plot(inline=True)
```

    In Degree for 2012...
    In Degree for 2013...
    In Degree for 2014...
    In Degree for 2015...
    In Degree for 2016...
    In Degree for 2017...
    Finished.
    In-Degree: http://localhost:4396/in-degree.html
    Starting server, listen at: http://localhost:4396



```python
HTML('<iframe src="http://localhost:4396/in-degree.html", width=1000, height=600></iframe>')
```




<iframe src="http://localhost:4396/in-degree.html", width=1000, height=600></iframe>



Community discovery algorithms can be used to hierarchically cluster sub-regions to show the structural characteristics of the city (the multi-layer network discovery algorithm based on infomap will be joined soon.)

Using the Louvain:


```python
from palettable.colorbrewer.sequential import Reds_9, GnBu_9, BuPu_9, Blues_9
closeness_centrality = get_closeness_centrality(gmg.nx_graph[0])
degree_centrality = get_degree_centrality(gmg.nx_graph[0])
eigenvector_centrality = get_eigenvector_centrality(gmg.nx_graph[0])
cluster_coefficient = get_cluster_coefficient(gmg.nx_graph[0])
plt = PlotView(column_num=2, row_num=2, title='Statistics-Information')
plt[0, 0].name = 'Closeness-Centrality'
plt[1, 0].name = 'Degree-centrality'
plt[1, 1].name = 'Eigenvector-centrality'
plt[0, 1].name = 'Cluster-Coefficient'
maps = [MapBox(name='map-si-%i' % i,
                  pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                  lon=116.37363,
                  lat=39.915606,
                  style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                  pitch=55,
                  bearing=0,
                  zoom=12,
                  viewport=plt[i]) for i in range(4)]
gmg.draw_choropleth_map(map_view=maps[0], data=closeness_centrality, value='closeness_centrality', title='closeness_centrality', cmap=Reds_9)
gmg.draw_choropleth_map(map_view=maps[1], data=in_degree, value='degree_centrality', title='degree_centrality', cmap=GnBu_9)
gmg.draw_choropleth_map(map_view=maps[2], data=out_degree, value='eigenvector_centrality', title='eigenvector_centrality', cmap=BuPu_9)
gmg.draw_choropleth_map(map_view=maps[3], data=cluster_coefficient, value='cluster_coefficient', title='cluster_coefficient', cmap=Blues_9)
plt.plot(inline=True)
```


```python
cl = gmg.community_detection_louvain(min_size=20, resolution=1.)
gmg.draw_multi_scale_community(community=cl, cmap=Set3_12, inline=True, title='Louvain')
```

    Louvain for network 2012...
    Network 2012 Modularity: 0.352558.
    Finished 2012, 4 communities found, 38 point unclassified.
    Louvain for network 2013...
    Network 2013 Modularity: 0.365434.
    Finished 2013, 6 communities found, 37 point unclassified.
    Louvain for network 2014...
    Network 2014 Modularity: 0.383300.
    Finished 2014, 5 communities found, 34 point unclassified.
    Louvain for network 2015...
    Network 2015 Modularity: 0.396568.
    Finished 2015, 9 communities found, 19 point unclassified.
    Louvain for network 2016...
    Network 2016 Modularity: 0.411507.
    Finished 2016, 7 communities found, 10 point unclassified.
    Louvain for network 2017...
    Network 2017 Modularity: 0.394334.
    Finished 2017, 7 communities found, 14 point unclassified.
    Louvain: http://localhost:4396/louvain.html
    Starting server, listen at: http://localhost:4396



```python
HTML('<iframe src="http://localhost:4396/louvain.html", width=1000, height=600></iframe>')
```




<iframe src="http://localhost:4396/louvain.html", width=1000, height=600></iframe>



Using Infomap:


```python
cl = gmg.community_detection_infomap(min_size=20)
gmg.draw_multi_scale_community(community=cl, cmap=Set3_12, inline=False, title='Info-Map')
```

    Found 11 top modules with codelength: 9.302540
    Finished 2012, 9 communities found, 3 point unclassified.
    Found 12 top modules with codelength: 9.043862
    Finished 2013, 10 communities found, 4 point unclassified.
    Found 10 top modules with codelength: 9.156855
    Finished 2014, 10 communities found, 0 point unclassified.
    Found 10 top modules with codelength: 8.963506
    Finished 2015, 9 communities found, 9 point unclassified.
    Found 12 top modules with codelength: 8.960712
    Finished 2016, 11 communities found, 19 point unclassified.
    Found 12 top modules with codelength: 9.031458
    Finished 2017, 10 communities found, 38 point unclassified.
    Info-Map: http://localhost:4396/info-map.html
    Starting server, listen at: http://localhost:4396



```python
HTML('<iframe src="http://localhost:4396/info-map.html", width=1000, height=600></iframe>')
```




<iframe src="http://localhost:4396/info-map.html", width=1000, height=600></iframe>




```python

```
