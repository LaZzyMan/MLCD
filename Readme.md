## Example
Use following code to create a map with layers.
```python
from pywebplot import *
# create a view with 1*1 sub views
plt = PlotView(column_num=1, row_num=1, title='MapBox')
plt[0, 0].name = 'network'
# create map on subview
mb = MapBox(name='map',
            pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
            lon=116.37363,
            lat=39.915606,
            style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
             pitch=55,
             bearing=0,
             zoom=12,
             viewport=plt[0, 0])
# create sources from data
network_source = GeojsonSource(id='network', data='data/network_0.geojson')
taz_source = GeojsonSource(id='taz', data='data/taz.geojson')
# add source to map
mb.add_source(network_source)
mb.add_source(taz_source)
# create layers with source and render params
bk_layer = BackgroundLayer(id='bk',
                           p_background_opacity=0.7,
                           p_background_color='#000000')
taz_layer = FillLayer(id='taz',
                          source='taz',
                          p_fill_color='#3BA1C3',
                          p_fill_opacity=0.1)
network_layer = LineLayer(id='network',
                          source='network',
                          p_line_color='white',
                          p_line_width=1.0,
                          p_line_opacity=['get', 'opacity'])
# add layers to map
mb.add_layer(bk_layer)
mb.add_layer(taz_layer)
mb.add_layer(network_layer)
# show in web browser
mb.update()
plt.plot() 
```
Result:
![Result](https://github.com/LaZzyMan/pywebplot/blob/master/example.png)