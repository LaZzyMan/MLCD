import json
from pywebplot import *
import time


class Layer(object):
    def __init__(self, id, type, source='', maxzoom=24, minzoom=0, **kwargs):
        '''
        create a layer
        :param id:
        :param type:
        :param source:
        :param maxzoom:
        :param minzoom:
        :param kwargs: paint and layout params use '_' instead of '-'. Start with p_ for paints l_ for layouts.
        '''
        self._id = id
        self._type = type
        self._source = source
        self._paint = {}
        self._layout = {}
        self._maxzoom = maxzoom
        self._minzoom = minzoom
        for key, item in kwargs.items():
            if key[:2] == 'p_':
                self._paint[key[2:].replace('_', '-')] = item
            if key[:2] == 'l_':
                self._layout[key[2:].replace('_', '-')] = item

    @property
    def script(self):
        return '(%s);' % json.dumps(self.__dict__).replace('_', '')

    @property
    def paint(self):
        return self._paint

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return self._id


class LineLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a line layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, 'line', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class BackgroundLayer(Layer):
    def __init__(self, id, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a line layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, type='background', maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class FillLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a fill layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, 'fill', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class FillExtrusionLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        :param id:
        :param source:
        :param maxzoom:
        :param minzoom:
        :param kwargs:
        '''
        super().__init__(id, 'fill-extrusion', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class SymbolLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a line layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, 'symbol', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class RasterLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a line layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, 'raster', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class CircleLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a line layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, 'circle', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class HeatMapLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a line layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, 'heatmap', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class HillShadeLayer(Layer):
    def __init__(self, id, source, maxzoom=24, minzoom=0, **kwargs):
        '''
        create a line layer
        :param id:
        :param source:
        :param kwargs: paint params
        '''
        super().__init__(id, 'hillshade', source, maxzoom=maxzoom, minzoom=minzoom, **kwargs)


class Source(object):
    def __init__(self, id, type, data=''):
        self._data = data
        self._id = id
        self._script = ''
        self._type = type

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def name(self):
        return self._id

    @property
    def script(self):
        return self._script


class GeojsonSource(Source):
    def __init__(self, id, data=''):
        super().__init__(id, type='geojson', data=data)
        self._script = "('%s', {type: '%s', data: '%s'});" % (self._id, self._type, self._data)


class VideoSource(Source):
    pass


class VectorSource(Source):
    pass


class RasterSource(Source):
    pass


class ImageSource(Source):
    pass


class CanvasSource(Source):
    pass


class Event(object):
    def __init__(self, type, event, listener, layer=None):
        '''

        :param type: on, off or once
        :param event: event type
        :param layer: specific layer of the map
        :param listener: function object
        '''
        self._type = type
        self._event = event
        self._listener = listener
        self._layer = layer

    @property
    def script(self):
        if self._layer is None:
            return "%s('%s', %s)" % (self._type, self._event, self._listener)
        else:
            return "%s('%s', '%s', %s)" % (self._type, self._event, self._layer, self._listener)


class MapBox(object):
    def __init__(self, viewport, pk, style, name='map', lon=116.37363, lat=39.915606, pitch=0, bearing=0, zoom=0):
        timestamp = str(int(time.time()))
        self._name = name
        self._viewport = viewport
        self._style = style
        self._pk = pk
        self._center = [lon, lat]
        self._bearing = bearing
        self._pitch = pitch
        self._zoom = zoom
        self._source = {}
        self._layer = {}
        self._event = []
        self._dir_js = 'dist/js/%s.js' % (self._name + timestamp)
        self._viewport.plv.add_js('%s.js' % (self._name + timestamp))

    @property
    def viewport(self):
        return self._viewport

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, value):
        self._style = value

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        if 0 <= value <= 24:
            self._style = value

    @property
    def bearing(self):
        return self._bearing

    @bearing.setter
    def bearing(self, value):
        if 0 <= value <= 180:
            self._bearing = value

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value

    @property
    def center(self):
        return self._pitch

    @center.setter
    def center(self, value):
        lon = value[0]
        lat = value[1]
        self._center = [lon, lat]

    @staticmethod
    def js_function(content, **kwargs):
        param_script = ''
        for key, item in kwargs:
            if isinstance(item, str):
                param_script += "%s='%s', " % (key, item)
            else:
                param_script += "%s=%s, " % (key, item)
        return 'function (%s){%s}' % (param_script, content)

    @property
    def load(self):
        source_script = ''
        layer_script = ''
        for _, item in self._source.items():
            source_script += '%s.addSource%s\n' % (self._name, item.script)
        for _, item in self._layer.items():
            layer_script += '%s.addLayer%s\n' % (self._name, item.script)
        return Event(type='on', event='load', listener=self.js_function(content=source_script+layer_script))

    @property
    def script(self):
        init_script = '''
            mapboxgl.accessToken = '%s';
            const %s = new mapboxgl.Map({
                container: '%s',
                style: '%s',
                center: [%f, %f],
                pitch: %f,
                zoom: %f,
                bearing: %f
            });
            ''' % (self._pk, self._name, self._viewport.name, self._style, self._center[0], self._center[1], self._pitch, self._zoom, self._bearing)
        init_script += '%s.%s' % (self._name, self.load.script)
        for event in self._event:
            init_script += '%s.%s' % (self._name, event.script)
        return init_script

    def add_source(self, source):
        '''
        add source for map
        :param source:
        :return:
        '''
        if source.name in self._source.keys():
            print('Name existed!')
            return
        self._source[source.name] = source

    def add_layer(self, layer):
        '''
        add layer for map
        :param layer:
        :return:
        '''
        if layer.name in self._layer.keys():
            print('Name existed!')
            return
        self._layer[layer.name] = layer

    def add_event(self, event):
        '''
        add event for map
        :param event:
        :return:
        '''
        self._event.append(event)

    def update(self):
        with open(self._dir_js, 'w') as f:
            f.write(self.script)
            f.close()


if __name__ == '__main__':
    plt = PlotView(column_num=1, row_num=1, title='MapBox')
    plt[0, 0].name = 'network'
    mb = MapBox(name='map',
                pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                lon=116.37363,
                lat=39.915606,
                style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                pitch=55,
                bearing=0,
                zoom=12,
                viewport=plt[0, 0])
    network_source = GeojsonSource(id='network', data='network_0.geojson')
    taz_source = GeojsonSource(id='taz', data='taz.geojson')
    mb.add_source(network_source)
    mb.add_source(taz_source)
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
    mb.add_layer(bk_layer)
    mb.add_layer(taz_layer)
    mb.add_layer(network_layer)
    mb.update()
    plt.plot()
