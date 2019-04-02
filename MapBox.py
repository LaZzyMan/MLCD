class Layer(object):
    def __init__(self, id, type, source='', maxzoom=24, minzoom=0):
        self._id = id
        self._type = type
        self._source = source
        self._script = ''
        self._paint = ''
        self._layout = ''

    @property
    def script(self):
        return self._script

    @def paint(self):
        return self._paint

    @property
    def name(self):
        return self._id


class LineLayer(Layer):
    def __init__(self, id, source,
                 line_cap='butt',
                 line_join='miter',
                 line_miter_limit=2,
                 line_round_limit=1.05,
                 visibility='visible',
                 line_opacity=1,
                 line_color='#000000',
                 line_translate=[0, 0],
                 line_translate_anchor='map',
                 line_width=1,
                 line_gap_width=0,):
        super().__init__()
        pass


class BackgroundLayer(Layer):
    pass


class FillLayer(Layer):
    pass


class SymbolLayer(Layer):
    pass


class RasterLayer(Layer):
    pass


class CircleLayer(Layer):
    pass


class HeatMapLayer(Layer):
    pass


class HillShadeLayer(Layer):
    pass


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
        self._script = '''
        addSource('%s', {type: '%s', data: '%s'});  
        ''' % (self._id, self._type, self._data)


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


class MapBox(object):
    def __init__(self, viewport, pk, style, title='MapBox', lon=116.37363, lat=39.915606, pitch=0, bearing=0, zoom=0):
        self._name = title
        self._viewport = viewport
        self._style = style
        # self.style = 'mapbox://styles/hideinme/cjjo0icb95w172slnj93q6y31'
        self._pk = pk
        self._center = [lon, lat]
        self._bearing = bearing
        self._pitch = pitch
        self._zoom = zoom
        self._source = {}
        self._layer = {}
        self._viewport.name = title

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

    @property
    def script(self):
        return '''
            mapboxgl.accessToken = '%s';
            const %s = new mapboxgl.Map({
                container: 'map',
                style: '%s',
                center: [%f, %f],
                pitch: %f,
                zoom: %f,
                bearing: %f
            });
            ''' % (self._name, self._pk, self._style, self._center[0], self._center[1], self._pitch, self._zoom, self._bearing)

    def load(self):
        '''
        add listener for map on load
        :return:
        '''
        load_code = '''
        map.on('load', function(){\n
        '''
        for source_id, source in self.source.items():
            load_code += self.transform_source(source_id, source)
        for layer_id, layer in self.layer.items():
            load_code += self.transform_layer(layer_id, layer)
        load_code += '});'
        with open(self.src_dir + self.index_js, 'a') as f:
            f.write(load_code)
            f.close()

    @staticmethod
    def transform_layer(name, layer):
        return '''
        map.addLayer(
            {
                'source': '%s',
                'type': '%s',
                'paint': %s,
                'id': '%s'
            }
        )
        ''' % (layer['source'], layer['type'], str(layer['paint']), name)

    @staticmethod
    def transform_source(name, source):
        return '''
        map.addSource('%s', {
            type: 'geojson',
            data: '%s'
            });
            
        ''' % (name, source)

    def add_source(self, source):
        '''
        add geojson source for map
        :param source:
        :return:
        '''
        if source.name in self._source.keys():
            print('Name existed!')
            return
        self._source[source.name] = source

    def add_layer(self, name, source=None, type='background', paint={}):
        if name in self.layer.keys():
            print('Name existed!')
            return
        self.layer[name] = {
            'source': source,
            'type': type,
            'paint': paint
        }

    def show(self):
        '''
        open web page
        :return:
        '''
        self.load()
        webbrowser.open(self.src_dir + self.index_html)


if __name__ == '__main__':
    mb = MapBox(title='Network',
                pk='pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ',
                lon=116.37363,
                lat=39.915606,
                style='mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                pitch=55,
                bearing=0,
                zoom=12)
    mb.add_geojson_source(geojson='network_0.geojson', name='network_0')
    mb.add_layer(name='network_0', source='network_0', type='line',
                 paint={'line-color': 'white', 'line-width': 0.3, 'line-opacity': ['get', 'opacity']})
    mb.show()
