class Layer(object):
    pass


class LineLayer(Layer):
    pass


class Source(object):
    pass


class GeojsonSource(Source):
    pass


class MapBox(object):
    def __init__(self, viewport, pk, style, title='MapBox', lon=116.37363, lat=39.915606, pitch=0, bearing=0, zoom=0):
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
        self._map_on_load = '''
        map.on('load', function(){})
        '''

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
        if 0 <= value <= 14:
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
            const map = new mapboxgl.Map({
                container: 'map',
                style: '%s',
                center: [%f, %f],
                pitch: %f,
                zoom: %f,
                bearing: %f
            });
            ''' % (self._pk, self._style, self._center[0], self._center[1], self._pitch, self._zoom, self._bearing)

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

    def add_layer(self, name, source=None, type='background', paint={}):
        if name in self.layer.keys():
            print('Name existed!')
            return
        self.layer[name] = {
            'source': source,
            'type': type,
            'paint': paint
        }

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

    def add_geojson_source(self, geojson, name):
        '''
        add geojson source for map
        :param geojson: dir of geojson file
        :param name: source name
        :return:
        '''
        if name in self.source.keys():
            print('Name existed!')
            return
        self.source[name] = geojson

    def add_video_source(self):
        pass

    def add_image_source(self):
        pass

    def add_canvas_source(self):
        pass

    def set_data(self, data, name):
        '''
        set data for source by id
        :param geojson: dir of geojson file
        :param name: source name
        :return:
        '''
        self.source[name] = data

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
