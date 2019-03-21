import webbrowser


class MapBox:
    def __init__(self, pk, style, title='MapBox', lon=116.37363, lat=39.915606, pitch=0, bearing=0, zoom=0):
        self.src_dir = 'src/'
        self.index_html = 'index.html'
        self.index_js = 'js/index.js'
        self.style = style
        # self.style = 'mapbox://styles/hideinme/cjjo0icb95w172slnj93q6y31'
        self.pk = pk
        self.center = [lon, lat]
        self.bearing = bearing
        self.pitch = pitch
        self.zoom = zoom
        self.source = {}
        self.layer = {}
        self.map_on_load = '''
        map.on('load', function(){})
        '''
        with open(self.src_dir + self.index_html, 'w') as f:
            html = '''
            <!DOCTYPE html>
            <html lang="zh-CN">
                <head>
                    <meta charset="utf-8">
                    <title>%s</title>
                    <script src='https://api.tiles.mapbox.com/mapbox-gl-js/v0.53.1/mapbox-gl.js'></script>
                    <link href='https://api.tiles.mapbox.com/mapbox-gl-js/v0.53.1/mapbox-gl.css' rel='stylesheet' />
                </head>
                <body>
                    <div id="map" style="width: 100vw; height: 100vh"></div>
                    <script src='%s'></script>
                </body>
                <style>body{ margin:0; padding:0; }</style>
            </html>
            ''' % (title, self.index_js)
            f.write(html)
            f.close()
        with open(self.src_dir + self.index_js, 'w') as f:
            js = '''
            mapboxgl.accessToken = '%s';
            const map = new mapboxgl.Map({
                container: 'map',
                style: '%s',
                center: [%f, %f],
                pitch: %f,
                zoom: %f,
                bearing: %f
            });
            ''' % (self.pk, self.style, self.center[0], self.center[1], self.pitch, self.zoom, self.bearing)
            f.write(js)
            f.close()

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
