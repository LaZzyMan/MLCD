
            mapboxgl.accessToken = 'pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ';
            const map = new mapboxgl.Map({
                container: 'network',
                style: 'mapbox://styles/hideinme/cjtgp37qv0kjj1fup07b9lf87',
                center: [116.373630, 39.915606],
                pitch: 55.000000,
                zoom: 12.000000,
                bearing: 0.000000
            });
            map.on('load', function (){map.addSource('network', {type: 'geojson', data: 'data/network_0.geojson'});
map.addSource('taz', {type: 'geojson', data: 'data/taz.geojson'});
map.addLayer({"id": "bk", "type": "background", "source": "", "paint": {"background-opacity": 0.7, "background-color": "#000000"}, "layout": {}, "maxzoom": 24, "minzoom": 0});
map.addLayer({"id": "taz", "type": "fill", "source": "taz", "paint": {"fill-color": "#3BA1C3", "fill-opacity": 0.1}, "layout": {}, "maxzoom": 24, "minzoom": 0});
map.addLayer({"id": "network", "type": "line", "source": "network", "paint": {"line-color": "white", "line-width": 1.0, "line-opacity": ["get", "opacity"]}, "layout": {}, "maxzoom": 24, "minzoom": 0});
})