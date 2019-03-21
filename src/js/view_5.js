
            mapboxgl.accessToken = 'pk.eyJ1IjoiaGlkZWlubWUiLCJhIjoiY2o4MXB3eWpvNnEzZzJ3cnI4Z3hzZjFzdSJ9.FIWmaUbuuwT2Jl3OcBx1aQ';
            const map_5 = new mapboxgl.Map({
                container: 'map-5',
                style: 'mapbox://styles/hideinme/cjhaowf2w176k2sr3qw1l0l8n',
                center: [116.373630, 39.915606],
                pitch: 0.000000,
                zoom: 12.000000,
                bearing: 0.000000
            });
            
        map_5.on('load', function(){

        
        map_5.addSource('taz', {
            type: 'geojson',
            data: 'taz.geojson'
            });
            
        
        map_5.addSource('network_0', {
            type: 'geojson',
            data: 'network_5.geojson'
            });
            
        
        map_5.addLayer(
            {
                'type': 'background',
                'paint': {'background-color': '#000000', 'background-opacity': 1.0},
                'id': 'bk'
            }
        )
        
        map_5.addLayer(
            {
                'source': 'network_0',
                'type': 'line',
                'paint': {'line-color': 'white', 'line-width': 1.5, 'line-opacity': ['get', 'opacity']},
                'id': 'network_0'
            }
        )
        });