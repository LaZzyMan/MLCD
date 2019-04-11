from pywebplot import *
import webbrowser
import os
from jinja2 import Template


class PlotView(object):
    def __init__(self, column_num=1, row_num=1, title='index'):
        '''
        create a plot view(html page)
        :param column_num: subview number per column
        :param row_num: subview number per row
        '''
        self._title = title
        self._column_num = column_num
        self._row_num = row_num
        width_subview = 100.0 / self._column_num
        height_subview = 100.0 / self._row_num
        self._subview = [SubView(width=width_subview, height=height_subview, name='subview-%d' % i, plv=self)
                         for i in range(self._column_num * self._row_num)]
        self._dir_html = 'src/%s.html' % title.lower().replace(' ', '_')
        self._dir_js = 'src/js/%s.js' % title.lower().replace(' ', '_')
        self._js = []
        if os.path.exists(self._dir_html):
            os.remove(self._dir_html)
        if os.path.exists(self._dir_js):
            os.remove(self._dir_js)

    def __getitem__(self, index):
        '''
        get subview by index
        :param index: int or tuple
        :return:
        '''
        if isinstance(index, int):
            return self._subview[index]
        elif isinstance(index, tuple):
            return self._subview[index[0] * self._row_num + index[1]]

    @property
    def dom(self):
        s = ''
        for i in range(self._column_num):
            s += '<div class="vertical-split" style="width: %fvw; height: 100vh">' % (100.0 / self._column_num)
            for j in range(self._row_num):
                s += self[i, j].dom
            s += '</div>'
        return s

    @property
    def title(self):
        return self._title

    @property
    def dir_js(self):
        return self._dir_js

    @property
    def dir_html(self):
        return self._dir_html

    @title.setter
    def title(self, value):
        self._title = value

    def add_js(self, url):
        self._js.append('<script src="%s"></script>' % url)

    def plot(self):
        html = '''
        <!DOCTYPE html>
        <html lang="zh-CN">
            <head>
                <meta charset="utf-8">
                <title>{{ title }}</title>
                <script src='https://api.tiles.mapbox.com/mapbox-gl-js/v0.53.1/mapbox-gl.js'></script>
                <script src="https://d3js.org/d3.v5.min.js"></script>
                <link href='https://api.tiles.mapbox.com/mapbox-gl-js/v0.53.1/mapbox-gl.css' rel='stylesheet' />
            </head>
            <body>
                <div class="plot-view">{{ dom }}</div>
                <link href='style/index.css' rel='stylesheet' />
                {% for link in js %}
                {{ link }}
                {% endfor %}
            </body>
            <style>body{ margin:0; padding:0; }</style>
        </html>
        '''
        template = Template(html)
        with open(self._dir_html, 'w') as f:
            f.write(template.render(title=self._title, js=self._js, dom=self.dom))
            f.close()
        webbrowser.open(self._dir_html)


class SubView(object):
    def __init__(self, width, height, plv, name=''):
        '''
        create a subview
        :param width: width relate to father dom
        :param height: height relate to father dom
        :param name: identifier for subview
        '''
        self._width = width
        self._height = height
        self._name = name
        self._plv = plv

    @property
    def dom(self):
        return '''
        <div class="sub-view" id="%s" style="width: 100%%; height: %f%%"></div>
        ''' % (self._name, self._height)

    @property
    def name(self):
        return self._name

    @property
    def plv(self):
        return self._plv

    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value
