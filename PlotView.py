from SubView import SubView
import webbrowser


class PlotView(object):
    def __init__(self, column_num=1, row_num=1, title='Plot View'):
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
        self._subview = [SubView(width=width_subview, height=height_subview) for _ in range(self._column_num * self._row_num)]

    def __getitem__(self, index):
        '''
        get subview by index
        :param index: int or tuple
        :return:
        '''
        if isinstance(index, int):
            return self._subview[index]
        elif isinstance(index, tuple):
            return self._subview[index[0] * self._column_num + index[1]]

    @property
    def dom(self):
        s = ''
        for i in range(self._column_num):
            s += '<div class="vertical-split" style="width: %f%%; height: 100%%">' % (100.0 / self._column_num)
            for j in range(self._row_num):
                s += self[i, j].dom
            s += '</div>'
        return s

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    def plot(self):
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
                %s
                <script src='js/index.js'></script>
                <link href='style/index.css' rel='stylesheet' />
            </body>
            <style>body{ margin:0; padding:0; }</style>
        </html>
        ''' % (self._title, self.dom)
        with open('src/index.html', 'w') as f:
            f.write(html)
            f.close()
        webbrowser.open('src/index.html')
