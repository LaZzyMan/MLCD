from jinja2 import Template
from pywebplot.utils import *
from palettable.colorbrewer.qualitative import Set3_12, Set2_8, Set1_9, Pastel2_8, Pastel1_9, Paired_12, Dark2_8, Accent_8
from palettable.colorbrewer.sequential import Blues_9, BuGn_9, BuPu_9, GnBu_9, Greens_9, Greys_9, OrRd_9, Oranges_9, PuBu_9, PuBuGn_9, PuRd_9, Purples_9, RdPu_9, Reds_9, YlGn_9,  YlGnBu_9, YlOrBr_9, YlOrRd_9

QUALITATIVE = [Set3_12, Set2_8, Set1_9, Pastel2_8, Pastel1_9, Paired_12, Dark2_8, Accent_8]
SEQUENTIAL = [Blues_9, Reds_9, Greens_9, Oranges_9, Greys_9, Purples_9, BuGn_9, BuPu_9, GnBu_9, OrRd_9, PuBu_9, PuBuGn_9, PuRd_9, RdPu_9, YlGn_9, YlGnBu_9, YlOrBr_9, YlOrRd_9]


class MultiGradColorMap(object):
    def __init__(self, num_color, grad_value):
        super().__init__()
        self._num_colors = num_color
        self._grad_value = grad_value
        colors = []
        for c_map in SEQUENTIAL:
            colors.append(c_map.mpl_colormap)
        for i in range(int(num_color / len(SEQUENTIAL))):
            colors.extend(colors)
        self._c_map = colors

    def get_rgb_color(self, n, v):
        return [int(i * 255) for i in self._c_map[n](int(v * (255. / self._grad_value)))[:3]]

    def get_hex_color(self, n, v):
        rgb = self.get_rgb_color(n, v)
        return rgb2hex(rgb[0], rgb[1], rgb[2])

    def get_c_map(self, n):
        return lambda x: self._c_map[n](int(x * (255. / self._grad_value)))


class IntegerColorMap(object):
    def __init__(self, n):
        super().__init__()
        colors = []
        for c_map in QUALITATIVE:
            colors.extend(c_map.colors)
        # colors.extend(colors)
        for _ in range(int(n / len(colors))):
            colors.extend(colors)
        self._c_map = colors
        self._num_colors = len(colors)

    def get_rgb_color(self, n):
        return self._c_map[n]

    def get_hex_color(self, n):
        return rgb2hex(self._c_map[n][0], self._c_map[n][1], self._c_map[n][2])


class CMap(object):
    def __init__(self):
        super().__init__()

    def interpolate_rgb(self, r1, g1, b1, r2, g2, b2):
        pass

    def category(self, type='schemePastel2'):
        '''
        create a category cmap, render it with index
        :param type: schemeCategory10, schemeAccent, schemaPaired, schemePastel1, schemePastel2, etc
        :return:
        '''
        return Template('d3.%s({{index}})' % type)

    def pre_interpolate(self, type='BrBG'):
        '''
        :param type: https://github.com/d3/d3-scale-chromatic
        :return:
        '''
        return Template('d3.interpolate%s({{index}})' % type)

    def pre_scheme(self, type='BrBG'):
        '''
        :param type: https://github.com/d3/d3-scale-chromatic
        :return:
        '''
        return Template('d3.scheme%s({{index}})' % type)


class Scale(object):
    def __init__(self, domain, range, type):
        '''
        create scale mapping.
        :param domain:
        :param range:
        :param type: linear, pow, sqrt, log, threshold, Time, Sequential
        '''
        super().__init__()