from jinja2 import Template
from pywebplot.utils import *
from palettable.colorbrewer.qualitative import Set3_12, Set2_8, Set1_9, Pastel2_8, Pastel1_9, Paired_12, Dark2_8, Accent_8

QUALITIVE = [Set3_12, Set2_8, Set1_9, Pastel2_8, Pastel1_9, Paired_12, Dark2_8, Accent_8]


class IntegerColorMap(object):
    def __init__(self, n):
        super().__init__()
        colors = []
        for c_map in QUALITIVE:
            colors.extend(c_map.colors)
        self._c_map = colors
        self._num_colors = len(colors)
        if n > len(colors):
            raise AttributeError('No enough colors.')
        self._c_map = colors
        self._num_colors = n

    def get_rgb_color(self, n):
        if n > self._num_colors:
            raise IndexError('%d out of range.' % n)
        return self._c_map[n]

    def get_hex_color(self, n):
        if n > self._num_colors:
            raise IndexError('%d out of range.' % n)
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