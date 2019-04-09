from jinja2 import Template


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