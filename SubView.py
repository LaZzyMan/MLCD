class SubView:
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
        <div class="sub-view" id="%s" style="width: %f%%; height: %f%%"/>
        ''' % (self._name, self._width, self._height)

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
