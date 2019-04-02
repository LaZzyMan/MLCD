class SubView:
    def __init__(self, width, height, name=''):
        '''
        create a subview
        :param width: width relate to father dom
        :param height: height relate to father dom
        :param name: identifier for subview
        '''
        self._width = width
        self._height = height
        self._name = name
        self._dom = '''
        <div class="sub-view" id="%s" style="width: %f%%; height: %f%%"/>
        ''' % (self._name, self._width, self._height)

    @property
    def dom(self):
        return self._dom

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value
