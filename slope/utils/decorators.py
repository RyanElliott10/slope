class setter(object):
    '''
    A custom class that supports setting non-properties in classes without a getter.
    '''

    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)
