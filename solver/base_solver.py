import abc
import six

@six.add_metaclass(abc.ABCMeta)
class BaseSolver(object):

    def __init__(self):
        return

    @abc.abstractmethod
    def solve(self, brick_layout):
        return NotImplementedError