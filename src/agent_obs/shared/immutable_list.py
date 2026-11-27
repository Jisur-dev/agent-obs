class ImmutableList(list):
    """A list that cannot be modified after creation."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("This list is immutable.")

    append = _immutable
    extend = _immutable
    insert = _immutable
    remove = _immutable
    pop = _immutable
    clear = _immutable
    sort = _immutable
    reverse = _immutable

    def __setitem__(self, key, value):
        raise TypeError("This list is immutable.")

    def __delitem__(self, key):
        raise TypeError("This list is immutable.")

    def __iadd__(self, other):
        raise TypeError("This list is immutable.")

    def __imul__(self, other):
        raise TypeError("This list is immutable.")
