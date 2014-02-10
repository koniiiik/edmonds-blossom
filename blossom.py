#!/usr/bin/env python3
import fractions
import functools
import sys


INF = float('inf')


def cached_property(fun):
    """A memoize decorator for class properties."""
    @functools.wraps(fun)
    def get(self):
        try:
            return self._cache[fun]
        except AttributeError:
            self._cache = {}
        except KeyError:
            pass
        ret = self._cache[fun] = fun(self)
        return ret
    return property(get)


class MaximumDualReached(Exception):
    """
    Indicates that we have reached the maximum dual solution and cannot
    improve it further.
    """


class EdgeTraversalError(Exception):
    pass


class EdgeNotOutgoing(EdgeTraversalError):
    """
    Indicates that a traversal was requested through an edge from a set of
    vertices containing both its endpoints.
    """


class EdgeNotIncident(EdgeTraversalError):
    """
    Indicates that a traversal was requested through an edge from a set
    that doesn't contain any of its endpoints.
    """


class Edge:
    def __init__(self, v1, v2, value):
        self.vertices = frozenset((v1, v2))
        self.value = value

    def __hash__(self):
        return hash((self.vertices, self.value))

    def __eq__(self, other):
        return (self.vertices, self.value) == (other.vertices, other.value)

    def traverse_from(self, v):
        """Returns the other endpoint of an edge.

        The argument can be either a vertex, a Blossom or a set of
        vertices. The argument is supposed to contain exactly one
        endpoint.
        """
        if isinstance(v, Blossom):
            v = v.members
        diff = self.vertices - v
        if len(diff) == 0:
            raise EdgeNotOutgoing()
        if len(diff) > 1:
            raise EdgeNotIncident()
        return next(iter(diff))

    def calculate_charge(self):
        """Calculates the total charge on this edge.

        The charge is calculated as the sum of all blossoms containing
        each vertex minus twice the sum of all blossoms containing both
        vertices at the same time.
        """
        it = iter(self.vertices)
        blossom = next(it)
        first_owners = set()
        charge = 0
        while blossom is not None:
            charge += blossom.charge
            first_owners.add(blossom)
            blossom = blossom.owner

        blossom = next(it)
        common = None
        while blossom is not None:
            charge += blossom.charge
            if common is None and blossom in first_owners:
                common = blossom
            blossom = blossom.owner

        while common is not None:
            charge -= 2 * common.charge
            common = common.owner

        return charge

    def get_remaining_charge(self):
        return self.value - self.calculate_charge()


class Blossom:
    # For nontrivial blossoms, the charge cannot decrease below 0.
    minimum_charge = 0

    def __init__(self, cycle, charge=0, level=0):
        self.charge = fractions.Fraction(charge)
        # Whether the blossom is in an even or odd level of a tree. A
        # value of -1 indicates that the blossom is not in any tree and
        # has a single outgoing tight edge which is matched.
        self.level = level
        # Reference to the blossom directly containing this one.
        self.owner = None
        self.cycle = tuple(cycle)
        # References to parent and children in a tree.
        self.parent = None
        self.children = []

    def __hash__(self):
        return hash(self.cycle)

    def __eq__(self, other):
        return self.cycle == other.cycle

    @cached_property
    def outgoing_edges(self):
        """
        Returns a list of pairs (edge, target vertex).
        """
        members = self.members
        return list({(e, v)
                     for blossom in self.cycle
                     for e, v in blossom.outgoing_edges
                     if v not in members})

    @cached_property
    def members(self):
        return {v for blossom in self.cycle for v in blossom.members}

    def get_outermost_blossom(self):
        if self.owner is not None:
            return self.owner.get_outermost_blossom()
        return self

    def get_max_delta(self):
        """
        Finds the maximum allowed charge adjust for this blossom and its
        children.
        """
        if self.level == 1:
            # Blossoms on odd levels are going to be decreased.
            delta = self.charge - self.minimum_charge
        elif self.level == 0:
            # Even levels get increased. We need to check each outgoing
            # edge.
            delta = INF
            for e, v in self.outgoing_edges:
                b = v.get_outermost_blossom()
                remaining = e.get_remaining_charge()

                if b.level == 0:
                    # Both ends of e are on even level, both get
                    # increased, therefore each can only get one half of
                    # remaining capacity.
                    delta = min(delta, remaining / 2)
                elif b.level == -1:
                    # The other end is an out-of-tree blossom whose charge
                    # remains the same.
                    delta = min(delta, remaining)
                # Odd blossoms don't limit us in any way as the difference
                # gets canceled out on the other end.

        # Recurse into children.
        return min([delta] + [child.get_max_delta() for child in self.children])

    def adjust_charge(self, delta):
        """
        Decides what is supposed to happen on charge adjusts and recurses.
        """
        # Store the list of children for later; it might change during the
        # execution of this method.
        children = self.children
        if self.level == 0:
            self.increase_charge(delta)
        elif self.level == 1:
            self.decrease_charge(delta)

        for child in children:
            child.adjust_charge(delta)

    def increase_charge(self, delta):
        raise NotImplementedError()

    def decrease_charge(self, delta):
        self.charge -= delta
        assert self.charge >= self.minimum_charge, ("the charge of a "
                                                    "blossom dropped "
                                                    "below minimum")
        if self.charge == 0:
            self.expand()

    def expand(self):
        """
        Expands this blossom back into its constituents.
        """
        raise NotImplementedError()


class Vertex(Blossom):
    # Single vertices are allowed to have negative charges.
    minimum_charge = -INF

    def __init__(self, id):
        self.id = id
        self.edges = []
        super().__init__(cycle=None, charge=0)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def add_edge_to(self, other, value):
        e = Edge(self, other, value)
        self.edges.append(e)
        other.edges.append(e)

    @cached_property
    def outgoing_edges(self):
        return [(e, e.traverse_from(self)) for e in self.edges]

    @property
    def members(self):
        return {self}

    def expand(self):
        # For simple vertices this is a no-op.
        pass


def get_max_delta(roots):
    """
    Returns the maximal value by which we can improve the dual solution
    by adjusting charges on alternating trees.
    """
    delta = INF
    for root in roots:
        delta = min(delta, root.get_max_delta())

    if not delta > 0:
        raise MaximumDualReached()

    return delta

# Input parsing

N, M = [int(x) for x in next(sys.stdin).split()]
vertices = dict()
max_weight = 0

for line in sys.stdin:
    u, v, w = [int(x) for x in line.split()]
    for _v in (u, v):
        if _v not in vertices:
            vertices[_v] = Vertex(_v)

    u, v = vertices[u], vertices[v]
    u.add_edge_to(v, fractions.Fraction(w))

roots = set(vertices.values())

# The main cycle

try:
    while True:
        delta = get_max_delta(roots)
        for root in roots:
            root.adjust_charge(delta)
except MaximumDualReached:
    pass
