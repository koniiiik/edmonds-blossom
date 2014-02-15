#!/usr/bin/env python3
"""
An implementation of Edmonds' blossom algorithm for finding minimum-weight
maximum matchings.
"""
import fractions
import functools
import sys


INF = float('inf')
# Possible levels of blossoms.
LEVEL_EVEN = 0
LEVEL_ODD = 1
# Out-of-tree blossoms; they always appear in pairs connected by a matched
# edge.
LEVEL_OOT = -1
# Blossoms embedded in another blossom.
LEVEL_EMBED = -2


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


class TreeStructureChanged(Exception):
    """
    Used whenever the structure of an alternating tree is changed to abort
    current traversal and initiate a new one.
    """


class StructureUpToDate(Exception):
    """
    This gets raised as soon as the structure of all trees is up-to-date,
    i.e. there are no more instances of any of the four cases.
    """


class Edge(object):
    def __init__(self, v1, v2, value):
        self.vertices = frozenset((v1, v2))
        self.value = value
        self.selected = 0

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return "(%d, %d)" % tuple(sorted(v.id for v in self.vertices))

    def __repr__(self):
        return "<Edge: %s>" % (str(self),)

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

    def toggle_selection(self):
        """Toggles the membership of self in the current matching.
        """
        assert self.get_remaining_charge() == 0, ("toggle_selection called "
                                                  "on non-tight edge")
        self.selected = 1 - self.selected


class Blossom(object):
    # For nontrivial blossoms, the charge cannot decrease below 0.
    minimum_charge = 0

    def __init__(self, cycle, charge=0, level=LEVEL_EVEN):
        self.charge = fractions.Fraction(charge)
        self.level = level
        # Reference to the blossom directly containing this one.
        self.owner = None
        # The cycle of blossoms this one consists of. The first element is
        # the base.
        self.cycle = tuple(cycle)
        assert len(self.cycle) % 2 == 1
        # References to parent and children in a tree. For out-of-tree
        # pairs the parent is a reference to the peer. For embedded
        # blossoms the parent is the predecessor in the cycle.
        self.parent = None
        self.parent_edge = None
        self.children = set()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return "(%s)" % (' '.join(str(b) for b in self.cycle))

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, str(self))

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

    def get_root(self):
        assert self.level in (LEVEL_EVEN, LEVEL_ODD), ("get_root called on "
                                                       "an out-of-tree blossom")
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_base_vertex(self):
        return self.cycle[0].get_base_vertex()

    def get_max_delta(self):
        """
        Finds the maximum allowed charge adjust for this blossom and its
        children.
        """
        if self.level == LEVEL_ODD:
            # Blossoms on odd levels are going to be decreased.
            delta = self.charge - self.minimum_charge
        elif self.level == LEVEL_EVEN:
            # Even levels get increased. We need to check each outgoing
            # edge.
            delta = INF
            for e, v in self.outgoing_edges:
                b = v.get_outermost_blossom()
                remaining = e.get_remaining_charge()

                if b.level == LEVEL_EVEN:
                    # Both ends of e are on even level, both get
                    # increased, therefore each can only get one half of
                    # remaining capacity.
                    delta = min(delta, remaining / 2)
                elif b.level == LEVEL_OOT:
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
        if self.level == LEVEL_EVEN:
            self.charge += delta
        elif self.level == LEVEL_ODD:
            self.charge -= delta
            assert self.charge >= self.minimum_charge, ("the charge of a "
                                                        "blossom dropped "
                                                        "below minimum")

        for child in self.children:
            child.adjust_charge(delta)

    def alter_tree(self):
        """Detects and handles the four cases where trees need to be altered.
        """
        if self.level == LEVEL_ODD:
            if self.charge == 0:
                self.expand()
        elif self.level == LEVEL_EVEN:
            self.handle_tight_edges()
        else:
            assert False, ("alter_tree called on blossom of level %d" %
                           self.level)

        for child in self.children:
            child.alter_tree()

    def handle_tight_edges(self):
        """Finds any fresh tight edges.

        If a tight edge leads to an out-of-tree blossom, attach the pair
        (P2).

        If a tight edge leads to a blossom in the same tree as this one
        (the root blossom is the same), shrink (P3).

        If a tight edge leads to a blossom with a different root, augment
        (P4).
        """
        assert self.level == LEVEL_EVEN, ("handle_tight_edges called on "
                                          "non-even blossom.")

        for e, v in self.outgoing_edges:
            if e is self.parent_edge:
                continue
            remaining_charge = e.get_remaining_charge()
            assert remaining_charge >= 0, "found an overcharged edge"
            if remaining_charge > 0:
                continue
            other_blossom = v.get_outermost_blossom()
            if other_blossom.level == LEVEL_ODD:
                # This blossom can be from a different tree -- we don't
                # particularly care. The charge adjusts will be
                # compensated anyway.
                continue
            if other_blossom.level == LEVEL_OOT:
                self.attach_out_of_tree_pair(other_blossom, e)
                continue
            if other_blossom.get_root() == self.get_root():
                self.shrink_with_peer(other_blossom, e)
            else:
                self.augment_matching(other_blossom, e)

    def attach_out_of_tree_pair(self, target, edge):
        """Handles case (P2).
        """
        assert self.level == LEVEL_EVEN
        assert target.level == LEVEL_OOT
        assert len(target.children) == 0

        self.children.add(target)
        target_peer = target.parent
        target.parent = self
        target.parent_edge = edge
        target.level = LEVEL_ODD
        target_peer.level = LEVEL_EVEN
        target.children.add(target_peer)
        assert len(target_peer.children) == 0
        raise TreeStructureChanged("Attached blossom on edge %s" % edge)

    def shrink_with_peer(self, other, edge):
        """Shrinks the cycle along given edge into a new blossom. (P3)
        """
        assert self.level == LEVEL_EVEN
        assert other.level == LEVEL_EVEN
        # Find the closest common ancestor and the chains of parents
        # leading to them.
        ancestors, parent_chain1, parent_chain2 = dict(), [], []
        blossom = self
        while blossom is not None:
            ancestors[blossom] = len(parent_chain1)
            parent_chain1.append(blossom)
            assert blossom.parent is None or blossom in blossom.parent.children
            blossom = blossom.parent

        blossom = other
        while blossom not in ancestors:
            parent_chain2.append(blossom)
            assert blossom.parent is None or blossom in blossom.parent.children
            blossom = blossom.parent

        parent_chain2.append(blossom)
        common_ancestor = blossom
        # We need to store these values here since they get rewritten in the
        # following loop.
        new_parent = common_ancestor.parent
        new_parent_edge = common_ancestor.parent_edge

        # Remove references to other components of the new blossom from each
        # component's children list.
        for blossom in (self, other):
            while blossom is not common_ancestor:
                blossom.parent.children.remove(blossom)
                blossom = blossom.parent

        # Repoint the parent references in parent_chain2 to point in the
        # other direction. This will close the cycle.
        prev_edge, prev_blossom = edge, self
        for blossom in parent_chain2:
            prev_edge, prev_blossom, blossom.parent, blossom.parent_edge = (
                blossom.parent_edge, blossom, prev_blossom, prev_edge
            )

        # The cycle now consists of the reverse of parent_chain1 up to and
        # including common_ancestor + parent_chain2 sans common_ancestor.
        cycle = parent_chain1[ancestors[common_ancestor]::-1] + parent_chain2[:-1]

        new_blossom = Blossom(cycle)
        for blossom in cycle:
            new_blossom.children.update(blossom.children)
            blossom.owner = new_blossom
            blossom.children.clear()
            blossom.level = LEVEL_EMBED

        for child in new_blossom.children:
            child.parent = new_blossom

        if new_parent is None:
            registry = roots
        else:
            registry = new_parent.children

        registry.remove(common_ancestor)
        registry.add(new_blossom)
        new_blossom.parent = new_parent
        new_blossom.parent_edge = new_parent_edge

        raise TreeStructureChanged("Shrunk cycle on edge %s" % edge)

    def augment_matching(self, other_blossom, edge):
        """Augments the matching along the alternating path containing given edge. (P4)
        """
        # Always look for a path of even length within a blossom. Recurse
        # into sub-blossoms.
        assert edge.selected == 0, ("trying to augment via an already "
                                    "selected edge")
        self.flip_root_path(edge)
        other_blossom.flip_root_path(edge)
        edge.toggle_selection()
        raise TreeStructureChanged("Augmented on edge %s" % edge)

    def flip_root_path(self, edge):
        """Flips edge selection on the alternating path from self to root.

        Argument edge is the edge from a child from which the alternating
        path leads through self.

        The children of this blossom are detached and this blossom becomes
        part of an out-of-tree pair around the given edge.
        """
        assert self.level in (LEVEL_EVEN, LEVEL_ODD)
        v1 = self.get_base_vertex()
        if self.level == LEVEL_EVEN:
            v2 = next(iter(self.members & edge.vertices))
        else:
            v2 = next(iter(self.members & self.parent_edge.vertices))

        self.flip_alternating_path(v1, v2)

        # We need to store these values because detach_from_parent
        # modifies them and we need to recurse later.
        prev_parent, prev_parent_edge = self.parent, self.parent_edge

        if self.level == LEVEL_EVEN:
            self.detach_children()
            # Become a peer to the blossom on the other side of edge.
            self.detach_from_parent(edge)
        else:
            # Adjust self to become a peer to our parent.
            # Our child should be detached by now.
            assert len(self.children) == 0
            assert self.parent is not None
            self.parent.children.remove(self)
            self.level = LEVEL_OOT

        if prev_parent is not None:
            prev_parent_edge.toggle_selection()
            prev_parent.flip_root_path(prev_parent_edge)

    def flip_alternating_path(self, v1, v2):
        """Flips edge selection on the alternating path from v1 to v2.

        v1 and v2 are the two vertices at the boundaries of this blossom
        along the augmenting alternating path. One of the two vertices
        needs to be the base of this blossom.
        """
        assert v1 in self.members
        assert v2 in self.members
        if v1 is v2:
            return

        if v1 not in self.cycle[0].members:
            v1, v2 = v2, v1

        # v1 is in the base blossom, find the blossom containing v2
        for i, b in enumerate(self.cycle):
            if v2 in b.members:
                break

        # Trivial case: if both v1 and v2 are in the same blossom, we
        # don't need to do anything at this level.
        if i == 0:
            self.cycle[0].flip_alternating_path(v1, v2)
            return

        # self.cycle has odd length, pick the direction in which the path
        # from cycle[i] to cycle[0] has even length.
        sub_calls, edges = [], []
        if i % 2 == 0:
            # Proceed from the base forwards toward self.cycle[i].
            start, finish = 0, i
        else:
            # Proceed from self.cycle[i] forwards toward base.
            start, finish = i - len(self.cycle), 0
            v1, v2 = v2, v1

        prev_vertex = v1
        for j in range(start, finish):
            edge = self.cycle[j + 1].parent_edge
            sub_calls.append((self.cycle[j], prev_vertex,
                              edge.traverse_from(self.cycle[j + 1])))
            edges.append(edge)
            prev_vertex = edge.traverse_from(self.cycle[j])
        sub_calls.append((self.cycle[finish], prev_vertex, v2))

        assert len(sub_calls) % 2 == 1
        assert len(edges) % 2 == 0

        for e in edges:
            e.toggle_selection()
        for blossom, x1, x2 in sub_calls:
            blossom.flip_alternating_path(x1, x2)

        # Shift self.cycle to new base to keep the invariant that cycle[0]
        # is the base. cycle[i] should become cycle[0].
        self.cycle = self.cycle[i:] + self.cycle[:i]

    def detach_children(self):
        """Detaches all children and turns them into out-of-tree pairs.
        """
        # We need to make a copy of self.children here since it gets
        # modified in each iteration.
        for child in list(self.children):
            child.detach_from_parent()

    def detach_from_parent(self, edge=None):
        """Detaches itself from the parent and forms an out-of-tree pair.

        If called on an odd blossom, edge needs to be None and the only
        child will be chosen. Otherwise, an edge leading to a peer needs
        to be supplied.

        If an edge is specified, we assume the peer adjusts itself,
        otherwise we also adjust the single child that becomes the peer.
        """
        assert (self.level == LEVEL_ODD) ^ (edge is not None)

        if self.parent is not None:
            self.parent.children.remove(self)
        else:
            roots.remove(self)
        if edge is not None:
            peer = edge.traverse_from(self).get_outermost_blossom()
            self.detach_children()
        else:
            peer = next(iter(self.children))
            self.children.clear()
        self.level = LEVEL_OOT
        self.parent = peer
        if edge is None:
            self.parent_edge = peer.parent_edge
            peer.level = LEVEL_OOT
            peer.detach_children()
        else:
            self.parent_edge = edge

    def expand(self):
        """
        Expands this blossom back into its constituents. (P1)
        """
        assert self.level == LEVEL_ODD
        assert self.charge == 0

        base = self.cycle[0]
        # Since self is odd, it has exactly one child.
        assert len(self.children) == 1
        child = next(iter(self.children))
        assert child.level == LEVEL_EVEN
        child.parent = base
        base.children.add(child)

        # Find the component connected to parent.
        boundary_vertex = next(iter(self.parent_edge.vertices & self.members))
        for i, blossom in enumerate(self.cycle):
            if boundary_vertex in blossom.members:
                break

        if i % 2 == 0:
            # Repoint the even-length part of the cycle to form a part of
            # the tree.
            prev_parent, prev_edge = self.parent, self.parent_edge
            for j in range(i, -1, -1):
                b = self.cycle[j]
                assert b.level == LEVEL_EMBED
                assert prev_parent.level in (LEVEL_EVEN, LEVEL_ODD)
                assert b.owner is self
                b.level = 1 - prev_parent.level
                b.owner = None
                if prev_parent is not None:
                    prev_parent.children.add(b)
                prev_parent, prev_edge, b.parent, b.parent_edge = (
                    b, b.parent_edge, prev_parent, prev_edge
                )
            pairs_start, pairs_end = -1, i - len(self.cycle)
        else:
            # The even part of our cycle has the correct pointers already,
            # fix their levels and ownership.
            blossom.parent = self.parent
            blossom.parent_edge = self.parent_edge
            for j in range(i - len(self.cycle), 1):
                b = self.cycle[j]
                assert b.level == LEVEL_EMBED
                assert b.parent.level in (LEVEL_EVEN, LEVEL_ODD)
                assert b.owner is self
                b.level = 1 - b.parent.level
                b.parent.children.add(b)
                b.owner = None
            pairs_start, pairs_end = i - 1, 0

        # Turn the odd-length part of the cycle into out-of-tree pairs.
        for j in range(pairs_start, pairs_end, -2):
            b = self.cycle[j]
            peer = b.parent
            assert b.level == LEVEL_EMBED
            assert b.parent.level == LEVEL_EMBED
            assert b.owner is self
            assert peer.owner is self
            assert len(b.children) == 0
            assert len(peer.children) == 0
            peer.parent, peer.parent_edge = b, b.parent_edge
            b.owner = peer.owner = None
            b.level = peer.level = LEVEL_OOT

        if self.parent is None:
            registry = roots
        else:
            registry = self.parent.children
        registry.remove(self)
        registry.add(self.cycle[i])

        raise TreeStructureChanged("Expanded a blossom")


class Vertex(Blossom):
    # Single vertices are allowed to have negative charges.
    minimum_charge = -INF

    def __init__(self, id):
        self.id = id
        self.edges = []
        super(Vertex, self).__init__(cycle=[self], charge=0)

    def __str__(self):
        return "%d" % (self.id,)

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

    def get_base_vertex(self):
        return self

    def expand(self):
        # For simple vertices this is a no-op.
        pass


def get_max_delta():
    """
    Returns the maximal value by which we can improve the dual solution
    by adjusting charges on alternating trees.
    """
    if len(roots) == 0:
        # All blossoms are matched.
        raise MaximumDualReached()

    delta = INF
    for root in roots:
        delta = min(delta, root.get_max_delta())

    assert delta >= 0

    if not delta > 0:
        raise MaximumDualReached()

    return delta


def read_input(input_file):
    N, M = [int(x) for x in next(input_file).split()]
    vertices = dict()
    max_weight = 0

    for line in input_file:
        u, v, w = [int(x) for x in line.split()]
        for _v in (u, v):
            if _v not in vertices:
                vertices[_v] = Vertex(_v)

        u, v = vertices[u], vertices[v]
        u.add_edge_to(v, fractions.Fraction(w))

    return vertices


def update_tree_structures():
    try:
        while True:
            try:
                for root in roots:
                    root.alter_tree()
                raise StructureUpToDate()
            except TreeStructureChanged:
                pass
    except StructureUpToDate:
        pass


if len(sys.argv) > 1:
    input_file = open(sys.argv[1])
else:
    input_file = sys.stdin
vertices = read_input(input_file)
roots = set(vertices.values())
try:
    while True:
        delta = get_max_delta()
        sys.stderr.write("Adjusting by %s\n" % (delta,))
        for root in roots:
            root.adjust_charge(delta)
        update_tree_structures()
except MaximumDualReached:
    pass

M = set()
for v in vertices.values():
    M.update(e for e in v.edges if e.selected)

total_weight = sum(e.value for e in M)
print(total_weight)
for e in M:
    print("%s %s" % (e, e.value))
