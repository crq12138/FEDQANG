import bisect
import hashlib


class HashRing:
    def __init__(self, nodes=None, replicas=100):
        self.replicas = replicas
        self.ring = dict()
        self._sorted_keys = []
        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node, weight=1):
        for i in range(self.replicas * max(int(weight), 1)):
            key = hashlib.sha256(f"{node}_{i}".encode()).hexdigest()
            self.ring[key] = node
            bisect.insort(self._sorted_keys, key)

    def remove_node(self, node):
        for i in range(self.replicas):
            key = hashlib.sha256(f"{node}_{i}".encode()).hexdigest()
            if key in self.ring:
                del self.ring[key]
                index = bisect.bisect_left(self._sorted_keys, key)
                if index < len(self._sorted_keys):
                    del self._sorted_keys[index]

    def get_node(self, key_str):
        if not self.ring:
            return None
        key = hashlib.sha256(key_str.encode()).hexdigest()
        idx = bisect.bisect(self._sorted_keys, key) % len(self._sorted_keys)
        return self.ring[self._sorted_keys[idx]]

    def get_nodes_for_committee(self, seed, committee_size):
        nodes = set()
        current_seed = seed
        count = 0
        while count < committee_size:
            node = self.get_node(current_seed)
            if node not in nodes:
                nodes.add(node)
                count += 1
            current_seed = hashlib.sha256(current_seed.encode()).hexdigest()
        return nodes

# HR = HashRing()
# HR.add_node('127.0.0.1:50051', 70)
# HR.add_node('127.0.0.1:50053', 70)
# HR.add_node('127.0.0.1:50055', 70)
# HR.add_node('127.0.0.1:50057', 70)
# HR.add_node('127.0.0.1:50059', 70)
# HR.add_node('127.0.0.1:50061', 70)
# HR.add_node('127.0.0.1:50063', 70)
# HR.add_node('127.0.0.1:50065', 70)
# HR.add_node('127.0.0.1:50067', 70)
# HR.add_node('127.0.0.1:50069', 70)

# committee = set()
# committee = HR.get_nodes_for_committee("genesis1", 3)
# print(committee)

