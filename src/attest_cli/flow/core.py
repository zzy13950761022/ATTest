from typing import Callable, Dict, List, Optional


NodeResult = Optional[str]


class Node:
    name: str = "node"

    def run(self, shared: Dict) -> NodeResult:
        raise NotImplementedError


class Flow:
    """
    Minimal sequential/branching flow runner. Uses string labels returned by nodes
    to choose the next node. If a node returns None or "end", flow ends.
    """

    def __init__(self, nodes: Dict[str, Node], start: str):
        self.nodes = nodes
        self.start = start

    def run(self, shared: Dict) -> None:
        current = self.start
        while current:
            node = self.nodes[current]
            next_step = node.run(shared)
            if next_step in (None, "end"):
                return
            if next_step not in self.nodes:
                raise ValueError(f"Unknown next step: {next_step}")
            current = next_step
