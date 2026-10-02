import ast


class CallNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.called_names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.called_names.add(node.func.id)
        self.generic_visit(node)


def extract_called_names(source: str) -> set[str]:
    tree = ast.parse(source)
    collector = CallNameCollector()
    collector.visit(tree)
    return collector.called_names