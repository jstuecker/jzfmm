import tree_sitter_cuda
from tree_sitter import Language, Parser, Query, QueryCursor, Node

CUDA = Language(tree_sitter_cuda.language())
parser = Parser(CUDA)

def node_text(node: Node, txt: str) -> str:
    return txt[node.start_byte:node.end_byte]

def query(node: Node, query_src: str) -> dict:
    q = QueryCursor(Query(CUDA, query_src))
    caps = q.captures(node)

    return caps

def find_function_declarations(node: Node, txt: str, name: str | None = None) -> dict[str, Node]:
    if name is not None:
        name_match = f'(#eq? @fname "{name}")'
    else:
        name_match=""

    query_src = f"""(
        function_definition
            declarator: (function_declarator
                declarator: (identifier) @fname {name_match}
                parameters: (parameter_list) @fparam
            )
    )"""

    cursor = QueryCursor(Query(CUDA, query_src))
    res = {}
    for i,match in cursor.matches(node):
        res[node_text(match["fname"][0], txt)] = match["fparam"][0]
    return res

from dataclasses import dataclass
@dataclass
class ParamInfo():
    name : str = ""
    dtype : str = ""
    is_ptr : bool = False
    is_const : bool = False

def interprete_parameter_list(node_param: Node, txt: str):
    assert node_param.type == "parameter_list"

    res = []
    for c in node_param.named_children:
        assert c.type == "parameter_declaration"

        pinfo = ParamInfo()

        pinfo.is_const = len(query(c, '(type_qualifier)? @tq (#eq? @tq "const")')) > 0

        pinfo.dtype = node_text(c.child_by_field_name("type"), txt)

        decl = c.child_by_field_name("declarator")
        if decl.type == "identifier":
            pinfo.name = node_text(decl, txt)
        elif decl.type == "pointer_declarator":
            pinfo.is_ptr = True
            pinfo.name = node_text(decl.child_by_field_name("declarator"), txt)
        else:
            raise ValueError("Unknown type %s" % decl.type)
        
        res.append(pinfo)

    return res