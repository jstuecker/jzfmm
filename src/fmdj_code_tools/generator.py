from .parse import ParamInfo, FunctionInfo
from jinja2 import Environment, PackageLoader, select_autoescape
from dataclasses import dataclass



@dataclass
class FFIFunctionInfo():
    name : str
    par : list[ParamInfo]
    template_par : list[ParamInfo] | None = None
    init_templates : dict = None
    block_size : int = 64
    kernel : FunctionInfo | None = None

env = Environment(
    loader=PackageLoader("fmdj_code_tools", "templates"),
    autoescape=select_autoescape()
)

def create_ffi_call(func: FFIFunctionInfo) -> str:
    template = env.get_template("template_ffi_call.j2")
    return template.render(f=func)