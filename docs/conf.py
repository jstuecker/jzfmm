"""Sphinx configuration for the jz-fmm documentation."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

# -- Project information -----------------------------------------------------

project = "jz-fmm"
author = "Jens St\N{LATIN SMALL LETTER U WITH DIAERESIS}cker"
copyright = f"2026, {author}"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_paramlinks",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "jztree": ("https://jstuecker.github.io/jztree/", None),
}

autodoc_preserve_defaults = True
autodoc_typehints = "description"

# These compiled modules are not needed to inspect the Python API. Mocking
# them lets documentation builds run on machines without CUDA or jz-fmm built.
autodoc_mock_imports = [
    "jzfmm_cuda",
    "jzfmm_cuda.ffi_fmm",
    "jzfmm_cuda.ffi_multipoles",
    "jzfmm_cuda.ffi_pair_summation",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "display_github": True,
    "github_user": "jstuecker",
    "github_repo": "jzfmm",
    "github_version": "main/docs/",
}


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Return a GitHub source URL for an object documented by autodoc."""
    if domain != "py" or not info.get("module"):
        return None

    module = info["module"]
    filename = module.replace(".", "/") + ".py"

    try:
        import inspect

        obj = sys.modules[module]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        source, line_start = inspect.getsourcelines(obj)
        line_end = line_start + len(source) - 1
        filename += f"#L{line_start}-L{line_end}"
    except (AttributeError, KeyError, OSError, TypeError):
        pass

    return f"https://github.com/jstuecker/jzfmm/blob/main/src/{filename}"
