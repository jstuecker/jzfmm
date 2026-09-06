"""Sphinx configuration for the jz-fmm documentation."""

from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path

from docutils import nodes
from sphinx.util.typing import stringify_annotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

# Documentation inspects Python objects without loading a real CUDA backend.
os.environ.setdefault("JZTREE_SKIP_JAX_CUDA_CHECK", "1")

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
if os.environ.get("JZFMM_DOCS_OFFLINE"):
    intersphinx_mapping = {}

autodoc_preserve_defaults = True
autodoc_typehints = "none"
autodoc_default_options = {
    "show-inheritance": True,
}

# These compiled modules are not needed to inspect the Python API. Mocking
# them lets documentation builds run on machines without CUDA or jz-fmm built.
autodoc_mock_imports = [
    "jztree_cuda",
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


def _badge_role(css_class: str, kind: str, refuri: str | None = None):
    """Create an inline API badge."""
    def role(name, rawtext, text, lineno, inliner, options=None, content=None):
        options = options or {}
        node_type = nodes.reference if refuri is not None else nodes.inline
        if refuri is not None:
            options["refuri"] = refuri
        node = node_type(rawtext, text, classes=[css_class, f"compatibility-{kind}"], **options)
        return [node], []

    return role


def _add_parameter_types(app, what, name, obj, options, lines):
    """Merge annotations into parameter fields after sphinx-paramlinks."""
    if what not in {"class", "function", "method"}:
        return

    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return

    annotations = {
        parameter.name: stringify_annotation(parameter.annotation, mode="smart")
        for parameter in signature.parameters.values()
        if parameter.annotation is not inspect.Parameter.empty
    }

    param_pattern = re.compile(r"^:param ([^:]+):")
    for index, line in enumerate(lines):
        match = param_pattern.match(line)
        if match is None:
            continue
        target = match.group(1)
        parameter_name = target.rsplit(".", 1)[-1].lstrip("*")
        annotation = annotations.get(parameter_name)
        if annotation is not None:
            lines[index] = line.replace(
                f":param {target}:", f":param {annotation} {target}:", 1
            )

    return_annotation = signature.return_annotation
    has_returns = any(line.startswith((":return:", ":returns:")) for line in lines)
    has_rtype = any(line.startswith(":rtype:") for line in lines)
    if (
        has_returns
        and not has_rtype
        and return_annotation is not inspect.Signature.empty
        and return_annotation is not None
    ):
        lines.append(
            f":rtype: {stringify_annotation(return_annotation, mode='smart')}"
        )


def setup(app):
    capabilities = ("jit", "shard", "autodiff")
    statuses = {
        "": "compat-yes",
        "-partial": "compat-partial",
        "-untested": "compat-untested",
        "-no": "compat-no",
    }

    for capability in capabilities:
        for suffix, css_class in statuses.items():
            app.add_role(
                f"compat-{capability}{suffix}",
                _badge_role(css_class, capability),
            )

    app.add_role("compat-shard-local", _badge_role("compat-local", "shard"))
    app.add_role(
        "helper-jit",
        _badge_role("compat-helper", "helper-jit", "jax_compatibility.html#jit-helper"),
    )
    app.add_role(
        "helper-smap",
        _badge_role("compat-helper", "helper-smap", "jax_compatibility.html#smap-helper"),
    )
    # Napoleon and sphinx-paramlinks use the default priority (500). Run after
    # both have converted and linked the Google-style parameter fields.
    app.connect("autodoc-process-docstring", _add_parameter_types, priority=600)
