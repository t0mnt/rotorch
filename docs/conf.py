project = "rotorch"
author = "Martin Roelfs"
copyright = "2026, Martin Roelfs"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# The README is included on the front page: it writes its tensor shapes as $2^n$, and
# its roadmap as GitHub task lists.
myst_enable_extensions = ["dollarmath", "tasklist"]
# The README is included below the title of index.rst rather than supplying its own, so
# its first heading is an H2. It is the only Markdown in these docs.
suppress_warnings = ["myst.header"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "kingdon": ("https://kingdon.readthedocs.io/en/latest", None),
}

# The layers subclass torch.nn.Module without redocumenting forward, and torch's own
# forward docstring both says "should be overridden by all subclasses" (they have) and
# cross-references a :class:`Module` that only resolves inside torch's docs.
autodoc_inherit_docstrings = False

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_static_path = ["_static"]
