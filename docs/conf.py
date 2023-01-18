from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_cargo = tomllib.loads((Path(__file__).parent/'..'/'Cargo.toml').read_text())
_pyproject = tomllib.loads((Path(__file__).parent/'..'/'pyproject.toml').read_text())

project = _pyproject['project']['name']
author = 'Evalf'
copyright = '2022, Evalf'
version = release = _cargo['package']['version']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
