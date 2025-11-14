# Sphinx Documentation

This directory contains the Sphinx documentation for `cslrtools2`.

## Structure

```
sphinx/
├── source/          # Source files (RST, conf.py)
│   ├── conf.py     # Sphinx configuration
│   ├── index.rst   # Main documentation page
│   ├── modules.rst # Auto-generated module index
│   └── cslrtools2.*.rst  # Auto-generated API docs
└── build/          # Built documentation
    └── html/       # HTML output
```

## Building Documentation

### Prerequisites

Install Sphinx and dependencies:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

Or using uv with dev group:

```bash
uv pip install --group dev
```

### Build HTML

```bash
sphinx-build -b html sphinx/source sphinx/build/html
```

Or using PowerShell from project root:

```bash
C:/Users/ikeko/Workspace/1github/python-cslrtools2/.venv/Scripts/sphinx-build.exe -b html sphinx/source sphinx/build/html
```

### View Documentation

Open `sphinx/build/html/index.html` in your browser.

## Auto-generating API Documentation

Re-generate API documentation from source code:

```bash
sphinx-apidoc -f -o sphinx/source src/cslrtools2
```

This will update:
- `modules.rst`
- `cslrtools2.rst`
- `cslrtools2.*.rst` (for subpackages)

## Configuration

Edit `sphinx/source/conf.py` to customize:

- **Theme**: Currently using `sphinx_rtd_theme` (Read the Docs theme)
- **Extensions**: autodoc, napoleon, viewcode, intersphinx, sphinx-autodoc-typehints
- **Intersphinx**: Links to Python, PyTorch, NumPy docs
- **Napoleon**: Google-style docstring support

## Extensions

### Enabled Extensions

1. **sphinx.ext.autodoc**: Auto-generate API docs from docstrings
2. **sphinx.ext.napoleon**: Support Google/NumPy style docstrings
3. **sphinx.ext.viewcode**: Add links to source code
4. **sphinx.ext.intersphinx**: Link to external documentation
5. **sphinx_autodoc_typehints**: Better type hint rendering

### Configuration Options

```python
# autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
```

## Updating Documentation

### After Adding New Modules

1. Re-run `sphinx-apidoc`:
   ```bash
   sphinx-apidoc -f -o sphinx/source src/cslrtools2
   ```

2. Rebuild HTML:
   ```bash
   sphinx-build -b html sphinx/source sphinx/build/html
   ```

### After Updating Docstrings

Just rebuild (no need to re-run apidoc):

```bash
sphinx-build -b html sphinx/source sphinx/build/html
```

### Clean Build

Remove build artifacts and rebuild from scratch:

```bash
# PowerShell
Remove-Item -Recurse -Force sphinx/build/html/*
sphinx-build -b html sphinx/source sphinx/build/html
```

## Publishing

### GitHub Pages

Option 1: Build locally and commit to `gh-pages` branch
Option 2: Use GitHub Actions to build automatically

### Read the Docs

Connect your GitHub repository to Read the Docs:

1. Sign up at https://readthedocs.org/
2. Import your repository
3. Configure build to use `sphinx/` directory

## Troubleshooting

### Import Errors

If Sphinx can't import modules, ensure source path is correct in `conf.py`:

```python
import sys
sys.path.insert(0, os.path.abspath('../../src'))
```

### Missing Dependencies

Install all optional dependencies before building:

```bash
pip install mediapipe torch torchvision numpy zarr
```

### Theme Not Found

Install the theme:

```bash
pip install sphinx-rtd-theme
```

## Links

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [autodoc Extension](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
