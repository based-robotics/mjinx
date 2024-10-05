# `mjinx` documentation

The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), [Read the docs](https://docs.readthedocs.io/en/stable/) template and Python.

The website is available at url.com.

## Building locally
To build and test the website locally, do the following:
```bash
pip install ".[docs]"
rm -r _build && sphinx-build -M html docs _build
```

And open the `file:///home/<path-to-repo>/mjinx/_build/index.html` file in the browser.