rotorch (Rotors in Torch)
=========================

.. The README is the front page. Its own title is skipped, since it is the title above;
   everything after it is included as-is, so the two never drift apart.

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: # rotorch (Rotors in Torch)

Documentation
-------------

The layers in :mod:`rotorch.nn` are written against
:class:`~kingdon.multivector.MultiVector` objects, so they infer their algebra, their
grades and the shape of their parameters from the input they are first given.

.. toctree::
   :maxdepth: 2

   api
   benchmark

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
