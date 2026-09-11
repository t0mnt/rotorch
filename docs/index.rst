gato (Geometric Algebra Torch)
==============================

.. The README is the front page. Its own title is skipped, since it is the title above;
   everything after it is included as-is, so the two never drift apart.

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: # gato (Geometric Algebra Torch)

Documentation
-------------

The layers in :mod:`gato.nn` are written against
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
