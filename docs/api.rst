API reference
=============

Both :mod:`gato.nn` and :mod:`gato.models` are organised one subpackage per architecture.
Only cgenn is implemented so far, so everything below lives under ``cgenn``; GATr and
others will appear alongside it rather than in place of it.

Layers
------

.. automodule:: gato.nn

cgenn
^^^^^

.. The layers are defined in submodules and re-exported from ``gato.nn.cgenn``, so
   ``imported-members`` is what documents them under the path users actually import.

.. automodule:: gato.nn.cgenn
   :members:
   :imported-members:
   :show-inheritance:

Models
------

.. automodule:: gato.models

cgenn
^^^^^

.. automodule:: gato.models.cgenn
   :members:
   :imported-members:
   :show-inheritance:
