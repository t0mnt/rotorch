Benchmark
=========

The convex hull regression of `Clifford Group Equivariant Neural Networks
<https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks>`_ (cgenn), run
with gato's layers and with the original implementation. The task is to predict the volume
of the convex hull of 16 points in :math:`\mathbb{R}^5`, so the algebra is :math:`Cl(5)`
and the model is a linear embedding followed by four geometric product layers.

Both implementations are driven by the same script, :code:`examples/hulls.py`, so they see
the same data, the same schedule and the same optimizer::

    python examples/hulls.py                       # gato
    python examples/hulls.py --compile operators   # torch.compile as kingdon's operator wrapper
    python examples/hulls.py --compile model       # torch.compile over the whole model
    python examples/hulls.py --impl cgenn          # the original implementation

Timings are the median over 64 steps of forward, backward and optimizer step, after eight
warm-up steps, and then the faster of two such runs, on an idle Apple M2 with torch 2.14 and
python 3.12. CPU only; the CUDA numbers will follow on a machine that has one.

Throughput
----------

Milliseconds per step, and the speedup over cgenn:

.. list-table::
   :header-rows: 1
   :widths: 8 14 16 16 18 16

   * - batch
     - cgenn
     - cgenn, compiled
     - gato
     - gato, operators
     - gato, model
   * - 32
     - 39.9
     - 31.6 (1.3×)
     - 28.8 (1.4×)
     - 10.8 (3.7×)
     - 9.6 (4.2×)
   * - 128
     - 101.1
     - 64.9 (1.6×)
     - 39.7 (2.5×)
     - 25.7 (3.9×)
     - 24.6 (4.1×)
   * - 512
     - 359.6
     - 205.3 (1.8×)
     - 119.4 (3.0×)
     - 68.6 (5.2×)
     - 79.9 (4.5×)
   * - 2048
     - 1475.4
     - 932.7 (1.6×)
     - 479.6 (3.1×)
     - 315.6 (4.7×)
     - 345.9 (4.3×)

Uncompiled, the lead grows with the batch size because cgenn contracts against the dense
Cayley tensor, paying for all :math:`32^3` entries whatever the input grades are, while gato
only evaluates the paths that the grades present can actually reach. Compiled, gato is between
four and five and a half times faster than cgenn.

Compiling
---------

:code:`--compile operators` hands :func:`torch.compile` to kingdon as its wrapper, so every
operator kingdon generates for the algebra is compiled on its own: 98 functions for this
model, plus their backward passes. :code:`--compile model` compiles the training step
instead, which fuses across those operators rather than stopping at each one.

Whole model compilation works because a multivector is a pytree whose coefficients are one
tensor and whose keys are static context, and because nothing in the path raises: the model
traces to **one graph with no breaks**, and :code:`fullgraph=True` succeeds and reproduces
the eager loss and gradients exactly. cgenn cannot be compiled that strictly, since it
indexes its weights with a boolean mask, whose result has a data dependent shape; its
:code:`--compile model` column is therefore compiled with graph breaks. The lorentz model below
traces to one graph as well, message passing, batch norms and gathers included.

Which of the two wins depends on the batch size, and they cross between 128 and 512.
Compiling the model removes almost all of the per step python and dispatch cost, which is
what dominates a small batch. Compiling each operator gives inductor smaller graphs to
schedule, which suits the memory bound work of a large batch better, and costs a fraction as
long to compile: 98 small graphs rather than one large one.

Other examples
--------------

The other four examples of cgenn are set up the same way, each with its own data and model but
the same loop, so they can be compared in the same terms. Milliseconds per step at each
example's own defaults, over 64 steps:

.. list-table::
   :header-rows: 1
   :widths: 10 20 10 10 12

   * - example
     - what it predicts
     - cgenn
     - gato
     - gato, model
   * - hulls
     - the volume of a convex hull in 5D
     - 39.6
     - 29.2
     - 9.5 (4.2×)
   * - o3
     - the determinant of three vectors in 3D
     - 6.3
     - 4.0
     - 3.4 (1.9×)
   * - o5
     - an O(5) invariant of two vectors in 5D
     - 3.7
     - 2.0
     - 1.2 (3.1×)
   * - nbody
     - where five charged particles end up
     - 258.4
     - 229.8
     - 94.7 (2.7×)
   * - lorentz
     - which jets came from a top quark
     - 329.7
     - 484.2
     - 194.0 (1.7×)

Each reaches the same validation loss as cgenn does on the same data, with the parameter
counts below. o3 gains least from compiling because its model is small enough that a step is
mostly fixed cost either way. The nbody and lorentz datasets are simulated by the examples rather
than read from the files the EGNN repository and the top tagging reference set ship, so their
trajectories and their jets are not the ones cgenn trains on.

lorentz is the one example where gato is slower than cgenn until it is compiled, and it is slower
for the reason given under `Where the time goes`_. Its products are wide, mixing 27 input features
into 8 output ones over the 2,912 edges of a batch, so every intermediate is an array of some six
hundred thousand numbers that eager mode writes out and reads back. Its multivectors are dense
besides: a vector in :math:`Cl(1,3)` has reached every grade by the end of the first layer, so
from the second on there is no sparsity left to spend, only the 16× fewer multiply-adds that the
sparse Cayley table saves. Fusing those intermediates is what turns 1.5× slower into 1.7× faster.

The validation losses of gato and cgenn differ even though they train on the same data, because
they do not start from the same place: their parameter counts differ, so the initial weights are
drawn differently, and since building the model draws from the same generator the batches are
shuffled differently too. Over five seeds of the o3 example they overlap, at 0.0105 to 0.0223
for gato against 0.0147 to 0.0339 for cgenn, and by 512 steps both settle around 0.001. The gap
between any two runs is the seed, not the implementation; the layers themselves agree to machine
precision when handed the same weights.

Where the time goes
-------------------

Only 1024 of the 32768 entries of the :math:`Cl(5)` Cayley tensor are nonzero, so cgenn's
einsum performs 32 times the necessary multiply-adds. The measured advantage is nowhere near
32 times, and fitting the timings above as a fixed cost plus a cost per sample says why
(:math:`r^2 \geq 0.997`):

=================  ===============  =================
run                fixed per step   marginal
=================  ===============  =================
cgenn              7.8 ms           715 µs / sample
cgenn, compiled    1.5 ms           452 µs / sample
gato               12.6 ms          227 µs / sample
gato, operators    1.8 ms           152 µs / sample
gato, model        1.0 ms           168 µs / sample
=================  ===============  =================

The marginal cost, which is the part that scales with the data, is where the sparsity shows
up: 3.1× cheaper than cgenn uncompiled, 4.7× compiled. It is not 32×, for two reasons, and
neither is about how much data fits in cache.

The first is that the product is not the whole layer. Per sample, each of the two linear maps
in a product layer costs 32 blades × 32 in × 32 out = 32,768 multiply-adds, and the sparse
product costs 1024 paths × 32 features = the same 32,768 again. The dense product costs 32
times that, 1,048,576. So the layer is 1.11 M multiply-adds for cgenn against 98 K for gato:
a factor of 11, not 32, because two thirds of gato's arithmetic is work both implementations
do identically.

The second is where the intermediates live. The geometric product of two full multivectors,
batch 512 and 32 features, counting only the multiply-adds that are not multiplications by a
structural zero:

======================  =====  ==============
run                        ms  useful GFLOP/s
======================  =====  ==============
sparse, eager            7.50             4.5
sparse, compiled         1.18            28.5
dense einsum            16.20             2.1
dense einsum, compiled   6.57             5.1
======================  =====  ==============

Eager, every one of those thousand multiply-adds is a separate torch call that reads two
arrays and writes a third, so each useful operation drags about a dozen bytes through memory
and the sparse product reaches 4.5 GFLOP/s. The dense einsum loads each value once and keeps
its intermediates in registers, so it runs at 67 GFLOP/s of raw arithmetic while doing 32
times too much of it, and still beats the sparse product by a factor of two. Fusing the
sparse product is what removes that traffic: compiled it reaches 28.5 GFLOP/s and is 5.6×
faster than the compiled dense one.

The same reasoning applies to gato's own layers, which is why the ones that hold a parameter
per grade contract over the blade axis in one go rather than a blade at a time: 32 small
einsums cost 0.61 ms where a single batched one costs 0.19, 55 against 178 GFLOP/s. Since the
coefficients of a multivector are one tensor, the layer only has to gather the parameter of
each blade first, and ``einops`` contracts multivectors directly, so this costs nothing in
readability. It is mostly a fixed cost saving, one kernel where there were 32, which is why it
moved the fixed cost of the fit above by a factor of two and left the marginal cost alone.

The fixed cost is the mirror image. gato pays 12.6 ms per step before any data is touched:
one forward and backward calls into 98 generated operators, each with its own dictionary
lookups, multivector construction and dispatch. That is python and framework time, not
arithmetic, and at batch 32 it is still 44% of the step. Compiling removes nearly all of it,
12.6 ms down to 1, which is the whole of the 4.2× at batch 32.

Parameters
----------

=========  ==========  ==========
example         cgenn        gato
=========  ==========  ==========
hulls          58,849      38,881
o3              8,657       4,973
o5            344,077     343,125
nbody         134,625     126,645
lorentz       320,594     303,525
=========  ==========  ==========

Same function, fewer parameters, a third of them in the case of hulls. Its input is a pure
vector, so the grades only fill in as the products generate them: the first product layer sees
grades (0, 1) and needs 5 path weights, the second (0, 1, 2) and 14, the third 45, the fourth
56. cgenn allocates all 56 paths and all six subspace matrices in every layer; the unreachable
ones multiply zero coefficients, contributing nothing and receiving no gradient. o5 is the
exception at almost the same count, since nearly all of its parameters sit in a plain MLP head
that both implementations share.

The two implementations agree to machine precision, module by module, in :math:`Cl(2)`,
:math:`Cl(3)`, :math:`Cl(3,1)` and :math:`Cl(3,0,1)`, and every column above reaches the same
validation loss, 21.7 to 22.7 depending on the batch size.

lorentz is checked whole rather than module by module, since its layers are wired together in a
way the others are not. Copying cgenn's weights into gato's model -- grade by grade for the
linear maps, path by path for the products, and column by column for the plain layers that read
invariants, since a grade cgenn allocates for and gato does not have contributes nothing -- and
running both over the same jets leaves at most :math:`5 \cdot 10^{-13}` between their logits in
double precision, four rounds of message passing deep. The same check in single precision leaves
2%, which is not disagreement but cancellation: a momentum of a few hundred GeV squares to a few
hundred thousand, and every invariant in this model is a difference of such numbers.

Start-up
--------

Compilation is paid on the first step:

=======================  ===================
run                      first step
=======================  ===================
gato                     0.2 to 0.8 s
compiled, warm cache     6 to 8 s
compiled, cold cache     139 s to 276 s
=======================  ===================

The cold range is the two modes: 139 s to compile the operators one at a time, 276 s for the
model as one graph. Inductor caches its kernels under :code:`$TMPDIR/torchinductor_$USER`,
some 60 MB for this model, so that is paid once per machine rather than once per run.
Compilation is specialized per shape, so each batch size compiles anew, as does the ragged
final batch of a validation loader. At batch 32, :code:`--compile model` breaks even against
the uncompiled run after about 14,000 steps cold, or 370 warm.

Compiling the model needs kingdon's operator cache to be warm, since kingdon generates its
operators on the first call and dynamo cannot trace code generation. The example therefore
runs one step eagerly before compiling.

Devices
-------

The example also runs on :code:`--device mps`, where kernel launch overhead dominates below a
batch size of a few hundred and the GPU wins above it: at batch 32 gato takes 84.7 ms/step on
mps against 28.8 on cpu, and at batch 2048 218.7 against 479.6. Neither :code:`--compile` mode
runs there, since inductor's Metal backend cannot compile these kernels: the wide ones exceed
Metal's limit of about 31 buffer arguments per kernel, one per blade, and the rest fail to
build their shaders.
