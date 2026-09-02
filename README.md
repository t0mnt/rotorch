# gato (Geometric Algebra Torch)

Gato is a package that supplies popular equivariant machine learning architectures employing Geometric Algebra (GA).
Group Equivariant neural networks rose to prominence as an answer to the question: what if my data is rotated/translated compared to the training data.
Instead of augmenting the training data with transformed versions of the training data, the neural network architecture should just be invariant to such transformations.
GA takes this to the next level because all of its operators are geometrically meaningful and equivariant, making it the natural framework to express these ideas and offering clear advantages over traditional implementations that used irreps of groups [cite?].

What gato offers:
- Geometric primitives for points, lines, planes, rotations, translations etc. that live on the GPU by leveraging torch.
- The binary (and most unary) operators of GA are equivariant by definition. As long as you build modules using these operations, equivariance is guaranteed.
- The modules of popular GA based architectures such as CGENN, GATr.
- Fully backwards compatible with existing torch modules, and indeed torch as a whole. Mind you, these are not always equivariant, so use them with care.
- GA operations at the core of the modules are highly optimized using `kingdon`, which uses symbolic optimization and Common Subexpression Elimination (CSE) to arrive at optimal code to perform the geometric operation at hand. A `torch.compile` or `triton` version of the same thing can never reach anywhere close to this level of optimization on the algebraic level.
- However, `torch.compile` or `triton` can still optimize this `kingdon` generated code even further to optimal for GPU computations.

The result of all this is the fastest code on the market to build equivariant neural networks.


## Roadmap
- [ ] Full compatibility with existing torch modules. This is something the competition does *not* have, because they all made the mistake of going for shape :code:`(..., channels_in, blades)` where blades is $2^n$. In `kingdon` v3 however, a multivector has shape :code:`(..., channels_in)` and the blade dimension is invisible. (Under the hood the underlying datastructure is :code:`(blades, ..., channels_in)` where moreover blades does not have to be the full algebra, which is where we will get our speed-up from.) The torch modules expect the channels to be the last dimension, which kingdon has!
  - [ ] Full compatibility can be achieved by defining `__torch_function__` on kingdon MultiVectors, at which point the typechecker of torch should accept multivectors directly in its forward function :). I have to decide if that will be part of kingdon, or of this package specifically so as not to polute the kingdon namespace with torch specific stuff.
  - [ ] Symbolic backwards improving over autograd
- [ ] Implement equivariant modules, which the torch ones are generally not. A challenge here will be the correct inference of the multivector types at this stage of a pipeline, such that we never have to do all the multiplies that a naive implementation does.
  - [ ] CGENN
  - [ ] GATr
- [ ] At this stage we have all the infrastructure, and can focus on speeding things up further by implementing further optimizations
  - [ ] `torch.compile` as a quick first try
  - [ ] triton specific printers that automatically generate triton kernels
