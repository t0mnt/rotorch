# gato
Geometric Algebra Torch

## Roadmap
- [] Full compatibility with existing torch modules. This is something the competition does *not* have, because they all made the mistake of going for shape :code:`(..., channels_in, blades)` where blades is $2^n$. In `kingdon` v3 however, a multivector has shape :code:`(..., channels_in)` and the blade dimension is invisible. (Under the hood the underlying datastructure is :code:`(blades, ..., channels_in)` where moreover blades does not have to be the full algebra, which is where we will get our speed-up from.) The torch modules expect the channels to be the last dimension, which kingdon has!
  - [] Full compatibility can be achieved by defining `__torch_function__` on kingdon MultiVectors, at which point the typechecker of torch should accept multivectors directly in its forward function :). I have to decide if that will be part of kingdon, or of this package specifically so as not to polute the kingdon namespace with torch specific stuff.
- [] Implement equivariant modules, which the torch ones are generally not. A challenge here will be the correct inverence of the multivector types at this stage of a pipeline, such that we never have to do all the multiplies that a naive implementation does.
  - [] CGENN
  - [] GATr
- [] At this stage we have all the infrastructure, and can focus on speeding things up further by implementing further optimizations
  - [] `torch.compile` as a quick first try
  - [] triton specific printers that automatically generate triton kernels
