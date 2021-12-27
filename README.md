# KroneckerTools

WORK IN PROGRESS

`KroneckerTools` implement functions for computing chains of Kronecker
products as described in Kamenik (2005).

The following computations are performed

- C = A * (B ⊗ B ⊗ .... ⊗ B)
- D = A * B * (C ⊗ C ⊗ .... ⊗ C)
- D = A<sup>T</sup> * B * (C ⊗ C ⊗ .... ⊗ C) 
- E = A*B*(C ⊗ D ⊗ ... ⊗ D)
- d = (A<sup>T</sup> ⊗ A<sup>T</sup> ⊗ ... ⊗ A<sup>T</sup> ⊗ B)*c
- c = (A ⊗ A ⊗ ... ⊗ A)*b
- c = (I<sub>p</sub> ⊗ A ⊗ I<sub>q</sub>)*b
- c = (I<sub>p</sub> ⊗ A<sup>T</sup> ⊗ I<sub>q</sub>)*b


## Functions



## References
O. Kamenik (2005), "Solving SDGE models: A new algorithm for the Sylvester
  equation", <i>Computational Economics 25</i>, 167--187.
