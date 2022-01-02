# KroneckerTools

WORK IN PROGRESS

`KroneckerTools` computes chains of Kronecker
products as described in Kamenik (2005).

The following computations are performed

- c = (I<sub>p</sub> ⊗ A ⊗ I<sub>q</sub>)*b
- c = (I<sub>p</sub> ⊗ A<sup>T</sup> ⊗ I<sub>q</sub>)*b
- c = (A ⊗ A ⊗ ... ⊗ A)*b
- d = (A<sup>T</sup> ⊗ A<sup>T</sup> ⊗ ... ⊗ A<sup>T</sup> ⊗ B)*c
- C = A * (B ⊗ B ⊗ .... ⊗ B)
- D = A * B * (C ⊗ C ⊗ .... ⊗ C)
- D = A<sup>T</sup> * B * (C ⊗ C ⊗ .... ⊗ C) 
- E = A*B*(C ⊗ D ⊗ ... ⊗ D)

## Installation
 ```
 julia> using Pkg
 julia> Pkg.add("KroneckerTools")
 ```
 
## Documentation

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://DynareJulia.github.io/KroneckerTools.jl/dev)

## Algorithms

We exploit the following property of the Kronecker product:
vec(A * B * C) = (C<sup>T</sup> ⊗ A) * vec(B), so as never to form the
matrix corresponding to the Kronecker product and whenever possible
use matrix product instead. 

Let A, a m * n matrix, B, a matrix whose size depends on the context, and b = vec(B). It follows that

- c = (I<sub>p</sub> ⊗ A) * b = vec(A * B), where B is a  n * p matrix.
- c = (A ⊗ I<sub>q</sub>) * b = vec(B * A<sup>T</sup>), where B is a q * m matrix
- c = (I<sub>p</sub> ⊗ A ⊗ I<sub>q</sub>) * b is computed in p blocks
c<sub>i</sub> = vec(B<sub>i</sub> * A<sup>T</sup>),  i = 1, ..., p where B<sub>i</sub> is the i<sup>th</sup> block of q * n elements of vector b

A chain of Kronecker products, (A<sub>1</sub> ⊗ A<sub>2</sub> ⊗ ... ⊗
    A<sub>n</sub>) * b can be written as 
    (A<sub>1</sub> ⊗ I<sub>p<sub>1</sub></sub>)*
    (I<sub>p<sub>2</sub></sub> ⊗ A<sub>2</sub> ⊗ I<sub>p<sub>2</sub></sub>)* ... *
        (I<sub>p<sub>n</sub></sub> ⊗ A<sub>n</sub>)b where b is a vector and p<sub>1></sub>, p<sub>2</sub>, ..., p<sub>n</sub>, q<sub>1</sub>, q<sub>2</sub>, ...,q<sub>n</sub> are such as making each group in brackets conformable.      


## References
O. Kamenik (2005), "Solving SDGE models: A new algorithm for the Sylvester
  equation", <i>Computational Economics 25</i>, 167--187.
