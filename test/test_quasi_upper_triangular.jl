using KroneckerTools
using LinearAlgebra
using Test

Aorig = [1 3; 0 2]
A1 = [Aorig Aorig; zeros(2,2) Aorig]
A = QuasiUpperTriangular(A1)
B = randn(4,4)
C = zeros(4,4)
QuasiTriangular.A_mul_B!(C, 1, B, 1, 4, 4, A, 1, 4)

@test C ≈ B*A1

Aorig = [1 3; 0.5 2]
A = QuasiUpperTriangular(Aorig)
B = Matrix{Float64}(I(2))
X = zeros(2,2)
QuasiTriangular.A_ldiv_B!(A,B)

@test B ≈ inv(Aorig)

B = Matrix{Float64}(I(2))
QuasiTriangular.A_rdiv_Bt!(B, A)
@test B ≈ inv(Aorig)'

B = Matrix{Float64}(I(2))
QuasiTriangular.A_rdiv_B!(B,A)

@test B ≈ inv(Aorig)

Random.seed!(123)
n = 7
a = randn(n,n)
S = schur(a)
t = S.T
b = randn(n,n)
c = similar(b)

@test t*b ≈ QuasiTriangular.A_mul_B!(QuasiUpperTriangular(t),b)
@test t'*b ≈ QuasiTriangular.At_mul_B!(QuasiUpperTriangular(t),b)
@test b*t ≈ QuasiTriangular.A_mul_B!(b,QuasiUpperTriangular(t))
@test b*t' ≈ QuasiTriangular.A_mul_Bt!(b,QuasiUpperTriangular(t))

@test t*b ≈ QuasiTriangular.A_mul_B!(c,QuasiUpperTriangular(t),b)
@test t'*b ≈ QuasiTriangular.At_mul_B!(c,QuasiUpperTriangular(t),b)
@test b*t ≈ QuasiTriangular.A_mul_B!(c,b,QuasiUpperTriangular(t))
@test b*t' ≈ QuasiTriangular.A_mul_Bt!(c,b,QuasiUpperTriangular(t))

b1 = copy(b)
x = zeros(n,n)
QuasiTriangular.A_ldiv_B!(QuasiUpperTriangular(t),b1)
@test t\b ≈ b1
b1 = copy(b)
QuasiTriangular.A_rdiv_B!(b1,QuasiUpperTriangular(t))
@test b/t ≈ b1
b1 = copy(b)
QuasiTriangular.A_rdiv_Bt!(b1,QuasiUpperTriangular(t))
@test b/t' ≈ b1

b = rand(n)
b1 = copy(b)
r = rand()
I_plus_rA_ldiv_B!(r,QuasiUpperTriangular(t),b1)
@test b1 ≈ (I(n) + r*t)\b
b1 = copy(b)
s = rand()
I_plus_rA_plus_sB_ldiv_C!(r,s,QuasiUpperTriangular(t),QuasiUpperTriangular(t*t),b1)
@test b1 ≈ (I(n) + r*t + s*t*t)\b

b = rand(n,n)
b1 = copy(b)
r = rand()
I_plus_rA_ldiv_B!(r,QuasiUpperTriangular(t),b1)
@test b1 ≈ (I(n) + r*t)\b
b1 = copy(b)
s = rand()
I_plus_rA_plus_sB_ldiv_C!(r,s,QuasiUpperTriangular(t),QuasiUpperTriangular(t*t),b1)
@test b1 ≈ (I(n) + r*t + s*t*t)\b

