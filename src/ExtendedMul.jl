using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS
import LinearAlgebra.BLAS: gemm!, @blasfunc, libblas

export A_mul_B!, At_mul_B!, A_mul_Bt!, At_mul_B!

function gemm!(ta::Char, tb::Char, alpha::Float64,
               a::Union{Ref{Float64}, AbstractVecOrMat{Float64}}, ma::Int64, na::Int64,
               b::Union{Ref{Float64}, AbstractVecOrMat{Float64}}, nb::Int64,
               beta::Float64, c::Union{Ref{Float64}, AbstractVecOrMat{Float64}})
    ccall((@blasfunc(dgemm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
           Ref{BlasInt}),
          ta, tb, ma, nb,
          na, alpha, a, max(ma, 1),
          b, max(na, 1), beta, c,
          max(ma, 1))
end

# A_mul_B!
function A_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                  a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
                  b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    gemm!('N', 'N', 1.0, Ref(a, offset_a), ma, na, Ref(b, offset_b),
          nb, 0.0, Ref(c, offset_c))
end

function A_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                  a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
                  b::AbstractVecOrMat{Float64})
    nb = (ndims(b) > 1) ? size(b, 2) : 1
    A_mul_B!(c, offset_c, a, offset_a, ma, na, b, 1, nb)
end

function A_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                  a::AbstractVecOrMat{Float64},
                  b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    ma = size(a, 1)
    na = (ndims(a) > 1) ? size(a, 2) : 1
    A_mul_B!(c, offset_c, a, 1, ma, na, b, offset_b, nb)
end

# At_mul_B!
function At_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                   a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
                   b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    ccall((@blasfunc(dgemm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
           Ref{BlasInt}),
          'T', 'N', ma, nb,
          na, 1.0, Ref(a, offset_a), max(na, 1),
          Ref(b, offset_b), max(na, 1), 0.0, Ref(c, offset_c),
          max(ma, 1))
end

function At_mul_B!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                   a::AbstractVecOrMat{Float64},
                   b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    na = size(a, 1)
    ma = (ndims(a) > 1) ? size(a, 2) : 1
    At_mul_B!(c, offset_c, a, 1, ma, na, b, offset_b, nb)
end

# A_mul_Bt!
function A_mul_Bt!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                   a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
                   b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    ccall((@blasfunc(dgemm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
           Ref{BlasInt}),
          'N', 'T', ma, nb,
          na, 1.0, Ref(a, offset_a), max(ma, 1),
          Ref(b, offset_b), max(nb, 1), 0.0, Ref(c, offset_c),
          max(ma, 1))
end

function A_mul_Bt!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                   a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
                   b::AbstractVecOrMat{Float64})
    nb = size(b, 1)
    A_mul_Bt!(c, offset_c, a, offset_a, ma, na, b, 1, nb)
end

# At_mul_Bt!
function At_mul_Bt!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
                    a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
                    b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    ccall((@blasfunc(dgemm_), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{BlasInt}, Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ptr{Float64},
           Ref{BlasInt}),
          'T', 'T', ma, nb,
          na, 1.0, Ref(a, offset_a), max(na, 1),
          Ref(b, offset_b), max(ma, 1), 0.0, Ref(c, offset_c),
          max(ma, 1))
end
