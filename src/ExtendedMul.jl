using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS
import LinearAlgebra.BLAS: gemm!, @blasfunc, libblas

export _mul!

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

# a * b, 3 methods (A_mul_B!) 
function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
               b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    gemm!('N', 'N', 1.0, Ref(a, offset_a), ma, na, Ref(b, offset_b),
          nb, 0.0, Ref(c, offset_c))
end

function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
               b::AbstractVecOrMat{Float64})
    nb = (ndims(b) > 1) ? size(b, 2) : 1
    _mul!(c, offset_c, a, offset_a, ma, na, b, 1, nb)
end

function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::AbstractVecOrMat{Float64},
               b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    ma = size(a, 1)
    na = (ndims(a) > 1) ? size(a, 2) : 1
    _mul!(c, offset_c, a, 1, ma, na, b, offset_b, nb)
end

# a' * b, 2 methods (At_mul_B!)
function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::Adjoint{Float64}, offset_a::Int64, ma::Int64, na::Int64,
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

function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::Adjoint{Float64},
               b::AbstractVecOrMat{Float64}, offset_b::Int64, nb::Int64)
    na = size(a.parent, 1)
    ma = (ndims(a.parent) > 1) ? size(a.parent, 2) : 1
    _mul!(c, offset_c, a, 1, ma, na, b, offset_b, nb)
end

# a * b', 2 methods (A_mul_Bt!)
function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
               b::Adjoint{Float64}, offset_b::Int64, nb::Int64)
    if typeof(b) <: Adjoint{Float64, QuasiUpperTriangular{Float64, Matrix{Float64}}}
    end
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

function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::AbstractVecOrMat{Float64}, offset_a::Int64, ma::Int64, na::Int64,
               b::Adjoint{Float64})
    nb = size(b.parent, 1)
    _mul!(c, offset_c, a, offset_a, ma, na, b, 1, nb)
end

# a' * b', 1 method (At_mul_Bt!)
function _mul!(c::AbstractVecOrMat{Float64}, offset_c::Int64,
               a::Adjoint{Float64}, offset_a::Int64, ma::Int64, na::Int64,
               b::Adjoint{Float64}, offset_b::Int64, nb::Int64)
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
