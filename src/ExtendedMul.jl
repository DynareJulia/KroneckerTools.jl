using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS
import LinearAlgebra.BLAS: @blasfunc, libblas

export unsafe_mul!

function ext_gemm!(ta::Char, tb::Char, ma::Int, nb::Int, na::Int, alpha,
                   a::StridedVecOrMat{Float64}, b::StridedVecOrMat{Float64}, beta,
                   c::StridedVecOrMat{Float64}, offset_a, offset_b, offset_c)
    lda = ndims(a) == 2 ? strides(a)[2] : ma
    ldb = ndims(b) == 2 ? strides(b)[2] : na
    ldc = ndims(c) == 2 ? strides(c)[2] : ma

    ccall((@blasfunc("dgemm_"), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{Float64},
           Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ref{Float64},
           Ptr{Float64}, Ref{BlasInt}), ta, tb, ma, nb, na, alpha, Ref(a, offset_a), lda,
          Ref(b, offset_b), ldb, beta, Ref(c, offset_c), ldc)
    return c
end

function unsafe_mul!(c::StridedVecOrMat, a::StridedVecOrMat, b::StridedVecOrMat;
                     offset1::Int = 1, offset2::Int = 1, offset3::Int = 1,
                     rows2::Int = size(a, 1), cols2::Int = size(a, 2),
                     cols3::Int = size(b, 2))
    blas_check(c, a, b, offset1, offset2, offset3, rows2, cols2, cols3)
    ext_gemm!('N', 'N', rows2, cols3, cols2, 1, a, b, 0, c, offset2, offset3, offset1)
end

function unsafe_mul!(c::StridedVecOrMat, a::StridedVecOrMat,
                     bAdj::Adjoint{Float64, <:StridedVecOrMat}; offset1::Int = 1,
                     offset2::Int = 1, offset3::Int = 1, rows2::Int = size(a, 1),
                     cols2::Int = size(a, 2), cols3::Int = size(bAdj, 2))
    blas_check(c, a, bAdj, offset1, offset2, offset3, rows2, cols2, cols3)
    b = bAdj.parent
    ext_gemm!('N', 'T', rows2, cols3, cols2, 1, a, b, 0, c, offset2, offset3, offset1)
end

function unsafe_mul!(c::StridedVecOrMat, aAdj::Adjoint{Float64, <:StridedArray},
                     b::StridedVecOrMat; offset1::Int = 1, offset2::Int = 1,
                     offset3::Int = 1, rows2::Int = size(aAdj, 1),
                     cols2::Int = size(aAdj, 2), cols3::Int = size(b, 2))
    blas_check(c, aAdj, b, offset1, offset2, offset3, rows2, cols2, cols3)
    a = aAdj.parent
    ext_gemm!('T', 'N', rows2, cols3, cols2, 1, a, b, 0, c, offset2, offset3, offset1)
end

function unsafe_mul!(c::StridedVecOrMat, aAdj::Adjoint{Float64, <:StridedVecOrMat},
                     bAdj::Adjoint{Float64, <:StridedVecOrMat}; offset1::Int = 1,
                     offset2::Int = 1, offset3::Int = 1, rows2::Int = size(aAdj, 1),
                     cols2::Int = size(aAdj, 2), cols3::Int = size(bAdj, 2))
    blas_check(c, aAdj, bAdj, offset1, offset2, offset3, rows2, cols2, cols3)
    a = aAdj.parent
    b = bAdj.parent
    ext_gemm!('T', 'T', rows2, cols3, cols2, 1, a, b, 0, c, offset2, offset3, offset1)
end

function blas_check(c, a, b, offset_c, offset_a, offset_b, ma, na, nb)
    @boundscheck begin
        # Make sure offset is a sane value
        @assert (length(a) >= offset_a >= 1)
        @assert (length(b) >= offset_b >= 1)
        @assert (length(c) >= offset_c >= 1)
        # Assert that there is enough data in each variable
        _mb = na
        @assert ma * na<=length(a) - offset_a + 1 "You're asking from A more than it has"
        @assert _mb * nb<=length(b) - offset_b + 1 "You're asking from B more than it has"
        @assert ma * nb<=length(c) - offset_c + 1 "You're assigning into C more than it can take"

        #TODO: if A is a vector, ma is supplied and na is not. Do not default to na=1
    end
end

## QuasiUpperTriangular

# B = alpha*A*B such that A is triangular
function ext_trmm!(side::Char, uplo::Char, ta::Char, diag::Char, mb::Int, nb::Int, alpha,
                   a::StridedMatrix{Float64}, b::StridedVecOrMat{Float64}, offset_b)
    # intentionally assume the triangular data is not in vector form 
    #TODO: Reconsider ma/na vs mb/nb. trmm docs usage might conflict a little with ours 

    # following trmm docs
    lda = uppercase(side) == 'L' ? mb : nb
    ldb = ndims(b) == 2 ? strides(b)[2] : mb

    ccall((@blasfunc("dtrmm_"), libblas), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
           Ref{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}), side,
          uplo, ta, diag, mb, nb, alpha, a, lda, Ref(b, offset_b), ldb)
    return b
end

# A is QuasiUpper
function unsafe_mul!(c::StridedVecOrMat, a::QuasiUpperTriangular, b::StridedVecOrMat;
                     offset1::Int = 1, offset2::Int = 1, offset3::Int = 1,
                     rows2::Int = size(a, 1), cols2::Int = size(a, 2),
                     cols3::Int = size(b, 2))
    rows3 = cols2
    blas_check(c, a, b, offset1, offset2, offset3, rows2, cols2, cols3)

    copyto!(c, offset1, b, offset3, rows3 * cols3)
    alpha = 1.0
    ext_trmm!('L', 'U', 'N', 'N', rows3, cols3, alpha, a.data, c, offset1)
    @inbounds for i in 2:rows2
        x = a[i, i - 1]
        indb = offset3 + i - 2
        indc = offset1 + i - 1
        @simd for j in 1:cols3
            c[indc] += x * b[indb]
            indb += rows2
            indc += rows2
        end
    end
    return c
end

# B is QuasiUpper
function unsafe_mul!(c::StridedVecOrMat, a::StridedVecOrMat, b::QuasiUpperTriangular;
                     offset1::Int = 1, offset2::Int = 1, offset3::Int = 1,
                     rows2::Int = size(a, 1), cols2::Int = size(a, 2),
                     cols3::Int = size(b, 2))
    rows3 = size(b, 1)
    blas_check(c, a, b, offset1, offset2, offset3, rows2, cols2, cols3)

    copyto!(c, offset1, a, offset2, rows2 * cols2)
    alpha = 1.0
    ext_trmm!('R', 'U', 'N', 'N', rows2, cols2, alpha, b.data, c, offset1)

    @inbounds for i in 2:cols2
        x = b[i, i - 1]
        inda = offset2 + rows2 * (i - 1)
        indc = offset1 + rows2 * (i - 2)
        @simd for j in 1:rows2
            c[indc] += x * a[inda]
            inda += 1
            indc += 1
        end
    end
    return c
end

# B is an Adjoint of QuasiUpper
function unsafe_mul!(c::StridedVecOrMat, a::StridedVecOrMat,
                     bAdj::Adjoint{Float64, <:QuasiUpperTriangular}; offset1::Int = 1,
                     offset2::Int = 1, offset3::Int = 1, rows2::Int = size(a, 1),
                     cols2::Int = size(a, 2), cols3::Int = size(bAdj, 2))
    blas_check(c, a, bAdj, offset1, offset2, offset3, rows2, cols2, cols3)
    b = bAdj.parent

    copyto!(c, offset1, a, offset2, rows2 * cols2)
    alpha = 1.0
    ext_trmm!('R', 'U', 'T', 'N', rows2, cols2, alpha, b.data, c, offset1)

    inda = offset2
    indc = offset1 + rows2
    @inbounds for j in 2:cols2
        x = alpha * b[j, j - 1]
        @simd for i in 1:rows2
            c[indc] += x * a[inda]
            inda += 1
            indc += 1
        end
    end
    c
end
