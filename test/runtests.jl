using LinearAlgebra
using Random
using Test
import KroneckerTools: kron_mul_elem!, kron_mul_elem_t!, a_mul_kron_b!, a_mul_b_kron_c!,
                       a_mul_b_kron_ct!, at_mul_b_kron_c!, a_mul_b_kron_c_d!, kron_a_mul_b!,
                       kron_at_mul_b!, kron_at_kron_b_mul_c!,
                       QuasiTriangular.QuasiUpperTriangular, 
                       ext_gemm!, ext_trmm!, unsafe_mul!

Random.seed!(123)
#a = rand(2,3)
#b = rand(3,4)
#c = zeros(8)

#gemm!('N', 'N', 1.0, vec(a), 2, 3, vec(b), 4, 1.0, c)
#@test reshape(c,2,4) ≈ a*b

#=
for m in [1, 3]
    for n in [3, 4]
        local a = randn(n,n)
        t = schur(a).T
        depth = 4
        for p = 0:4
            for q = depth - p
                d = randn(m*n^(p+q+1))
                d_orig = copy(d)
                w = similar(d)
                println("m = $m, p = $p, q = $q")
                d = copy(d_orig)
                @time kron_mul_elem_t!(w, a, d, n^p, n^q*m)
                @test w ≈ kron(kron(Matrix{Float64}(I(n^p)),a'),Matrix{Float64}(I(m*n^q)))*d_orig
                d = copy(d_orig)
                @time kron_mul_elem_t!(w, t, d, n^p, n^q*m)
                @test w ≈ kron(kron(Matrix{Float64}(I(n^p)),t'),Matrix{Float64}(I(m*n^q)))*d_orig
                d = copy(d_orig)
                @time kron_mul_elem!(w, a, d, n^p, n^q*m)
                @test w ≈ kron(kron(Matrix{Float64}(I(n^p)),a),Matrix{Float64}(I(m*n^q)))*d_orig
                d = copy(d_orig)
                @time kron_mul_elem!(w, t, d, n^p, n^q*m)
                @test w ≈ kron(kron(Matrix{Float64}(I(n^p)),t),Matrix{Float64}(I(m*n^q)))*d_orig
            end
        end
    end
end
=#

@testset verbose=true "KroneckerTools" begin
    @testset verbose=true "ExtendedMul" begin
        @testset "Few Sanity Checks" begin
            #TODO: this is an infinitesimal sample of the allowed possibilities
            # undersized c output
            n = 5
            a = randn(n + 1, n)
            b = randn(n, n + 1)
            c = randn(n, n)
            @test_throws AssertionError unsafe_mul!(c, a, b)

            # undersized b 
            a = randn(n * 2, n)
            b = randn(n - 1, n * 3)
            c = randn(n * 2, n * 3)
            @test_throws AssertionError unsafe_mul!(c, a, b)
        end

        # Sometimes res variables are used because B is mutated
        @testset "Extended trmm" begin
            # Simplest case
            a = triu(rand(5, 5))
            b = (rand(5, 5))
            res1 = a * b
            res2 = ext_trmm!('L', 'U', 'N', 'N', 5, 5, 1, a, b, 1)
            @test res1 ≈ res2
        end

        @testset "unsafe_mul! Matrix" begin
            ## trying a different combination of default arguments each time wdw
            # a * b
            ma = 2
            na = 4
            a = randn(ma, na)
            mb = 4
            nb = 3
            b = randn(mb, nb)
            c = randn(ma, nb)
            unsafe_mul!(c, a, b; rows2 = ma, cols2 = na, cols3 = nb)
            @test c ≈ a * b

            # a' * b
            mb = 2
            nb = 3
            b = randn(mb, nb)
            c = randn(na, nb)
            unsafe_mul!(c, a', b; offset1 = 1, offset2 = 1, cols2 = ma, cols3 = nb)
            @test c ≈ transpose(a) * b
            c = randn(na, nb)
            unsafe_mul!(c, a', b)
            @test c ≈ a' * b

            # a * b'
            nb = 4
            mb = 3
            b = randn(mb, nb)
            c = randn(ma, mb)
            unsafe_mul!(c, a, b'; offset1 = 1, offset3 = 1, rows2 = ma)
            @test c ≈ a * transpose(b)
            c = randn(ma, mb)
            unsafe_mul!(c, a, b')
            @test c ≈ a * transpose(b)

            # a' * b'
            mb = 4
            nb = 2
            b = randn(mb, nb)
            c = randn(na, mb)
            unsafe_mul!(c, a', b')
            @test c ≈ transpose(a) * transpose(b)
        end

        @testset "unsafe_mul! QuasiUpper" begin
            # a * b, a is Quasi
            n = 5
            a = QuasiUpperTriangular(triu(randn(n, n)))
            b = randn(n, n * 2)
            c = randn(n, n * 2)
            unsafe_mul!(c, a, b)
            @test c ≈ a * b

            # a * b', b is Quasi
            n = 4
            a = randn(n, n)
            b = QuasiUpperTriangular(triu(randn(n, n)))
            c = randn(n, n)
            unsafe_mul!(c, a, b')
            @test c ≈ a * transpose(b)
            # write output to a vector
            c = randn(n * n)
            unsafe_mul!(c, a, b')
            @test c ≈ vec(a * transpose(b))
        end
    end

    @testset "Kron operations" begin
        order = 2
        ma = 2
        na = 4
        q = 2
        p = na^(order - 1)
        a = rand(ma, na)
        b_orig = rand(q * na^order)
        c = rand(ma * p * q)
        b = copy(b_orig)
        kron_mul_elem!(c, a, b, p, q)
        @test c ≈ kron(kron(I(p), a), I(q)) * b_orig

        ma = 2
        na = 2
        a = randn(ma, na)
        mc = 3
        order = 4
        c = randn(mc, mc)
        b = randn(na, mc^order)
        b_orig = copy(b)
        d = randn(ma, mc^order)
        w1 = Vector{Float64}(undef, ma * mc^order)
        w2 = Vector{Float64}(undef, ma * mc^order)
        a_mul_b_kron_c!(d, a, b, c, order, w1, w2)
        cc = c
        for i in 2:order
            cc = kron(cc, c)
        end
        @test d ≈ a * b_orig * cc

        b = copy(b_orig)
        a_mul_kron_b!(d, b, c, order)
        cc = c
        for i in 2:order
            cc = kron(cc, c)
        end
        @test d ≈ b_orig * cc

        order = 3
        ma = 2
        na = 4
        a = randn(ma, na)
        mb1 = 2
        nb1 = 4
        b1 = randn(mb1, nb1)
        mb2 = 2
        nb2 = 2
        b2 = randn(mb2, nb2)
        b = [b1, b2]
        c = randn(ma, nb1 * nb2)
        work = zeros(16)
        a_mul_kron_b!(c, a, b, work)
        @test c ≈ a * kron(b[1], b[2])

        order = 3
        ma = 2
        na = 4
        a = randn(ma, na)
        mb = 4
        nb = 8
        b = randn(mb, nb)
        c = randn(2, 2)
        d = randn(2, 2)
        work1 = zeros(mb * nb)
        work2 = zeros(mb * nb)
        e = randn(ma, 8)
        a_mul_b_kron_c_d!(e, a, b, c, d, order, work1, work2)
        @test e ≈ a * b * kron(c, kron(d, d))

        order = 2
        ma = 2
        na = 4
        a = randn(ma, na)
        mb = 4
        nb = 8
        b = randn(mb, nb)
        c = randn(ma * ma * nb)
        d = randn(na * na * mb)
        work1 = rand(na * na * mb)
        work2 = rand(na * na * mb)
        kron_at_kron_b_mul_c!(d, a, order, b, c, work1, work2)
        kron_at_kron_b_mul_c!(d, a, order, b, c, work1, work2)
        @test d ≈ kron(kron(a', a'), b) * c

        @testset "kron_a_mul_b" begin
            order = 2
            ma = 2
            na = 4
            q = 2
            a = rand(ma, na)
            b = rand(q * na^order)
            c = rand(q * ma^order)
            work1 = rand(q * max(ma, na)^order)
            work2 = similar(work1)
            q = 2
            kron_a_mul_b!(c, a, order, b, q, work1, work2)
            kron_a_mul_b!(c, a, order, b, q, work1, work2)
            @test c ≈ kron(kron(a, a), Matrix{Float64}(I(q))) * b

            # test2
            b = rand(q * ma^order)
            c = rand(q * na^order)
            kron_at_mul_b!(c, a, order, b, q, work1, work2)
            kron_at_mul_b!(c, a, order, b, q, work1, work2)
            @test c ≈ kron(kron(a', a'), Matrix{Float64}(I(q))) * b
        end

        @testset "a_mul_b_kron_c" begin
            order = 2
            mc = 2
            nc = 3
            c = rand(mc, nc)
            ma = 2
            na = 4
            a = rand(ma, na)
            nb = nc^order
            b = rand(na, nb)
            d = rand(ma, mc^order)
            work1 = rand(ma * max(mc, nc)^order)
            work2 = similar(work1)
            a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
            a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
            @test d ≈ a * b * kron(c', c')

            # test4
            nb = mc^order
            b = rand(ma, nb)
            d = rand(na, nc^order)
            work1 = rand(na * max(mc, nc)^order)
            work2 = similar(work1)
            at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
            at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
            @test d ≈ a' * b * kron(c, c)

            order = 2
            mc = 2
            nc = 3
            c = rand(mc, nc)
            ma = 4
            na = 4
            a = rand(ma, na)
            nb = nc^order
            b = rand(na, nb)
            d = rand(ma, mc^order)
            work1 = rand(ma * max(mc, nc)^order)
            work2 = similar(work1)
            a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
            a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
            @test d ≈ a * b * kron(c', c')

            #test4a
            nb = mc^order
            b = rand(ma, nb)
            d = rand(na, nc^order)
            work1 = rand(na * max(mc, nc)^order)
            work2 = similar(work1)
            at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
            at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
            @test d ≈ a' * b * kron(c, c)
        end
    end

    @testset "Kron with QuasiUpper" begin
        order = 2
        ma = 4
        na = 4
        q = 2
        a = QuasiUpperTriangular(triu(rand(ma, na)))
        b = rand(q * na^order)
        c = rand(q * ma^order)
        work1 = rand(q * max(ma, na)^order)
        work2 = similar(work1)
        kron_a_mul_b!(c, a, order, b, q, work1, work2)
        kron_a_mul_b!(c, a, order, b, q, work1, work2)
        @test c ≈ kron(kron(a, a), Matrix{Float64}(I(q))) * b

        # test2a
        b = rand(q * ma^order)
        c = rand(q * na^order)
        kron_at_mul_b!(c, a, order, b, q, work1, work2)
        kron_at_mul_b!(c, a, order, b, q, work1, work2)
        @test c ≈ kron(kron(a', a'), Matrix{Float64}(I(q))) * b

        # Views of QuasiUpper
        order = 2
        ma = 4
        na = 4
        q = 2
        a = QuasiUpperTriangular(triu(rand(ma, na)))
        b = view(rand(q * na^order), :)
        c = view(rand(q * ma^order), :)
        work1 = view(rand(q * max(ma, na)^order), :)
        work2 = view(rand(q * max(ma, na)^order), :)
        kron_a_mul_b!(c, a, order, b, q, work1, work2)
        kron_a_mul_b!(c, a, order, b, q, work1, work2)
        @test c ≈ kron(kron(a, a), Matrix{Float64}(I(q))) * b

        # test2b
        b = view(rand(q * ma^order), :)
        c = view(rand(q * na^order), :)
        work1 = view(rand(q * max(ma, na)^order), :)
        work2 = view(rand(q * max(ma, na)^order), :)
        kron_at_mul_b!(c, a, order, b, q, work1, work2)
        kron_at_mul_b!(c, a, order, b, q, work1, work2)
        @test c ≈ kron(kron(a', a'), Matrix{Float64}(I(q))) * b
    end

    @testset "Kron with views" begin
        order = 2
        mc = 2
        nc = 3
        c = view(rand(mc, nc), :, :)
        ma = 4
        na = 4
        a = rand(ma, na)
        nb = nc^order
        b = view(rand(na, nb), :, :)
        d = view(rand(ma, mc^order), :, :)
        work1 = view(rand(ma * max(mc, nc)^order), :)
        work2 = view(rand(ma * max(mc, nc)^order), :)
        a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
        a_mul_b_kron_ct!(d, a, b, c, order, work1, work2)
        @test d ≈ a * b * kron(c', c')

        # test3b
        nb = mc^order
        b = view(rand(ma, nb), :, :)
        d = view(rand(na, nc^order), :, :)
        work1 = rand(na * max(mc, nc)^order)
        work2 = similar(work1)
        at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
        at_mul_b_kron_c!(d, a, b, c, order, work1, work2)
        @test d ≈ a' * b * kron(c, c)
    end
end
