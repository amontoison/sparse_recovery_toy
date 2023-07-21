using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
include("mapping.jl")
include("Mtgeneration.jl")
include("Mperptz.jl")
# compute function value, gradient, Newton direction


# compute function value
# @param t The parameter of central path in interior point method (scalar)
# @param beta (a 1-dimensional vector)
# @param c (a 1-dimensional vector)
# (beta, c) is feasible
# @param paramf A tuple consist of 3 entries: (Mt, M_perptz, d)
# Mt M^{\top}: The transpose of M
# M_perptz M_{\perp}^{\top}*z
# d The constraint on the l1 norm of beta (scalar)

# @details Compute the function value of \phi at (t, beta, c)

# @return fval Function value of \phi (scalar)
# @example
# >t = 1;
# >dim = 1;
# >size1 = 10;
# >d = 5;
# >beta = zeros(size1);
# >c = (d/(2*size1)).*ones(size1);
# >missing_idx = [2; 6];
# >z = randn(size1);
# >z_zero = z;
# >z_zero[missing_idx].= 0;
# >Mt = generate_Mt(dim, size1, missing_idx);
# >M_perptz = M_perp_tz(z_zero, dim, size1);
# >paramf = (Mt, M_perptz, d);
# >fval(t, beta, c, paramf)

function fval(t, beta, c, paramf)
    Mt = paramf[6];
    M_perptz = paramf[3];
    d = paramf[4];

    # compute l, u, h, g
    l, u, h, g = auxiliary_func(beta, c, d);

    fval = t * sum((beta' * beta).- (Mt' * beta)'*(Mt' * beta).-((M_perptz'*beta).*2)) - sum(log.((-1).*l)) - sum(log.((-1).*u)) - sum(log.((-1).*h)) - log((-1)*g);
    return fval
end

function fval2(t, beta, c, paramf)
    dim = paramf[1];
    size = paramf[2];
    M_perptz = paramf[3];
    d = paramf[4];
    idx_missing = paramf[5];

    # compute l, u, h, g
    l, u, h, g = auxiliary_func(beta, c, d);

    fval = t * sum((sum((M_perp_beta(dim, size, beta, idx_missing)).^2)).-((M_perptz'*beta).*2)) - sum(log.((-1).*l)) - sum(log.((-1).*u)) - sum(log.((-1).*h)) - log((-1)*g);

    return fval
end

# compute gradient

# @param t The parameter of central path in interior point method (scalar)
# @param beta (a 1-dimensional vector)
# @param c (a 1-dimensional vector)
# (beta, c) is feasible
# @param paramf A tuple consist of 3 entries: (Mt, M_perptz, d)
# Mt M^{\top}: The transpose of M
# M_perptz M_{\perp}^{\top}*z
# d The constraint on the l1 norm of beta (scalar)

# @details Compute gradient of \phi on beta and c respectively at (t, beta, c)

# @return gbeta, gc Gradient of \phi on beta and c respectively
# (2 vectors each with length of problem size)

# @example
# >t = 1;
# >dim = 1;
# >size1 = 10;
# >d = 5;
# >beta = zeros(size1);
# >c = (d/(2*size1)).*ones(size1);
# >missing_idx = [2; 6];
# >z = randn(size1);
# >z_zero = z;
# >z_zero[missing_idx].= 0;
# >Mt = generate_Mt(dim, size1, missing_idx);
# >M_perptz = M_perp_tz(z_zero, dim, size1);
# >paramf = (Mt, M_perptz, d);
# >gbeta, gc = fgrad(t, beta, c, paramf);

function fgrad(t, beta, c, paramf)
    Mt = paramf[6];
    M_perptz = paramf[3];
    d = paramf[4];

    l, u, h, g = auxiliary_func(beta, c, d);
    n = length(l);

    gbeta = ((2*t).*beta).- ((2*t).*Mt*(Mt'*beta)).- ((2*t).*M_perptz).+ (inv.(l)).- (inv.(u));
    gc =  (inv.(l)).+ (inv.(u)).+ (inv.(h)).- ((1/g).*ones(n));

    return gbeta, gc
end

function fgrad2(t, beta, c, paramf)
    dim = paramf[1];
    size = paramf[2];
    M_perptz = paramf[3];
    d = paramf[4];
    idx_missing = paramf[5];
    #Mt = paramf[6];
    #M_perpt = paramf[7];

    l, u, h, g = auxiliary_func(beta, c, d);
    n = length(l);

    gbeta = ((2*t).*M_perpt_M_perp_vec(dim, size, beta, idx_missing)).- ((2*t).*M_perptz).+ (inv.(l)).- (inv.(u));
    gc =  (inv.(l)).+ (inv.(u)).+ (inv.(h)).- ((1/g).*ones(n));

    #println(norm(M_perpt*M_perpt'*beta - M_perpt_M_perp_vec(dim, size, beta, idx_missing)))
    #println(norm(beta - Mt*Mt'*beta - M_perpt_M_perp_vec(dim, size, beta, idx_missing)))

    return gbeta, gc
end

# compute Newton direction

# @param t The parameter of central path in interior point method (scalar)
# @param beta (a 1-dimensional vector)
# @param c (a 1-dimensional vector)
# (beta, c) is feasible
# @param gbeta Gradient of \phi on \beta (a 1-dimensional vector)
# @param gc Gradient of \phi on c (a 1-dimensional vector)
# @param paramf A tuple consist of 3 entries: (Mt, M_perptz, d)
# Mt M^{\top}: The transpose of M
# M_perptz M_{\perp}^{\top}*z
# d The constraint on the l1 norm of beta (scalar)

# @details Compute the Newton direction of \phi on beta and c respectively at (t, beta, c)

# @return delta_betab, delta_c The Newton direction of \phi on beta and c respectively
# (2 vectors each with length of problem size)

# @example
# >t = 1;
# >dim = 1;
# >size1 = 10;
# >d = 5;
# >beta = zeros(size1);
# >c = (d/(2*size1)).*ones(size1);
# >missing_idx = [2; 6];
# >z = randn(size1);
# >z_zero = z;
# >z_zero[missing_idx].= 0;
# >Mt = generate_Mt(dim, size1, missing_idx);
# >M_perptz = M_perp_tz(z_zero, dim, size1);
# >paramf = (Mt, M_perptz, d);
# >gbeta, gc = fgrad(t, beta, c, paramf);
# >delta_beta, delta_c = NT_direction(t, beta, c, gbeta, gc, paramf);

function NT_direction(t, beta, c, gradb, gradc, paramf)
    Mt = paramf[6];
    d = paramf[4];

    l, u, h, g = auxiliary_func(beta, c, d);
    Mt = Float64.(Mt);
    (n,m) = size(Mt);

    a = ((2t).*ones(n)).+ (inv.(l.^2)).+ (inv.(u.^2));
    b = (inv.(l.^2)).- (inv.(u.^2));
    d = (inv.(l.^2)).+ (inv.(u.^2)).+ (inv.(h.^2));

    l22_tilde_inv = inv.(d.- (inv.(a)).*(b.^2));
    l11_tilde_inv = (inv.(a)).+ (((inv.(a)).*b).^2).*l22_tilde_inv;
    l12_tilde_inv = (-1).*(inv.(a)).*b.*l22_tilde_inv;


    L11_invgb_L12inv_gc = (l11_tilde_inv.*gradb).+ (l12_tilde_inv.*gradc).- ((1/(g^2+sum(l22_tilde_inv))).*((l12_tilde_inv'*gradb).+ (l22_tilde_inv'*gradc)).*l12_tilde_inv);
    L21_invgb_L22inv_gc = (l12_tilde_inv.*gradb).+ (l22_tilde_inv.*gradc).- ((1/(g^2+sum(l22_tilde_inv))).*((l12_tilde_inv'*gradb).+ (l22_tilde_inv'*gradc)).*l22_tilde_inv);

    L11_invMt = (l11_tilde_inv.*Mt).- ((1/(g^2+sum(l22_tilde_inv))).*(l12_tilde_inv).*(l12_tilde_inv'*Mt));
    L21_invMt = (l12_tilde_inv.*Mt).- ((1/(g^2+sum(l22_tilde_inv))).*(l22_tilde_inv).*(l12_tilde_inv'*Mt));
    block = Symmetric(Matrix(I, m, m).- ((2t).*(Mt'*L11_invMt)));
    temp = cholesky(block, check=false)\(Mt'*L11_invgb_L12inv_gc);

    delta_beta = (-1).*(L11_invgb_L12inv_gc.+ (2t).*(L11_invMt*temp));
    delta_c = (-1).*(L21_invgb_L22inv_gc.+ (2t).*(L21_invMt*temp));

    return delta_beta, delta_c
end

function L_inv_r(t, l, u, h, r)
    n = length(l);
    r_beta = r[1:n];
    r_c = r[(n+1):(2*n)];

    l11 = (inv.(l.^2)).+(inv.(u.^2)).+(2*t);
    l12 = (inv.(l.^2)).-(inv.(u.^2));
    l22 = (inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2));

    l22_inv = inv.(l22.-((l12.^2).*(inv.(l11))));
    l12_inv = (inv.(l11)).*l12.*l22_inv.*(-1);
    l11_inv = (inv.(l11)).+((inv.(l11)).^2).*(l12.^2).*l22_inv;

    return [(l11_inv.*r_beta).+(l12_inv.*r_c); (l12_inv.*r_beta).+(l22_inv.*r_c)]
end

function Hessian_vec(t, l, u, h, g, Mt, p)
    n = length(l);
    p_beta = p[1:n];
    p_c = p[(n+1):(2*n)];

    l11 = (inv.(l.^2)).+(inv.(u.^2)).+(2*t);
    l12 = (inv.(l.^2)).-(inv.(u.^2));
    l22 = (inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2));

    H11_pbeta = (l11.*p_beta).-((Mt*(Mt'*p_beta)).*(2*t));
    H12_pc = l12.*p_c;
    H21_pbeta = l12.*p_beta;
    H22_pc = (l22.*p_c).+(ones(n).*(sum(p_c)/g^2));

    return [H11_pbeta.+H12_pc; H21_pbeta.+H22_pc]
end

function Hessian_vec2(t, l, u, h, g, dim, size, idx_missing, p)
    n = length(l);
    p_beta = p[1:n];
    p_c = p[(n+1):(2*n)];

    l11 = (inv.(l.^2)).+(inv.(u.^2));
    l12 = (inv.(l.^2)).-(inv.(u.^2));
    l22 = (inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2));

    H11_pbeta = (l11.*p_beta).+((M_perpt_M_perp_vec(dim, size, p_beta, idx_missing)).*(2*t));
    H12_pc = l12.*p_c;
    H21_pbeta = l12.*p_beta;
    H22_pc = (l22.*p_c).+(ones(n).*(sum(p_c)/g^2));

    return [H11_pbeta.+H12_pc; H21_pbeta.+H22_pc]
end

function CG(t, beta, c, gradb, gradc, paramf, CG_esp = 10e-6)
    Mt = paramf[6];
    d = paramf[4];

    l, u, h, g = auxiliary_func(beta, c, d);
    Mt = Float64.(Mt);
    n = length(beta);

    b = [gradb; gradc].*(-1);
    x0 = zeros(2*n);
    r0 = Hessian_vec(t, l, u, h, g, Mt, x0).-b;
    y0 = L_inv_r(t, l, u, h, r0);
    p0 = y0.*(-1);

    iter = 0
    while(norm(r0)>CG_esp)
        iter = iter + 1;
        alpha = (r0'*y0)/(p0'*Hessian_vec(t, l, u, h, g, Mt, p0));
        x1 = x0.+(p0.*alpha);
        r1 = r0.+(Hessian_vec(t, l, u, h, g, Mt, p0).*alpha);
        y1 = L_inv_r(t, l, u, h, r1);
        beta = (r1'*y1)/(r0'*y0);
        p0 = (p0.*beta).-y1;
        r0 = r1;
        y0 = y1;
        x0 = x1;
    end

    return x0[1:n], x0[(n+1):(2*n)], iter
end

function CG2(t, beta, c, gradb, gradc, paramf, CG_esp = 10e-6)
    dim = paramf[1];
    size = paramf[2];
    d = paramf[4];
    idx_missing = paramf[5];
    #Mt = paramf[6];

    #Mt = Float64.(Mt);

    l, u, h, g = auxiliary_func(beta, c, d);
    n = length(beta);

    b = [gradb; gradc].*(-1);
    x0 = zeros(2*n);
    r0 = Hessian_vec2(t, l, u, h, g, dim, size, idx_missing, x0).-b;
    y0 = L_inv_r(t, l, u, h, r0);
    p0 = y0.*(-1);

    iter = 0
    while(norm(r0)>CG_esp)
        iter = iter + 1;
        Hes_p = Hessian_vec2(t, l, u, h, g, dim, size, idx_missing, p0);
        alpha = (r0'*y0)/(p0'*Hes_p);
        #println("t=", t, " norm =", norm(Hessian_vec2(t, l, u, h, g, dim, size, idx_missing, p0)-Hessian_vec(t, l, u, h, g, Mt, p0)))
        x1 = x0.+(p0.*alpha);
        r1 = r0.+(Hes_p.*alpha);
        y1 = L_inv_r(t, l, u, h, r1);
        beta = (r1'*y1)/(r0'*y0);
        p0 = (p0.*beta).-y1;
        r0 = r1;
        y0 = y1;
        x0 = x1;
    end

    return x0[1:n], x0[(n+1):(2*n)], iter
end

function CG_precond(t, beta, c, gradb, gradc, paramf, CG_esp = 10e-6)
    Mt = paramf[1];
    d = paramf[3];

    l, u, h, g = auxiliary_func(beta, c, d);
    Mt = Float64.(Mt);
    n = length(beta);

    b = [gradb; gradc].*(-1);

    H11 = diagm((inv.(l.^2)).+(inv.(u.^2))).+((2*t).*Matrix(I, n, n)).-((Mt*Mt').*(2*t));
    H12 = diagm((inv.(l.^2)).-(inv.(u.^2)));
    H22 = diagm((inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2))).+(ones(n, n)/g^2);
    H = [H11 H12; H12 H22];

    l11 = (inv.(l.^2)).+(inv.(u.^2)).+(2*t);
    l12 = (inv.(l.^2)).-(inv.(u.^2));
    l22 = (inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2));
    L = [diagm(l11) diagm(l12); diagm(l12) diagm(l22)];

    x0 = zeros(2*n);
    r0 = H*x0-b;
    y0 = inv(L)*r0;
    p0 = y0.*(-1);

    while(norm(r0)>CG_esp)
        println("norm:", norm(r0))
        alpha = (r0'*y0)/(p0'*H*p0);
        x1 = x0.+(p0.*alpha);
        r1 = r0.+((H*p0).*alpha);
        y1 = inv(L)*r1;
        beta = (r1'*y1)/(r0'*y0);
        p0 = (p0.*beta).-y1;
        r0 = r1;
        y0 = y1;
        x0 = x1;
    end

    return x0[1:n], x0[(n+1):(2*n)]
end

function CG_unprecond(t, beta, c, gradb, gradc, paramf, CG_esp = 10e-6)
    Mt = paramf[1];
    d = paramf[3];

    l, u, h, g = auxiliary_func(beta, c, d);
    Mt = Float64.(Mt);
    n = length(beta);

    b = [gradb; gradc].*(-1);

    H11 = diagm((inv.(l.^2)).+(inv.(u.^2))).+((2*t).*Matrix(I, n, n)).-((Mt*Mt').*(2*t));
    H12 = diagm((inv.(l.^2)).-(inv.(u.^2)));
    H22 = diagm((inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2))).+(ones(n, n)./(g^2));
    H = [H11 H12; H12 H22];

    x0 = zeros(2*n);
    r0 = H*x0-b;
    p0 = r0.*(-1);

    while(norm(r0)>CG_esp)
        println("norm:", norm(r0))
        alpha = (r0'*r0)/(p0'*H*p0);
        x1 = x0.+(p0.*alpha);
        r1 = r0.+((H*p0).*alpha);
        beta = (r1'*r1)/(r0'*r0);
        p0 = (p0.*beta).-r1;
        r0 = r1;
        x0 = x1;
    end

    return x0[1:n], x0[(n+1):(2*n)]

end





# compute l, u, h, g
function auxiliary_func(beta, c, d)
    return ((-1).*beta.-c, beta.-c, (-1).*c, sum(c)-d)
end
