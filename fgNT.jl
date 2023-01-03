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
    Mt = paramf[1];
    M_perptz = paramf[2];
    d = paramf[3];

    # compute l, u, h, g
    l, u, h, g = auxiliary_func(beta, c, d);

    fval = t * sum((beta' * beta).- (Mt' * beta)'*(Mt' * beta).-((M_perptz'*beta).*2)) - sum(log.((-1).*l)) - sum(log.((-1).*u)) - sum(log.((-1).*h)) - log((-1)*g);
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
    Mt = paramf[1];
    M_perptz = paramf[2];
    d = paramf[3];

    l, u, h, g = auxiliary_func(beta, c, d);
    n = length(l);

    gbeta = ((2t).*beta).- ((2t).*Mt*(Mt'*beta)).- ((2t).*M_perptz).+ (inv.(l)).- (inv.(u));
    gc =  (inv.(l)).+ (inv.(u)).+ (inv.(h)).- ((1/g).*ones(n));

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
    Mt = paramf[1];
    d = paramf[3];

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



# compute l, u, h, g
function auxiliary_func(beta, c, d)
    return ((-1).*beta.-c, beta.-c, (-1).*c, sum(c)-d)
end
