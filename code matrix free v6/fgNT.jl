using LinearAlgebra;
using SparseArrays;

include("Mperptz.jl")

function fval(t, beta, c, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    M_perptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
    Mt = paramf[6];

    l = (-1).*beta.-c;
    u = beta.-c;

    fval = (t/2)*beta'*beta - (t/2)*sum(beta'*Mt*Mt'*beta) - t*beta'*M_perptz + t*lambda*sum(c)- sum(log.((-1).*l)) - sum(log.((-1).*u));
    return fval
end

function fval2(t, beta, c, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    M_perptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
#    Mt = paramf[6];

    l = (-1).*beta.-c;
    u = beta.-c;

    fval = (t/2)*sum((M_perp_beta(DFTdim, DFTsize, beta, index_missing)).^2) - t*beta'*M_perptz + t*lambda*sum(c)- sum(log.((-1).*l)) - sum(log.((-1).*u));
    return fval
end

function fgrad(t, beta, c, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    Mperptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
    Mt = paramf[6];

    n = length(beta);

    l = (-1).*beta.-c;
    u = beta.-c;

    gradb = t.*(beta.-Mt*Mt'*beta.-Mperptz).+inv.(l).-inv.(u);
    gradc = (t*lambda).*ones(n).+inv.(l).+inv.(u);

    return gradb, gradc
end

function fgrad2(t, beta, c, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    Mperptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
#    Mt = paramf[6];

    n = length(beta);

    l = (-1).*beta.-c;
    u = beta.-c;

    gradb = t.*(M_perpt_M_perp_vec(DFTdim, DFTsize, beta, index_missing).-Mperptz).+inv.(l).-inv.(u);
    gradc = (t*lambda).*ones(n).+inv.(l).+inv.(u);

    return gradb, gradc
end


function NTdir(t, beta, c, gradb, gradc, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    Mperptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
    Mt = paramf[6];

    n = length(beta);
    l = (-1).*beta.-c;
    u = beta.-c;

    H22 = Matrix(Diagonal(inv.(l.^2).+inv.(u.^2)));
    H11 = t.*(Matrix(I,n,n).-Mt*Mt').+H22;
    H12 = Matrix(Diagonal(inv.(l.^2).-inv.(u.^2)));

    H = [H11 H12; H12 H22];
    b = [gradb; gradc].*(-1);
    delta = inv(H)*b;

    delta_beta = delta[1:n];
    delta_c = delta[n+1:2n];

    return delta_beta, delta_c
end


function NTdir_SM1(t, beta, c, gradb, gradc, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    Mperptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
    Mt = paramf[6];

    n = length(beta);
    m = size(Mt)[2];
    l = (-1).*beta.-c;
    u = beta.-c;

    l22 = inv.(l.^2).+inv.(u.^2);
    l12 = inv.(l.^2).-inv.(u.^2);
    l11 = l22.+t;

    linv22 = inv.(l22.-(l12.^2).*(inv.(l11)));
    linv12 = linv22.*l12.*(inv.(l11)).*(-1);
    linv11 = inv.(l11).+(inv.(l11.^2)).*(l12.^2).*linv22;

    a = (linv11.*gradb).+(linv12.*gradc);
    b = (linv12.*gradb).+(linv22.*gradc);

    block = Symmetric(Matrix(I, m, m).- (t.*(Mt'*(linv11.*Mt))));
    temp = cholesky(block, check=false)\(Mt'*a);

    delta_beta = (-1).*a.-t.*(linv11.*Mt)*temp;
    delta_c = (-1).*b.-t.*(linv12.*Mt)*temp;

    return delta_beta, delta_c
end


function NTdir_SM2(t, beta, c, gradb, gradc, paramf)
    DFTdim = paramf[1];
    DFTsize = paramf[2];
    Mperptz = paramf[3];
    lambda = paramf[4];
    index_missing = paramf[5];
    Mt = paramf[6];

    n = length(beta);
    m = size(Mt)[2];
    l = (-1).*beta.-c;
    u = beta.-c;

    l22 = inv.(l.^2).+inv.(u.^2);
    l12 = inv.(l.^2).-inv.(u.^2);
    l11 = l22.+t;

    linv11 = inv.(l11.-(l12.^2).*(inv.(l22)));
    linv12 = linv11.*l12.*(inv.(l22)).*(-1);
    linv22 = inv.(l22).+(inv.(l22.^2)).*(l12.^2).*linv11;

    a = (linv11.*gradb).+(linv12.*gradc);
    b = (linv12.*gradb).+(linv22.*gradc);

    block = Symmetric(Matrix(I, m, m).- (t.*(Mt'*(linv11.*Mt))));
    temp = cholesky(block, check=false)\(Mt'*a);

    delta_beta = (-1).*a.-t.*(linv11.*Mt)*temp;
    delta_c = (-1).*b.-t.*(linv12.*Mt)*temp;

    return delta_beta, delta_c
end


function L_inv_r(t, l, u, r)
    n = length(l);
    r_beta = r[1:n];
    r_c = r[(n+1):(2*n)];

    l11 = (inv.(l.^2)).+(inv.(u.^2)).+t;
    l12 = (inv.(l.^2)).-(inv.(u.^2));
    l22 = (inv.(l.^2)).+(inv.(u.^2));

    l22_inv = inv.(l22.-((l12.^2).*(inv.(l11))));
    l12_inv = (inv.(l11)).*l12.*l22_inv.*(-1);
    l11_inv = (inv.(l11)).+((inv.(l11)).^2).*(l12.^2).*l22_inv;

    return [(l11_inv.*r_beta).+(l12_inv.*r_c); (l12_inv.*r_beta).+(l22_inv.*r_c)]
end


function Hessian_vec(t, l, u, dim, size, idx_missing, p)
    n = length(l);
    p_beta = p[1:n];
    p_c = p[(n+1):(2*n)];

    l11 = (inv.(l.^2)).+(inv.(u.^2));
    l12 = (inv.(l.^2)).-(inv.(u.^2));

    H11_pbeta = (l11.*p_beta).+((M_perpt_M_perp_vec(dim, size, p_beta, idx_missing)).*t);
    H12_pc = l12.*p_c;
    H21_pbeta = l12.*p_beta;
    H22_pc = l11.*p_c;

    return [H11_pbeta.+H12_pc; H21_pbeta.+H22_pc]
end


function CG(t, beta, c, gradb, gradc, paramf, CG_esp = 10e-6)
    dim = paramf[1];
    size = paramf[2];
    lambda = paramf[4];
    idx_missing = paramf[5];
    #Mt = paramf[6];

    l = (-1).*beta.-c;
    u = beta.-c;
    n = length(beta);

    b = [gradb; gradc].*(-1);
    x0 = zeros(2*n);
    r0 = Hessian_vec(t, l, u, dim, size, idx_missing, x0).-b;
    y0 = L_inv_r(t, l, u, r0);
    p0 = y0.*(-1);

    #iter = 0
    while(norm(r0)>CG_esp)
        #iter = iter + 1;
        Hes_p = Hessian_vec(t, l, u, dim, size, idx_missing, p0);
        alpha = (r0'*y0)/(p0'*Hes_p);
        #println("t=", t, " norm =", norm(Hessian_vec2(t, l, u, h, g, dim, size, idx_missing, p0)-Hessian_vec(t, l, u, h, g, Mt, p0)))
        x1 = x0.+(p0.*alpha);
        r1 = r0.+(Hes_p.*alpha);
        y1 = L_inv_r(t, l, u, r1);
        beta = (r1'*y1)/(r0'*y0);
        p0 = (p0.*beta).-y1;
        r0 = r1;
        y0 = y1;
        x0 = x1;
    end

    return x0[1:n], x0[(n+1):(2*n)]#, iter
end
