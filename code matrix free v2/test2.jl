using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
using Ipopt, JuMP;
using Plots;
using GLMNet;
include("mapping.jl")
include("Mtgeneration.jl")
include("Mperptz.jl")
include("fgNT.jl")
include("backtracking.jl")
include("Newton.jl")
include("IPmtd.jl")
include("ADMM.jl")

Nt = 50;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(1)
y = x + rand(Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);

missprob = 0.05;
m = Int(floor(Nt*(missprob)))
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

# interior point method parameters
eps_barrier = 10e-6;
mu_barrier = 10;
alpha_LS = 0.1;
gamma_LS = 0.8;
eps_NT = 10e-6;

# unify all the parameters for interior point method
function param_unified(DFTdim, DFTsize, M_perptz, d, idx_missing, Mt, M_perpt, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS)
    paramf = (DFTdim, DFTsize, M_perptz, d, idx_missing, Mt, M_perpt);
    paramLS = (alpha_LS, gamma_LS, paramf);
    paramNT = (eps_NT, paramLS);
    paramB = (eps_barrier, mu_barrier, paramNT);
    return paramB;
end

M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing)
M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing);
d = 100;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, Mt, M_perpt, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);


beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;
gradb = randn(Nt);
gradc = randn(Nt);

paramf = (DFTdim, DFTsize, M_perptz, d, index_missing, Mt);
NT_beta, NT_c = NT_direction(t_init, beta_init, c_init, gradb, gradc, paramf)
CG_beta, CG_c = CG(t_init, beta_init, c_init, gradb, gradc, paramf)
norm(NT_beta-CG_beta)
norm(NT_c-CG_c)
CG_beta_prec, CG_c_prec = CG_precond(t_init, beta_init, c_init, gradb, gradc, paramf)
norm(NT_beta-CG_beta_prec)
norm(NT_c-CG_c_prec)

CG_beta_upc, CG_c_upc = CG_unprecond(t_init, beta_init, c_init, gradb, gradc, paramf)
norm(NT_beta-CG_beta_upc)
norm(NT_c-CG_c_upc)


b = [gradb; gradc].*(-1);
l, u, h, g = auxiliary_func(beta_init, c_init, d);
n = length(l);
H11 = diagm((inv.(l.^2)).+(inv.(u.^2))).+((2*t_init).*Matrix(I, n, n)).-((Mt*Mt').*(2*t_init));
H12 = diagm((inv.(l.^2)).-(inv.(u.^2)));
H22 = diagm((inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2))).+(ones(n, n)./(g^2));
H = [H11 H12; H12 H22];

norm(inv(H)*b-[CG_beta; CG_c])
cond(H)


l11 = (2*t_init).+(inv.(l.^2)).+(inv.(u.^2));
l12 = (inv.(l.^2)).-(inv.(u.^2));
l22 = (inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2));
L = [diagm(l11) diagm(l12); diagm(l12) diagm(l22)];

cond(L)


a = fval(t_init, beta_init, c_init, paramf);
b = fval2(t_init, beta_init, c_init, paramf);
norm(a-b)

a = fgrad(t_init, beta_init, c_init, paramf);
b = fgrad2(t_init, beta_init, c_init, paramf);
norm(a[1]-b[1])
norm(a[2]-b[2])

x = NT_direction(t_init, beta_init, c_init, a[1], a[2], paramf);
y = CG(t_init, beta_init, c_init, a[1], a[2], paramf);
z = CG2(t_init, beta_init, c_init, a[1], a[2], paramf);

norm(x[1]-y[1])
norm(x[2]-y[2])
norm(y[1]-z[1])
norm(y[2]-z[2])


vec = randn(Nt);

a = M_perpt_M_perp_vec(DFTdim, DFTsize, vec, index_missing)
b = vec - Mt*Mt'*vec
