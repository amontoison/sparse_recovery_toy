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

# verify the mapping is correct
# 1 dim
Random.seed!(1)
N = 100;
x = randn(N);
w = fft(x)./sqrt(N);
DFTsize = size(x);
DFTdim = length(DFTsize);
beta = DFT_to_beta(DFTdim, DFTsize, w);
v = beta_to_DFT(DFTdim, DFTsize, beta);
maximum(abs.(w-v))
# w=v, so the 1d mapping is correct

# 2 dim
Random.seed!(1)
N1 = 6;
N2 = 8;
x = randn(N1, N2);
w = fft(x)./sqrt(N1*N2);
DFTsize = size(x);
DFTdim = length(DFTsize);
beta = DFT_to_beta(DFTdim, DFTsize, w);
v = beta_to_DFT(DFTdim, DFTsize, beta);
maximum(abs.(w-v))
# w = v, so the 2d mapping is correct

# 3 dim
Random.seed!(1)
N1 = 6;
N2 = 8;
N3 = 10;
x = randn(N1, N2, N3);
w = fft(x)./sqrt(N1*N2*N3);
DFTsize = size(x);
DFTdim = length(DFTsize);
beta = DFT_to_beta(DFTdim, DFTsize, w);
v = beta_to_DFT(DFTdim, DFTsize, beta);
maximum(abs.(w-v))
# w = v, so the 3d mapping is correct


############################################
# Mt generation function & M_{\perp}^{\top}*z computation function verification
# 1 dim
Random.seed!(1)
N = 100;
x = randn(N);
w = fft(x)./sqrt(N); #true DFT

DFTsize = size(x);
DFTdim = length(DFTsize);
beta = DFT_to_beta(DFTdim, DFTsize, w);

# A is orthogonal
idx = collect(1:N);
At = generate_Mt(DFTdim, DFTsize, idx);
At*At' # A is an orthogomal matrix

# A*beta = ifft(w) = x
maximum(abs.(ifft(w).*sqrt(N).-At'*beta)) # A*beta = ifft(v)
maximum(abs.(At'*beta.-x)) # A*beta = x

# generate random missing indices
missprob = 0.05;
m = Int(floor(N*(missprob)))
index_nonmissing = sort(sample(1:N, N - m, replace = false));
index_missing = setdiff(collect(1:N), index_nonmissing);
z_zero = x;
z_zero[index_missing].= 0; #zero imputed z
z = x[index_nonmissing]; #observed z

# M_{\perp}^{\top}*z
M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing)
maximum(abs.(M_perpt*z.- M_perp_tz(z_zero, DFTdim, DFTsize)))
# direct computation result equals to the function return

# 2 dim
Random.seed!(1)
N1 = 6;
N2 = 8;
x = randn(N1, N2);
w = fft(x)./sqrt(N1*N2); #true DFT

DFTsize = size(x);
DFTdim = length(DFTsize);
beta = DFT_to_beta(DFTdim, DFTsize, w);

# A is orthogonal
idx = collect(1:N1*N2);
idx = map(i->CartesianIndices(x)[i], idx); #Cartesian indices
At = generate_Mt(DFTdim, DFTsize, idx);
At*At' # A is an orthogomal matrix

# A*beta = ifft(w) = x
maximum(abs.(ifft(w).*sqrt(N1*N2).-reshape(At'*beta, N1, N2))) # A*beta = vec(ifft(v))
maximum(abs.(reshape(At'*beta, N1, N2).-x)) # A*beta = vec(x)

# generate random missing indices
missprob = 0.05;
m = Int(floor(N1*N2*(missprob)))
index_nonmissing = sort(sample(1:N1*N2, N1*N2 - m, replace = false));
index_nonmissing = map(i->CartesianIndices(x)[i], index_nonmissing); #Cartesian indices
index_missing = setdiff(collect(1:N1*N2), index_nonmissing);
index_missing = map(i->CartesianIndices(x)[i], index_missing); #Cartesian indices
z_zero = x;
z_zero[index_missing].= 0; #zero imputed z
z = x[index_nonmissing]; #observed z

# M_{\perp}^{\top}*z
M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing)
maximum(abs.(M_perpt*z.- M_perp_tz(z_zero, DFTdim, DFTsize)))
# direct computation result equals to the function return

# 3 dim
Random.seed!(1)
N1 = 6;
N2 = 8;
N3 = 10;
x = randn(N1, N2, N3);
w = fft(x)./sqrt(N1*N2*N3); #true DFT

DFTsize = size(x);
DFTdim = length(DFTsize);
beta = DFT_to_beta(DFTdim, DFTsize, w);

# A is orthogonal
idx = collect(1:N1*N2*N3);
idx = map(i->CartesianIndices(x)[i], idx); #Cartesian indices
At = generate_Mt(DFTdim, DFTsize, idx);
At*At' # A is an orthogomal matrix

# A*beta = ifft(w) = x
maximum(abs.(ifft(w).*sqrt(N1*N2*N3).-reshape(At'*beta, N1, N2, N3))) # A*beta = vec(ifft(v))
maximum(abs.(reshape(At'*beta, N1, N2, N3).-x)) # A*beta = vec(x)

# generate random missing indices
missprob = 0.05;
m = Int(floor(N1*N2*N3*(missprob)))
index_nonmissing = sort(sample(1:N1*N2*N3, N1*N2*N3 - m, replace = false));
index_nonmissing = map(i->CartesianIndices(x)[i], index_nonmissing); #Cartesian indices
index_missing = setdiff(collect(1:N1*N2*N3), index_nonmissing);
index_missing = map(i->CartesianIndices(x)[i], index_missing); #Cartesian indices
z_zero = x;
z_zero[index_missing].= 0; #zero imputed z
z = x[index_nonmissing]; #observed z

# M_{\perp}^{\top}*z
M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing)
maximum(abs.(M_perpt*z.- M_perp_tz(z_zero, DFTdim, DFTsize)))
# direct computation result equals to the function return

##############
# test the Newton's direction
function NT_direction_test(t, beta, c, gradb, gradc, Mt, d)
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

    # directly inverse
    Hessian = [(2t).*Matrix(I, n, n).-((2t).*Mt*Mt').+ diagm((inv.(l.^2)).+(inv.(u.^2)))  diagm((inv.(l.^2)).-(inv.(u.^2)));
    diagm((inv.(l.^2)).-(inv.(u.^2))) diagm((inv.(l.^2)).+(inv.(u.^2)).+(inv.(h.^2))).+((1/g^2).*ones(n,n))];
    delta2 = -inv(Hessian)*[gradb; gradc];
    return maximum(abs.(delta2.- [delta_beta; delta_c]))
end

# 1 dim testing
Nt = 100;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal

DFTsize = size(x);
DFTdim = length(DFTsize);
missing_idx = [2;5;15;67;97];
Mt = generate_Mt(DFTdim, DFTsize, missing_idx);
z_zero = x;
z_zero[missing_idx].= 0;
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);


beta = zeros(Nt)
d = 5
c = (d/(2*Nt)).*ones(Nt)
t = 1
paramf = (Mt, M_perptz, d)

gradb, gradc = fgrad(t, beta, c, paramf)
NT_direction_test(t, beta, c, gradb, gradc, Mt, d)
# the NT direction is the same as directly inverse the hessian matrix

##########################
# verify the Interior point method gets the same results as Ipopt package

# min_{b}||y-Xb|| s.t. ||b||_1<=thre
op = function(y, X, thre, n)
    model = Model(Ipopt.Optimizer);
    k = length(y);
    @variable(model, b[1:n]);
    @variable(model, t[1:n]>=0);
    @objective(model, Min, sum(((y - X*b).^2)[i] for i in 1:k)); #only NL expression can use NLobjective; sum can only use for form like i in 1:40
    @constraint(model, [i = 1:n], b[i]-t[i]<=0);
    @constraint(model, [i = 1:n], b[i]+t[i]>=0);
    @constraint(model, sum(t[i] for i in 1:n) <= thre);
    optimize!(model);
    return(JuMP.value.(b))
end

# data generation
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
function param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS)
    paramf = (Mt, M_perptz, d);
    paramLS = (alpha_LS, gamma_LS, paramf);
    paramNT = (eps_NT, paramLS);
    paramB = (eps_barrier, mu_barrier, paramNT);
    return paramB;
end

M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing);
d = 100;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);


beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

beta_LASSO = op(z, M_perpt', d, Nt);
beta, c, timeave = barrier_mtd(beta_init, c_init, t_init, paramB);
maximum(abs.(beta_LASSO.-beta))
# the interior point method gets the same result as Ipopt package
