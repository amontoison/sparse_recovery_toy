using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
using Ipopt, JuMP;
using Plots;
include("mapping.jl")
include("Mtgeneration.jl")
include("Mperptz.jl")
include("fgNT.jl")
include("backtracking.jl")
include("Newton.jl")
include("IPmtd.jl")
include("ADMM.jl")
include("punching.jl")

# interior point method parameters
eps_barrier = 10e-6;
mu_barrier = 10;
alpha_LS = 0.1;
gamma_LS = 0.8;
eps_NT = 10e-6;

# unify all the parameters
function param_unified(DFTdim, DFTsize, M_perptz, d, idx_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS)
    paramf = (DFTdim, DFTsize, M_perptz, d, idx_missing);
    paramLS = (alpha_LS, gamma_LS, paramf);
    paramNT = (eps_NT, paramLS);
    paramB = (eps_barrier, mu_barrier, paramNT);
    return paramB;
end

# 1 dim Interior point and ADMM

# data generation
Nt = 500;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(1)
y = x + randn(Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true))

centers = collect(1:100:Nt)
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 300;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

# barrier method
beta, c, t_vec, iter_vec, timevec, count_nt = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_1d = plot(log.(t_vec), iter_vec, seriestype=:scatter, title = "1d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)


# 2 dim Interior point and ADMM

# data generation
Nt = 20;
Ns = 24;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)#noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true))

# randomly generate missing indices
centers = CartesianIndices((1:6:Nt, 1:5:Ns))
radius = 1

index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)


# unify parameters for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero);
d = 300;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing_Cartesian, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

println("start 2d barrier")
beta, c, t_vec, iter_vec = barrier_mtd(beta_init, c_init, t_init, paramB);
plot_2d = plot(log.(t_vec), iter_vec, seriestype=:scatter, title = "2d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)


# 3 dim Interior point and ADMM

# data generation
N1 = 6;
N2 = 8;
N3 = 10;
idx1 = collect(0:(N1-1));
idx2 = collect(0:(N2-1));
idx3 = collect(0:(N3-1));
x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3];
Random.seed!(2)
y = x + rand(N1, N2, N3); # noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true))


# randomly generate missing indices
centers = CartesianIndices((1:4:N1, 1:4:N2, 1:4:N3))
radius = 1
index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameters for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero);
d = 500;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing_Cartesian, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;



index_missing_Cartesian, z_zero = punching(DFTdim, DFTsize, centers, radius, y)


# barrier method
# barrier method
beta, c, t_vec, iter_vec = barrier_mtd(beta_init, c_init, t_init, paramB);
plot_3d = plot(log.(t_vec), iter_vec, seriestype=:scatter, title = "3d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



Nt = 500;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(1)
y = x + randn(Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x); # problem dim
DFTdim = length(DFTsize); # problem size
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true))

missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 300;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

# barrier method
beta, c, t_vec, iter_vec, timevec, count_nt = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_1d = plot(log.(t_vec), iter_vec, seriestype=:scatter, title = "1d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)
