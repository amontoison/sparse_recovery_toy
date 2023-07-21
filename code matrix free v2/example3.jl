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

# only need to take care of the following 3 parameters
# M_perptz can be computed by M_perp_tz(DFTdim, DFTsize, z_zero)
# d is |\beta|_1\leq d
# idx_missing is the missing indices, it should be Cartesian index for 2d and 3d


###  3d
N1 = 6;
N2 = 8;
N3 = 10;

idx1 = collect(0:(N1-1));
idx2 = collect(0:(N2-1));
idx3 = collect(0:(N3-1));
x = [(cos(2*pi*1/N1*i)+ 2*sin(2*pi*1/N1*i))*(cos(2*pi*2/N2*j) + 2*sin(2*pi*2/N2*j))*(cos(2*pi*3/N3*k) + 2*sin(2*pi*3/N3*k)) for i in idx1, j in idx2, k in idx3];
Random.seed!(1)
dist = Normal(0,1);
y = x + rand(dist, N1, N2, N3);

w = round.(fft(x)./sqrt(N1*N2*N3), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #225

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)


######### start from here ###########
# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz, z_zero is the data setting punched locations as 0
d = 200;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

# barrier method
beta, c, t_vec_3d_1, iter_vec_3d_1, timevec_3d_1, count_nt_3d_1 = barrier_mtd(beta_init, c_init, t_init, paramB);
# t_vec_3d_1 records the t in log-barrier method for each Newton system
# iter_vec_3d_1 records the iteration number inside CG alogirhtm
mean(timevec_3d_1) #average time of solving Newton system
# count_nt_3d_1 is the total number of Newton system
plot_3d_1 = plot(log.(t_vec_3d_1), iter_vec_3d_1, seriestype=:scatter, title = "3d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)
