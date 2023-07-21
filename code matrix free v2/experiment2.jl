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

# 1 dim
Nt = 480;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(1)
dist = Normal(0,1);
y = x + rand(dist, Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #418

missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y);


# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 300;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

# barrier method
beta, c, t_vec_1d_1, iter_vec_1d_1, timevec_1d_1, count_nt_1d_1 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_1d_1 = plot(log.(t_vec_1d_1), iter_vec_1d_1, seriestype=:scatter, title = "1d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)


Nt = 960;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(2)
dist = Normal(0,1);
y = x + rand(dist, Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #592

missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y);


# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 400;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

# barrier method
beta, c, t_vec_1d_2, iter_vec_1d_2, timevec_1d_2, count_nt_1d_2 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_1d_2 = plot(log.(t_vec_1d_2), iter_vec_1d_2, seriestype=:scatter, title = "1d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



Nt = 1920;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(2)
dist = Normal(0,1);
y = x + rand(dist, Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #837

missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y);


# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 600;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

# barrier method
beta, c, t_vec_1d_3, iter_vec_1d_3, timevec_1d_3, count_nt_1d_3 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_1d_3 = plot(log.(t_vec_1d_3), iter_vec_1d_3, seriestype=:scatter, title = "1d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)


Nt = 7680;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(2)
dist = Normal(0,1);
y = x + rand(dist, Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #1673

missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y);


# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 1000;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

# barrier method
beta, c, t_vec_1d_4, iter_vec_1d_4, timevec_1d_4, count_nt_1d_4 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_1d_4 = plot(log.(t_vec_1d_4), iter_vec_1d_4, seriestype=:scatter, title = "1d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



Nt = 3200;
t = collect(0:(Nt-1));
x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
x = x1.+x2.+x3; #signal
Random.seed!(2)
dist = Normal(0,1);
y = x + rand(dist, Nt); #noisy signal

w = round.(fft(x)./sqrt(Nt), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #1079

missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1
index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y);


# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 700;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

# barrier method
beta, c, t_vec_1d_5, iter_vec_1d_5, timevec_1d_5, count_nt_1d_5 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_1d_5 = plot(log.(t_vec_1d_5), iter_vec_1d_5, seriestype=:scatter, title = "1d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



### 2dim
Nt = 24;
Ns = 20;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)#noisy signal

w = round.(fft(x)./sqrt(Nt*Ns), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #93

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 70;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

# barrier method
beta, c, t_vec_2d_1, iter_vec_2d_1, timevec_2d_1, count_nt_2d_1 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_2d_1 = plot(log.(t_vec_2d_1), iter_vec_2d_1, seriestype=:scatter, title = "2d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



Nt = 24;
Ns = 40;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)#noisy signal

w = round.(fft(x)./sqrt(Nt*Ns), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #131

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 100;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

# barrier method
beta, c, t_vec_2d_2, iter_vec_2d_2, timevec_2d_2, count_nt_2d_2 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_2d_2 = plot(log.(t_vec_2d_2), iter_vec_2d_2, seriestype=:scatter, title = "2d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)


Nt = 48;
Ns = 40;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)#noisy signal

w = round.(fft(x)./sqrt(Nt*Ns), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #186

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 150;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

# barrier method
beta, c, t_vec_2d_3, iter_vec_2d_3, timevec_2d_3, count_nt_2d_3 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_2d_3 = plot(log.(t_vec_2d_3), iter_vec_2d_3, seriestype=:scatter, title = "2d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



Nt = 96;
Ns = 80;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)#noisy signal

w = round.(fft(x)./sqrt(Nt*Ns), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #372

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 250;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

# barrier method
beta, c, t_vec_2d_4, iter_vec_2d_4, timevec_2d_4, count_nt_2d_4 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_2d_4 = plot(log.(t_vec_2d_4), iter_vec_2d_4, seriestype=:scatter, title = "2d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



Nt = 64;
Ns = 50;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)#noisy signal

w = round.(fft(x)./sqrt(Nt*Ns), digits = 4);#true DFT
DFTsize = size(x);
DFTdim = length(DFTsize);
beta_true = DFT_to_beta(DFTdim, DFTsize, w);
sum(abs.(beta_true)) #240

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 200;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

# barrier method
beta, c, t_vec_2d_5, iter_vec_2d_5, timevec_2d_5, count_nt_2d_5 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_2d_5 = plot(log.(t_vec_2d_5), iter_vec_2d_5, seriestype=:scatter, title = "2d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



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

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 200;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

# barrier method
beta, c, t_vec_3d_1, iter_vec_3d_1, timevec_3d_1, count_nt_3d_1 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_3d_1 = plot(log.(t_vec_3d_1), iter_vec_3d_1, seriestype=:scatter, title = "3d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)




N1 = 12;
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
sum(abs.(beta_true)) #318

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 250;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

# barrier method
beta, c, t_vec_3d_2, iter_vec_3d_2, timevec_3d_2, count_nt_3d_2 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_3d_2 = plot(log.(t_vec_3d_2), iter_vec_3d_2, seriestype=:scatter, title = "3d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)


N1 = 12;
N2 = 16;
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
sum(abs.(beta_true)) #449

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 380;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

# barrier method
beta, c, t_vec_3d_3, iter_vec_3d_3, timevec_3d_3, count_nt_3d_3 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_3d_2 = plot(log.(t_vec_3d_3), iter_vec_3d_3, seriestype=:scatter, title = "3d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)


N1 = 24;
N2 = 16;
N3 = 20;

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
sum(abs.(beta_true)) #899

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 800;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

# barrier method
beta, c, t_vec_3d_4, iter_vec_3d_4, timevec_3d_4, count_nt_3d_4 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_3d_4 = plot(log.(t_vec_3d_4), iter_vec_3d_4, seriestype=:scatter, title = "3d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



N1 = 20;
N2 = 8;
N3 = 20;

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
sum(abs.(beta_true)) #580

# randomly generate missing indices
missing_prob = 0.15;
centers = center(DFTdim, DFTsize, missing_prob);
radius = 1

index_missing, z_zero = punching(DFTdim, DFTsize, centers, radius, y)

# unify parameter for barrier method
M_perptz = M_perp_tz(DFTdim, DFTsize, z_zero); # M_perptz
d = 500;
paramB = param_unified(DFTdim, DFTsize, M_perptz, d, index_missing, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# initial values for barrier method
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

# barrier method
beta, c, t_vec_3d_5, iter_vec_3d_5, timevec_3d_5, count_nt_3d_5 = barrier_mtd(beta_init, c_init, t_init, paramB);
# this is the average time of computing Newton's direction
plot_3d_5 = plot(log.(t_vec_3d_5), iter_vec_3d_5, seriestype=:scatter, title = "3d case", xlabel = "log(t_barrier)", ylabel = "CG_iter", label = false)



time_1d = [mean(timevec_1d_1); mean(timevec_1d_2); mean(timevec_1d_3); mean(timevec_1d_5); mean(timevec_1d_4)];
time_2d = [mean(timevec_2d_1); mean(timevec_2d_2); mean(timevec_2d_3); mean(timevec_2d_5); mean(timevec_2d_4)];
time_3d = [mean(timevec_3d_1); mean(timevec_3d_2); mean(timevec_3d_3); mean(timevec_3d_5); mean(timevec_3d_4)];

count_1d = [count_nt_1d_1; count_nt_1d_2; count_nt_1d_3; count_nt_1d_5; count_nt_1d_4];
count_2d = [count_nt_2d_1; count_nt_2d_2; count_nt_2d_3; count_nt_2d_5; count_nt_2d_4];
count_3d = [count_nt_3d_1; count_nt_3d_2; count_nt_3d_3; count_nt_3d_5; count_nt_3d_4];

problemsize = [480; 960; 1920; 3200; 7680]
plot(problemsize, [time_1d time_2d time_3d], label = ["1d" "2d" "3d"], shape = [:circle :diamond :cross], title = "time against size", xlabel = "size", ylabel = "time", legend=:bottomright)

[count_1d count_2d count_3d]
