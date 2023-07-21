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


# Exp1: show the average time for Newton step is linear in n
# Fixed missing numbers m = 200
# 1 dim: n = 480, 960, 1920, 3200, 7680
# 2 dim: (n1, n2) = (24, 20), (24, 40), (48, 40), (64, 50), (96, 80)
# 3 dim: (n1,n2,n3) = (6,8,10), (12,8,10), (12,16,10), (20,8,20), (24,16,20)

# parameter choice
eps_barrier = 10e-6;
mu_barrier = 10;
alpha_LS = 0.1;
gamma_LS = 0.8;
eps_NT = 10e-6;

# unify all the parameters
function param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS)
    paramf = (Mt, M_perptz, d);
    paramLS = (alpha_LS, gamma_LS, paramf);
    paramNT = (eps_NT, paramLS);
    paramB = (eps_barrier, mu_barrier, paramNT);
    return paramB;
end

# MSE
function MSE(DFT, DFT_est)
    index_nz_DFT = findall(!iszero, DFT);
    mse = mean((norm.(DFT.-DFT_est)).^2);
    mse_sparsity = mean((norm.(DFT[index_nz_DFT].-DFT_est[index_nz_DFT])).^2);
    return mse, mse_sparsity
end

# identify sparsity
function sparseresult(DFT, DFT_est)
    DFT = round.(DFT, digits = 8);
    DFT_est = round.(DFT_est, digits = 8);

    # true nonzero DFT indices
    index_nz_DFT = findall(!iszero, DFT);
    true_nnz = length(index_nz_DFT);
    # nonzero DFTest indices
    index_nz_DFTest = findall(!iszero, DFT_est);
    est_nnz = length(index_nz_DFTest);

    # # of nonzero est/# of true nonzero DFT
    true_recovery_prop = length(findall(in(index_nz_DFT),index_nz_DFTest))/length(index_nz_DFT);
    # # of true nonzero est/# of all nonzero est
    false_finding_prop = (length(index_nz_DFTest)-length(findall(in(index_nz_DFT),index_nz_DFTest)))/length(index_nz_DFTest);

    return true_recovery_prop, false_finding_prop, true_nnz, est_nnz
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

# randomly generate missing indices
m = 200;
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

# unify parameters
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 420;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

beta, c, timeave1d1 = barrier_mtd(beta_init, c_init, t_init, paramB);
println("1d, N= 480, ave time = ", timeave1d1);
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse1d1, mse_sparsity1d1 = MSE(w, DFTest);
true_recovery_prop1d1, false_finding_prop1d1, true_nnz1d1, est_nnz1d1 = sparseresult(w, DFTest);


Nt = 960;
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

# randomly generate missing indices
m = 200;
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

# unify parameters
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 600;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

beta, c, timeave1d2 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("1d, N= 960, ave time = ", timeave1d2)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse1d2, mse_sparsity1d2 = MSE(w, DFTest);
true_recovery_prop1d2, false_finding_prop1d2, true_nnz1d2, est_nnz1d2 = sparseresult(w, DFTest);


Nt = 1920;
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

# randomly generate missing indices
m = 200;
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

# unify parameters
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 840;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

beta, c, timeave1d3 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("1d, N= 1920, ave time = ", timeave1d3)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse1d3, mse_sparsity1d3 = MSE(w, DFTest);
true_recovery_prop1d3, false_finding_prop1d3, true_nnz1d3, est_nnz1d3 = sparseresult(w, DFTest);


Nt = 7680;
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

# randomly generate missing indices
m = 200;
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

# unify parameters
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 1680;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

beta, c, timeave1d4 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("1d, N= 7680, ave time = ", timeave1d4)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse1d4, mse_sparsity1d4 = MSE(w, DFTest);
true_recovery_prop1d4, false_finding_prop1d4, true_nnz1d4, est_nnz1d4 = sparseresult(w, DFTest);


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

# randomly generate missing indices
m = 200;
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

# unify parameters
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 1080;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

beta, c, timeave1d5 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("1d, N= 3200, ave time = ", timeave1d5)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse1d5, mse_sparsity1d5 = MSE(w, DFTest);
true_recovery_prop1d5, false_finding_prop1d5, true_nnz1d5, est_nnz1d5 = sparseresult(w, DFTest);



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

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:Nt*Ns, Int(Nt*Ns - m), replace = false));
index_missing_Linear = collect(setdiff(collect(1:Nt*Ns), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, Nt*Ns, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 95;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

beta, c, timeave2d1 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("2d, N1 = 24, N2 = 20, ave time = ", timeave2d1)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse2d1, mse_sparsity2d1 = MSE(w, DFTest);
true_recovery_prop2d1, false_finding_prop2d1, true_nnz2d1, est_nnz2d1 = sparseresult(w, DFTest);


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

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:Nt*Ns, Int(Nt*Ns - m), replace = false));
index_missing_Linear = collect(setdiff(collect(1:Nt*Ns), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, Nt*Ns, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 130;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

beta, c, timeave2d2 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("2d, N1 = 24, N2 = 40, ave time = ", timeave2d2)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse2d2, mse_sparsity2d2 = MSE(w, DFTest);
true_recovery_prop2d2, false_finding_prop2d2, true_nnz2d2, est_nnz2d2 = sparseresult(w, DFTest);



Nt = 48;
Ns = 40;
t = collect(0:(Nt-1));
s = collect(0:(Ns-1));
x = (cos.(2*pi*2/Nt*t)+ 2*sin.(2*pi*2/Nt*t))*(cos.(2*pi*3/Ns*s) + 2*sin.(2*pi*3/Ns*s))';
Random.seed!(1)
y = x + randn(Nt,Ns)#noisy signal

w = round.(fft(x)./sqrt(Nt*Ns), digits = 4);#true DFT
index_nz_w = findall(!iszero, w);
[index_nz_w w[index_nz_w]]
sum(abs.(real.(w[index_nz_w])).+abs.(imag.(w[index_nz_w])))./sqrt(2)


DFTsize = size(x);
DFTdim = length(DFTsize);
Random.seed!(2)

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:Nt*Ns, Int(Nt*Ns - m), replace = false));
index_missing_Linear = collect(setdiff(collect(1:Nt*Ns), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, Nt*Ns, 1)[index_nonmissing_Linear];

# unify parameters
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 185;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

beta, c, timeave2d3 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("2d, N1 = 48, N2 = 40, ave time = ", timeave2d3)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse2d3, mse_sparsity2d3 = MSE(w, DFTest);
true_recovery_prop2d3, false_finding_prop2d3, true_nnz2d3, est_nnz2d3 = sparseresult(w, DFTest);


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
Random.seed!(2)

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:Nt*Ns, Int(Nt*Ns - m), replace = false));
index_missing_Linear = collect(setdiff(collect(1:Nt*Ns), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, Nt*Ns, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 370;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

beta, c, timeave2d4 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("2d, N1 = 96, N2 = 80, ave time = ", timeave2d4)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse2d4, mse_sparsity2d4 = MSE(w, DFTest);
true_recovery_prop2d4, false_finding_prop2d4, true_nnz2d4, est_nnz2d4 = sparseresult(w, DFTest);


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

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:Nt*Ns, Int(Nt*Ns - m), replace = false));
index_missing_Linear = collect(setdiff(collect(1:Nt*Ns), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, Nt*Ns, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 240;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(Nt*Ns);
c_init = (d/(2*Nt*Ns)).*ones(Nt*Ns);
t_init = 1;

beta, c, timeave2d5 = barrier_mtd(beta_init, c_init, t_init, paramB)
println("2d, N1 = 64, N2 = 50, ave time = ", timeave2d5)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse2d5, mse_sparsity2d5 = MSE(w, DFTest);
true_recovery_prop2d5, false_finding_prop2d5, true_nnz2d5, est_nnz2d5 = sparseresult(w, DFTest);


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

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:N1*N2*N3, N1*N2*N3 - m, replace = false));
index_missing_Linear = collect(setdiff(collect(1:N1*N2*N3), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, N1*N2*N3, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 225;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

timeIP1 = @elapsed begin
beta, c, timeave3d1 = barrier_mtd(beta_init, c_init, t_init, paramB);
end
println("3d, N1 = 6, N2 = 8, N3 = 10, ave time = ", timeave3d1)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse3d1, mse_sparsity3d1 = MSE(w, DFTest);
true_recovery_prop3d1, false_finding_prop3d1, true_nnz3d1, est_nnz3d1 = sparseresult(w, DFTest);

lambda = 35;
rho = 10;
timeADMM1 = @elapsed begin
z0 = cgADMM(Mt, M_perptz, lambda, rho)
end
DFTestADMM = beta_to_DFT(DFTdim, DFTsize, z0);
mseADMM3d1, mse_sparsityADMM3d1 = MSE(w, DFTestADMM);
true_recovery_propADMM3d1, false_finding_propADMM3d1, true_nnzADMM3d1, est_nnzADMM3d1 = sparseresult(w, DFTestADMM);




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

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:N1*N2*N3, N1*N2*N3 - m, replace = false));
index_missing_Linear = collect(setdiff(collect(1:N1*N2*N3), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, N1*N2*N3, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 320;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

timeIP2 = @elapsed begin
beta, c, timeave3d2 = barrier_mtd(beta_init, c_init, t_init, paramB);
end
println("3d, N1 = 12, N2 = 8, N3 = 10, ave time = ", timeave3d2)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse3d2, mse_sparsity3d2 = MSE(w, DFTest);
true_recovery_prop3d2, false_finding_prop3d2, true_nnz3d2, est_nnz3d2 = sparseresult(w, DFTest);

lambda = 2;
rho = 10;
timeADMM2 = @elapsed begin
z0 = cgADMM(Mt, M_perptz, lambda, rho)
end
DFTestADMM = beta_to_DFT(DFTdim, DFTsize, z0);
mseADMM3d2, mse_sparsityADMM3d2 = MSE(w, DFTestADMM);
true_recovery_propADMM3d2, false_finding_propADMM3d2, true_nnzADMM3d2, est_nnzADMM3d2 = sparseresult(w, DFTestADMM);


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

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:N1*N2*N3, N1*N2*N3 - m, replace = false));
index_missing_Linear = collect(setdiff(collect(1:N1*N2*N3), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, N1*N2*N3, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 450;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

timeIP3 = @elapsed begin
beta, c, timeave3d3 = barrier_mtd(beta_init, c_init, t_init, paramB);
end
println("3d, N1 = 12, N2 = 16, N3 = 10, ave time = ", timeave3d3)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse3d3, mse_sparsity3d3 = MSE(w, DFTest);
true_recovery_prop3d3, false_finding_prop3d3, true_nnz3d3, est_nnz3d3 = sparseresult(w, DFTest);

lambda = 2;
rho = 10;
timeADMM3 = @elapsed begin
z0 = cgADMM(Mt, M_perptz, lambda, rho)
end
DFTestADMM = beta_to_DFT(DFTdim, DFTsize, z0);
mseADMM3d3, mse_sparsityADMM3d3 = MSE(w, DFTestADMM);
true_recovery_propADMM3d3, false_finding_propADMM3d3, true_nnzADMM3d3, est_nnzADMM3d3 = sparseresult(w, DFTestADMM);


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

# randomly generate missing indices
m = 200;
Random.seed!(2);
index_nonmissing_Linear = sort(sample(1:N1*N2*N3, N1*N2*N3 - m, replace = false));
index_missing_Linear = collect(setdiff(collect(1:N1*N2*N3), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, N1*N2*N3, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 900;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

timeIP4 = @elapsed begin
beta, c, timeave3d4 = barrier_mtd(beta_init, c_init, t_init, paramB);
end
println("3d, N1 = 24, N2 = 16, N3 = 20, ave time = ", timeave3d4)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse3d4, mse_sparsity3d4 = MSE(w, DFTest);
true_recovery_prop3d4, false_finding_prop3d4, true_nnz3d4, est_nnz3d4 = sparseresult(w, DFTest);

lambda = 2;
rho = 10;
timeADMM4 = @elapsed begin
z0 = cgADMM(Mt, M_perptz, lambda, rho)
end
DFTestADMM = beta_to_DFT(DFTdim, DFTsize, z0);
mseADMM3d4, mse_sparsityADMM3d4 = MSE(w, DFTestADMM);
true_recovery_propADMM3d4, false_finding_propADMM3d4, true_nnzADMM3d4, est_nnzADMM3d4 = sparseresult(w, DFTestADMM);


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

# randomly generate missing indices
m = 200;
index_nonmissing_Linear = sort(sample(1:N1*N2*N3, N1*N2*N3 - m, replace = false));
index_missing_Linear = collect(setdiff(collect(1:N1*N2*N3), index_nonmissing_Linear));
index_nonmissing_Cartesian = map(i->CartesianIndices(y)[i], index_nonmissing_Linear);
index_missing_Cartesian = map(i->CartesianIndices(y)[i], index_missing_Linear);
z_zero = y;
z_zero[index_missing_Cartesian].= 0;
z = reshape(y, N1*N2*N3, 1)[index_nonmissing_Linear];

# unify parameter
M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing_Cartesian);
d = 580;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);

# init
beta_init = zeros(N1*N2*N3);
c_init = (d/(2*N1*N2*N3)).*ones(N1*N2*N3);
t_init = 1;

timeIP5 = @elapsed begin
beta, c, timeave3d5 = barrier_mtd(beta_init, c_init, t_init, paramB);
end
println("3d, N1 = 20, N2 = 16, N3 = 10, ave time = ", timeave3d5)
DFTest = beta_to_DFT(DFTdim, DFTsize, beta);
mse3d5, mse_sparsity3d5 = MSE(w, DFTest);
true_recovery_prop3d5, false_finding_prop3d5, true_nnz3d5, est_nnz3d5 = sparseresult(w, DFTest);

lambda = 2;
rho = 10;
timeADMM5 = @elapsed begin
z0 = cgADMM(Mt, M_perptz, lambda, rho)
end
DFTestADMM = beta_to_DFT(DFTdim, DFTsize, z0);
mseADMM3d5, mse_sparsityADMM3d5 = MSE(w, DFTestADMM);
true_recovery_propADMM3d5, false_finding_propADMM3d5, true_nnzADMM3d5, est_nnzADMM3d5 = sparseresult(w, DFTestADMM);


problemsize = [480; 960; 1920; 3200; 7680]
time1d = [timeave1d1; timeave1d2; timeave1d3; timeave1d5; timeave1d4];
time2d = [timeave2d1; timeave2d2; timeave2d3; timeave2d5; timeave2d4];
time3d = [timeave3d1; timeave3d2; timeave3d3; timeave3d5; timeave3d4];

plot(problemsize, [time1d time2d time3d], label = ["1d" "2d" "3d"], shape = [:circle :diamond :cross], title = "time against size", xlabel = "size", ylabel = "time", legend=:bottomright)

[problemsize time1d time2d time3d]


# mse
problemsize = [480; 960; 1920; 3200; 7680]
mse1d = [mse1d1; mse1d2; mse1d3; mse1d5; mse1d4];
mse2d = [mse2d1; mse2d2; mse2d3; mse2d5; mse2d4];
mse3d = [mse3d1; mse3d2; mse3d3; mse3d5; mse3d4];
[problemsize mse1d mse2d mse3d]

# mse of nonzero entries
mse_sparsity1d = [mse_sparsity1d1; mse_sparsity1d2; mse_sparsity1d3; mse_sparsity1d5; mse_sparsity1d4];
mse_sparsity2d = [mse_sparsity2d1; mse_sparsity2d2; mse_sparsity2d3; mse_sparsity2d5; mse_sparsity2d4];
mse_sparsity3d = [mse_sparsity3d1; mse_sparsity3d2; mse_sparsity3d3; mse_sparsity3d5; mse_sparsity3d4];
[problemsize mse_sparsity1d mse_sparsity2d mse_sparsity3d]

# true nnz
true_nnz1d = [true_nnz1d1; true_nnz1d2; true_nnz1d3; true_nnz1d5; true_nnz1d4];
true_nnz2d = [true_nnz2d1; true_nnz2d2; true_nnz2d3; true_nnz2d5; true_nnz2d4];
true_nnz3d = [true_nnz3d1; true_nnz3d2; true_nnz3d3; true_nnz3d5; true_nnz3d4];
[problemsize true_nnz1d true_nnz2d true_nnz3d]

# est nnz
est_nnz1d = [est_nnz1d1; est_nnz1d2; est_nnz1d3; est_nnz1d5; est_nnz1d4];
est_nnz2d = [est_nnz2d1; est_nnz2d2; est_nnz2d3; est_nnz2d5; est_nnz2d4];
est_nnz3d = [est_nnz3d1; est_nnz3d2; est_nnz3d3; est_nnz3d5; est_nnz3d4];
[problemsize est_nnz1d est_nnz2d est_nnz3d]

# true recovery prop (# of discovered nonzero in true nonzero/# of true nonzero)
true_recovery_prop1d = [true_recovery_prop1d1; true_recovery_prop1d2; true_recovery_prop1d3; true_recovery_prop1d5; true_recovery_prop1d4];
true_recovery_prop2d = [true_recovery_prop2d1; true_recovery_prop2d2; true_recovery_prop2d3; true_recovery_prop2d5; true_recovery_prop2d4];
true_recovery_prop3d = [true_recovery_prop3d1; true_recovery_prop3d2; true_recovery_prop3d3; true_recovery_prop3d5; true_recovery_prop3d4];
[problemsize true_recovery_prop1d true_recovery_prop2d true_recovery_prop3d]

# true recovery prop (# of discovered nonzero in true nonzero/# of true nonzero)
true_recovery_prop1d = [true_recovery_prop1d1; true_recovery_prop1d2; true_recovery_prop1d3; true_recovery_prop1d5; true_recovery_prop1d4];
true_recovery_prop2d = [true_recovery_prop2d1; true_recovery_prop2d2; true_recovery_prop2d3; true_recovery_prop2d5; true_recovery_prop2d4];
true_recovery_prop3d = [true_recovery_prop3d1; true_recovery_prop3d2; true_recovery_prop3d3; true_recovery_prop3d5; true_recovery_prop3d4];
[problemsize true_recovery_prop1d true_recovery_prop2d true_recovery_prop3d]

# false finding prop (# of discovered nonzero in true zero entries/# of discovered nonzero)
false_finding_prop1d = [false_finding_prop1d1; false_finding_prop1d2; false_finding_prop1d3; false_finding_prop1d5; false_finding_prop1d4];
false_finding_prop2d = [false_finding_prop2d1; false_finding_prop2d2; false_finding_prop2d3; false_finding_prop2d5; false_finding_prop2d4];
false_finding_prop3d = [false_finding_prop3d1; false_finding_prop3d2; false_finding_prop3d3; false_finding_prop3d5; false_finding_prop3d4];
[problemsize false_finding_prop1d false_finding_prop2d false_finding_prop3d]

# 3 dim interior point method
[problemsize mse3d mse_sparsity3d true_nnz3d est_nnz3d true_recovery_prop3d false_finding_prop3d]
# 3 dim ADMM
mseADMM3d = [mseADMM3d1; mseADMM3d2; mseADMM3d3; mseADMM3d5; mseADMM3d4];
mse_sparsityADMM3d = [mse_sparsityADMM3d1; mse_sparsityADMM3d2; mse_sparsityADMM3d3; mse_sparsityADMM3d5; mse_sparsityADMM3d4];
true_nnzADMM3d = [true_nnzADMM3d1; true_nnzADMM3d2; true_nnzADMM3d3; true_nnzADMM3d5; true_nnzADMM3d4];
est_nnzADMM3d = [est_nnzADMM3d1; est_nnzADMM3d2; est_nnzADMM3d3; est_nnzADMM3d5; est_nnzADMM3d4];
true_recovery_propADMM3d = [true_recovery_propADMM3d1; true_recovery_propADMM3d2; true_recovery_propADMM3d3; true_recovery_propADMM3d5; true_recovery_propADMM3d4];
false_finding_propADMM3d = [false_finding_propADMM3d1; false_finding_propADMM3d2; false_finding_propADMM3d3; false_finding_propADMM3d5; false_finding_propADMM3d4];
[problemsize mseADMM3d mse_sparsityADMM3d true_nnzADMM3d est_nnzADMM3d true_recovery_propADMM3d false_finding_propADMM3d]

timeIP = [timeIP1; timeIP2; timeIP3; timeIP5; timeIP4]
timeADMM = [timeADMM1; timeADMM2; timeADMM3; timeADMM4; timeADMM4]
[problemsize timeIP timeADMM]
# time cost compared with Ipopt
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


Nt = 50;
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

missprob = 0.05;
m = Int(floor(Nt*(missprob)))
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 100;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);


beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

timeIP1 = @elapsed begin
beta, c, timeave = barrier_mtd(beta_init, c_init, t_init, paramB);
end

M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing);
timeIpopt1 = @elapsed begin
beta_LASSO = op(z, M_perpt', d, Nt);
end
maximum(abs.(beta.-beta_LASSO))



Nt = 100;
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

missprob = 0.05;
m = Int(floor(Nt*(missprob)))
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 100;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);


beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

timeIP2 = @elapsed begin
beta, c, timeave = barrier_mtd(beta_init, c_init, t_init, paramB);
end

M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing);
timeIpopt2 = @elapsed begin
beta_LASSO = op(z, M_perpt', d, Nt);
end
maximum(abs.(beta.-beta_LASSO))

Nt = 150;
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

missprob = 0.05;
m = Int(floor(Nt*(missprob)))
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 100;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);


beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

timeIP3 = @elapsed begin
beta, c, timeave = barrier_mtd(beta_init, c_init, t_init, paramB);
end

M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing);
timeIpopt3 = @elapsed begin
beta_LASSO = op(z, M_perpt', d, Nt);
end
maximum(abs.(beta.-beta_LASSO))




Nt = 200;
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

missprob = 0.05;
m = Int(floor(Nt*(missprob)))
index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
index_missing = setdiff(collect(1:Nt), index_nonmissing);
z_zero = y;
z_zero[index_missing].= 0;
z = y[index_nonmissing];

M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
Mt = generate_Mt(DFTdim, DFTsize, index_missing);
d = 100;
paramB = param_unified(Mt, M_perptz, d, eps_barrier, mu_barrier, eps_NT, alpha_LS, gamma_LS);


beta_init = zeros(Nt);
c_init = (d/(2*Nt)).*ones(Nt);
t_init = 1;

timeIP4 = @elapsed begin
beta, c, timeave = barrier_mtd(beta_init, c_init, t_init, paramB);
end

M_perpt = generate_Mt(DFTdim, DFTsize, index_nonmissing);
timeIpopt4 = @elapsed begin
beta_LASSO = op(z, M_perpt', d, Nt);
end
maximum(abs.(beta.-beta_LASSO))

[50 100 150 200;
timeIP1 timeIP2 timeIP3 timeIP4;
timeIpopt1 timeIpopt2 timeIpopt3 timeIpopt4]
