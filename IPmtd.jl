using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
include("mapping.jl")
include("Mtgeneration.jl")
include("Mperptz.jl")
include("fgNT.jl")
include("backtracking.jl")
include("Newton.jl")

# Interior point method

# It helps to find the constrained minimization problem.

# @param beta_init Starting beta (a 1-dimensional vector)
# @param c_init Starting c (a 1-dimensional vector)
# (beta_init, c_init) is feasible
# @param t_init Starting parameter in the central path (scalar)
# @paramB A tuple consist of 3 entries: (eps_barrier, mu_barrier, paramNT)
# eps_barrier Tolerance of interior point method (scalar)
# mu_barrier Update factor of the central path parameter t (scalar, mu>1)
# paramNT A tuple consist of 2 entries: (eps_NT, paramLS)
# eps_NT Tolerance in Newton step (scalar)
# paramLS A tuple consist of 3 entries: (alpha_LS, gamma_LS, paramf)
# alpha_LS The reduction fraction parameter (scalar, 0<alpha_LS<1)
# gamma_LS The update factor of t_NT parameter (scalar, 0<gamma_LS<1)
# paramf A tuple consist of 3 entries: (Mt, M_perptz, d)
# Mt M^{\top}: The transpose of M
# M_perptz M_{\perp}^{\top}*z
# d The constraint on the l1 norm of beta (scalar)

# @return beta, c The minimum of \phi at given t (two vectors)
# @return timeave The average time cost for the Newton step computation (scalar)

# @examples
# >Nt = 480;
# >t = collect(0:(Nt-1));
# >x1 = 2*cos.(2*pi*t*6/Nt).+ 3*sin.(2*pi*t*6/Nt);
# >x2 = 4*cos.(2*pi*t*10/Nt).+ 5*sin.(2*pi*t*10/Nt);
# >x3 = 6*cos.(2*pi*t*40/Nt).+ 7*sin.(2*pi*t*40/Nt);
# >x = x1.+x2.+x3; #signal
# >Random.seed!(1)
# >dist = Normal(0,1);
# >y = x + rand(dist, Nt); #noisy signal
# >DFTsize = size(x);
# >DFTdim = length(DFTsize);

# >m = 200;
# >index_nonmissing = sort(sample(1:Nt, Nt - m, replace = false));
# >index_missing = setdiff(collect(1:Nt), index_nonmissing);
# >z_zero = y;
# >z_zero[index_missing].= 0;
# >z = y[index_nonmissing];
# >M_perptz = M_perp_tz(z_zero, DFTdim, DFTsize);
# >Mt = generate_Mt(DFTdim, DFTsize, index_missing);

# >d = 420;
# >eps_barrier = 10e-6;
# >mu_barrier = 10;
# >alpha_LS = 0.1;
# >gamma_LS = 0.8;
# >eps_NT = 10e-6;

# >paramf = (Mt, M_perptz, d);
# >paramLS = (alpha_LS, gamma_LS, paramf);
# >paramNT = (eps_NT, paramLS);
# >paramB = (eps_barrier, mu_barrier, paramNT);

# >beta_init = zeros(Nt);
# >c_init = (d/(2*Nt)).*ones(Nt);
# >t_init = 1;

# >beta, c, timeave1d1 = barrier_mtd(beta_init, c_init, t_init, paramB)

function barrier_mtd(beta_init, c_init, t_init, paramB)
    eps_barrier = paramB[1];
    mu_barrier = paramB[2];
    paramNT = paramB[3];

    beta = beta_init;
    c = c_init;
    t = t_init;

    n = length(beta_init);
    timevec = Array{Float64, 1}[];
    println("barrier started")

    count = 0;
    while(true)
        count = count + 1;
        if(count>1000)
            println("barrier method doesn't terminate");
        end
        # Newton step
        beta, c, timevec = NT_mtd(t, beta, c, paramNT, timevec)

        # Check termination condition
        if(((3n+1)/t) < eps_barrier)
            break;
        end

        # Update t
        t = mu_barrier * t;

    end

    # compute the average time cost for the Newton step computation
    timeave = mean(timevec);

    return beta, c, timeave
end
