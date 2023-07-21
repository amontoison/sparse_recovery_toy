using LinearAlgebra;
using FFTW;
using SparseArrays;
using Random, Distributions;
include("mapping.jl")
include("Mtgeneration.jl")
include("Mperptz.jl")
include("fgNT.jl")

# backtracking line search to choose an appropriate step size
# @param t The parameter of central path in interior point method (scalar)
# @param beta_curr Current beta (a 1-dimensional vector)
# @param c_curr Current c (a 1-dimensional vector)
# (beta_curr, c_curr) is feasible
# @param delta_beta Newton direction in beta (a 1-dimensional vector)
# @param delta_c Newton direction in c (a 1-dimensional vector)
# @param paramLS A tuple consist of 3 entries: (alpha_LS, gamma_LS, paramf)
# alpha_LS The reduction fraction parameter (scalar, 0<alpha_LS<1)
# gamma_LS The update factor of t_NT parameter (scalar, 0<gamma_LS<1)
# paramf A tuple consist of 3 entries: (Mt, M_perptz, d)
# Mt M^{\top}: The transpose of M
# M_perptz M_{\perp}^{\top}*z
# d The constraint on the l1 norm of beta (scalar)


# @return t_NT The step size of line search (scalar)

function backtracking(t, beta_curr, c_curr, delta_beta, delta_c, paramLS)
    alpha_LS = paramLS[1];
    gamma_LS = paramLS[2];
    paramf = paramLS[3];

    t_NT = 1;
    f_curr = fval2(t, beta_curr, c_curr, paramf);
    #f_curr2 = fval2(t, beta_curr, c_curr, paramf);
    #println(norm(f_curr-f_curr2))
    gradb, gradc = fgrad2(t, beta_curr, c_curr, paramf);
    #gradb1, gradc1 = fgrad2(t, beta_curr, c_curr, paramf);
    #println("b:", norm(gradb - gradb1))
    #println("c:", norm(gradc - gradc1))

    beta_new = beta_curr;
    c_new = c_curr;
    l, u, h, g = auxiliary_func(beta_new, c_new, d);

    count = 0;

    # find a step size t_NT such that (beta_new, c_new) is feasible
    while(true)
        count = count + 1;
        if(count>1000)
            println("cann't make the new point feasible")
            break;
        end
        beta_new = beta_curr.+(t_NT.*delta_beta);
        c_new = c_curr.+(t_NT.*delta_c);
        l, u, h, g = auxiliary_func(beta_new, c_new, d);
        if (if_domain(l, u, h, g))
            break;
        else
            t_NT = gamma_LS * t_NT;
        end
    end

    # backtracking LS
    count = 0;
    while(true)
        count = count + 1;
        if(count>1000)
            println("can't find a step size")
            break;
        end
        # new beta & new c
        beta_new = beta_curr.+(t_NT.*delta_beta);
        c_new = c_curr.+(t_NT.*delta_c);

        f_new = fval2(t, beta_new, c_new, paramf);

        # check sufficient reduction condition
        if(f_new <= f_curr + alpha_LS*t_NT*sum((gradb'*delta_beta).+(gradc'*delta_c)))
            break;
        end
        t_NT = gamma_LS* t_NT;
    end

    return t_NT
end

# check whether (beta, c) is feasible
function if_domain(l, u, h, g)
    if ((minimum(Int.(l.<0))>0)&(minimum(Int.(u.<0))>0)&(minimum(Int.(h.<0))>0)&(g<0))
        return true
    else
        return false
    end
end
