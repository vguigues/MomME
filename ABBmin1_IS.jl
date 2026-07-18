
include("gradient_ImgSmooth.jl")

function ABBmin1_IS(xk::AbstractMatrix,y::AbstractMatrix,lambda::Real,mxitr::Int,tol::Real,m::Int )

vecBB2 = zeros(mxitr)
kappa  = 0.8
gk     = gradient_ImgSmooth(xk, y, lambda)
    

t_ini = time()
ng2 = sum(gk.*gk)
res = sqrt(ng2)
Agk   = gradient_ImgSmooth(gk, y, lambda) + y
lp = ng2/sum(gk.*Agk)

k = 1
while res > tol && k < mxitr
    lp_old = lp
    
    wk  = gradient_ImgSmooth(gk, y, lambda) + y
    xk = xk - lp*gk
    gtw = sum(gk.*wk) 
    alphaBB1 = ng2/gtw
    alphaBB2 = gtw/sum(wk.*wk)
    gk = gk - lp_old*wk
    
    vecBB2[k] = alphaBB2
    if alphaBB2 < kappa*alphaBB1
        if k <= m
           lp = minimum( vecBB2[1:k] ) 
        else
           lp = minimum( vecBB2[k-m:k] )
        end
    else
        lp = alphaBB1
    end
    
    k = k + 1
    ng2 = sum(gk.*gk)
    res = sqrt(ng2)
end
tf = time() - t_ini

return xk,tf, res, k-1

end