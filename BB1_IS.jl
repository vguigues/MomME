
include("gradient_ImgSmooth.jl")

function BB1_IS( xk::AbstractMatrix,y::AbstractMatrix,lambda::Real,mxitr::Int,tol::Real )

t_ini = time()
gk    = gradient_ImgSmooth(xk, y, lambda)
nrmG  = norm(gk)^2
Agk   = gradient_ImgSmooth(gk, y, lambda) + y
alpha = nrmG/sum(gk.*Agk)
k = 1

while sqrt(nrmG) > tol && k < mxitr 
    wk = gradient_ImgSmooth(gk, y, lambda) + y           
    alpha_old = alpha  
    alpha = nrmG/sum(gk.*wk)
    xk = xk - alpha_old*gk 
    gk = gk - alpha_old*wk
    nrmG  = norm(gk)^2
    k = k + 1
end
tf = time() - t_ini;

return xk,tf, sqrt(nrmG), k-1

end

