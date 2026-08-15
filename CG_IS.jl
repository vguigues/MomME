

include("gradient_ImgSmooth.jl")

function CG_IS(x::AbstractMatrix,y::AbstractMatrix,lambda::Real,mxitr::Int,tol::Real)

t_ini = time()
r    = gradient_ImgSmooth(x, y, lambda)
p    = -r

nrmR = norm(r)^2
k = 1
while sqrt(nrmR) > tol && k < mxitr 
    Ap    = gradient_ImgSmooth(p, y, lambda) + y
    nrmRk = nrmR
    alpha = nrmRk/sum(sum(p.*Ap))
    x     = x + alpha*p
    r     = r + alpha*(Ap)
    nrmR  = norm(r)^2
    beta  = nrmR/nrmRk   
    p     = -r + beta*p
    k     = k + 1
end
tf = time() - t_ini

return x,tf, sqrt(nrmR), k-1

end
