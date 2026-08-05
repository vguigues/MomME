

using LinearAlgebra
include("util.jl")


function conjugate_gradient(A::AbstractMatrix{Float64},b::AbstractVector{Float64}, x::AbstractVector{Float64}, tol::Real, mxitr::Int)
	
	
	
start_time = time()
r = A*x - b
p = -r
nrmR = r'*r
k = 1

while sqrt(nrmR) > tol && k < mxitr 
    Ap = A*p
    nrmRk = nrmR
    alpha = nrmRk/(p'*Ap)
    x = x + alpha*p 
    r = r + alpha*(Ap)
    nrmR = r'*r
    beta = nrmR/nrmRk   
    
    p = -r + beta*p
    
    k = k + 1
end
elapsed = time() - start_time
optimal_value = f(x, A, b)
return x, elapsed, optimal_value, k
end
