

function [f] = fun_ImgSmooth(X::Matrix{Float64}, Y::Matrix{Float64}, lambda::Float64)
   
    sum_{i,j} (X[i,j] - Y[i,j])^2 + lambda * (sum_{i,j} (X[i,j] - X[i-1,j])^2 + sum_{i,j} (X[i,j] - X[i,j-1])^2)

    fidelity = 0.5*sum((X - Y).^2)
    
    diff_v = X[2:end, :]-X[1:end-1, :]
    
    diff_h = X[:, 2:end] - X[:, 1:end-1]
    
    reg = 0.5 * lambda * (sum(diff_v.^2) + sum(diff_h.^2))
    f = fidelity + reg
end