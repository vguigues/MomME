function gradient_ImgSmooth(X::Matrix{Float64}, Y::Matrix{Float64}, lambda::Float64)
    
    n, m = size(X)
    LX = zeros(n, m)

    diff_v = X[2:end, :] - X[1:end-1, :]
    
    diff_h = X[:, 2:end] - X[:, 1:end-1]
    
    LX[1:end-1, :] = LX[1:end-1, :] - diff_v
    LX[2:end, :]   = LX[2:end, :]   + diff_v
    
    LX[:, 1:end-1] = LX[:, 1:end-1] - diff_h
    LX[:, 2:end]   = LX[:, 2:end]   + diff_h
    
    g = (X - Y) + lambda * LX
    
end