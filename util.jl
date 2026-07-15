
using LinearAlgebra


function f(x, A, b)
	return 0.5 * x' * A * x - b' * x
end

function sintetic_data(n::Int, kappa::Float64)

	w1 = rand(n)
	w1 = w1/norm(w1)
	w2 = rand(n)
	w2 = w2/norm(w2)
	w3 = rand(n)
	w3 = w3/norm(w3)
	Id = I(n)
	P = (Id - 2*(w3*w3'))*(Id - 2*(w2*w2'))*(Id - 2*(w1*w1'))
	D = zeros(n, n)
	vdiag=collect(0:(n-1)) ./ (n-1)
	D=Diagonal(exp.(vdiag * kappa))

	A     = P*D*P'
	x0    = zeros(n)
	x_opt = 2*rand(n) - ones(n)
	b     = A*x_opt
	condA = D[n, n]/D[1, 1]

	return A, b, x0

end

function two_point_boundary_value(n::Int)

	h = 11/n
	A = zeros(n, n)
	for i in 1:n
		A[i, i] = 2/h^2
		if i-1 >= 1
			A[i, i-1] = -1/(h^2)
		end

		if i+1 <= n
			A[i, i+1] = -1/(h^2)
		end
	end
	b  = -1 .+ 2*rand(n)
	x0 = zeros(n)

	return A, b, x0

end
