

function barzilai_borwein(A::AbstractMatrix{Float64},b::AbstractVector{Float64},epsilon::Real,x::AbstractVector{Float64},option::Bool)

	start_time = time()
	k = 0
	d = b - A * x
	xp = zeros(eltype(x), size(x))
	dp = zeros(eltype(d), size(d))
	while (norm(d, 2) > epsilon)
		if k == 0
			# step = wolfe_search(x, d, simulator, a, m1, m2)
			step=1/norm(d, Inf)
		else
			s = x - xp
			y = dp - d

			if option
				num = s' * y
				denom = y' * y
				step = num / denom
			else
				num = s' * s
				denom = s' * y
				step = num / denom
			end
		end
		xp = x
		dp = d
		x += step * d
		d = b - A * x
		k += 1
	end
	elapsed = time() - start_time
	optimal_value = f(x, A, b)
	return x, elapsed, optimal_value, k
end
