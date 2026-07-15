function ABBmin1(A::AbstractMatrix, x::AbstractVector, b::AbstractVector, mxitr::Integer, tol::Real, m::Integer)

	vecBB2 = zeros(mxitr)
	kappa = 0.8
	xk = x
	gk = A*xk - b

	t_ini = time()
	ng2 = gk'*gk
	res = sqrt(ng2)
	lp = ng2/(gk'*(A*gk))

	k = 1;
	while res > tol && k < mxitr
		lp_old = lp

		wk = A*gk
		xk = xk - lp*gk
		gtw = gk'*wk
		alphaBB1 = ng2/gtw
		alphaBB2 = gtw/(wk'*wk)
		gk = gk - lp_old*wk

		vecBB2[k] = alphaBB2
		if alphaBB2 < kappa*alphaBB1
			if k <= m
				lp = minimum(vecBB2[1:k])
			else
				lp = minimum(vecBB2[(k-m):k])
			end
		else
			lp = alphaBB1
		end

		k = k + 1
		ng2 = gk'*gk
		res = sqrt(ng2)
	end
	tf = time() - t_ini

	return xk, k, tf, res

end
