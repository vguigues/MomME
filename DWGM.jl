function DWGM(A::Matrix, x::Vector, b::Vector, mxitr::Int, tol::Float64)
	# // % This function corresponds to an implementation of the Delayed 
	# // % Weighted Gradient Method proposed in .

	# // %-------------------------------------------------------------------------
	# // % This function solve the following problem
	# // %              min f(x) = 0.5 x'*A*x - x'*b, with x \in R^{n}  (1)
	# // %  where A is a n-by-n symmetric and positive definite matrix and b \in R^{n} 
	# // %  is a given vector.  This problem is equivalent to solve the linear
	# // %  system of equation: Ax = b.
	# // %-------------------------------------------------------------------------

	# // % INPUT: 
	# // %        A -- is a n-by-n symmetric and positive definite matrix.
	# // %        x -- is the initial point.
	# // %        b -- the right side vector of the linear system Ax = b. 
	# // %    mxitr -- is the maximun number of iterations.
	# // %      tol -- is the tolerance for the gradient norm criterion.

	# // % OUTPUT:
	# // %        x -- the aproximate solution of (1)
	# // %      out -- a structure with output information.
	# // %     grad -- is a vector that which contains the history of the gradient 
	# // %             norm along the iterations.

	# // % Reference:
	# // % ----------
	# // % Harry F. Oviedo  "A Delayed Weighted Gradient Method for Strictly 
	# // %                   Convex Quadratic Minimization"
	# // %
	# // % Author:
	# // % ----------
	# // % Harry F. Oviedo <harry.oviedo@cimat.mx>
	# // %-------------------------------------------------------------------------

	xk = x

	t_ini = time()
	gk    = A*xk - b
	res   = norm(gk)
	xold  = xk
	gold  = gk

	k = 1
	while res > tol && k < mxitr
		xkp     = xk
		gkp     = gk
		wk      = A*gk
		lp      = (gk'*wk)/(wk'*wk)
		xcauchy = xk - lp*gk
		gcauchy = gk - lp*wk
		pk      = gcauchy - gold
		omega   = 1 - (gcauchy'*pk)/(pk'*pk)
		xk      = xold + omega*(xcauchy - xold)
		gk      = gold + omega*pk
		xold    = xkp
		gold    = gkp
		k       = k + 1
		res     = norm(gk)
	end
	tf = time()-t_ini

	return xk, k, tf, res


end
