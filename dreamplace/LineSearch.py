##
# @file   LineSearch.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Linear search functions. 
#

import math
import torch  
import logging
import pdb 

def build_line_search_fn_armijo(obj_fn):
    """
    @brief initialization
    @param obj_fn a callable function to evaluate the objective given input parameters 
    """
    def line_search_fn(xk, pk, gfk, fk, alpha0, c1=1e-4):
        """
        @brief line search 
        @param xk current point  
        @param pk search direction 
        @param gfk gradient of f at xk  
        @param fk value of f at xk, will evaluate if None  
        @param alpha0 initial step size 
        @return step size 
        """
        return line_search_armijo(f=obj_fn, xk=xk, pk=pk, gfk=gfk, old_fval=fk, alpha0=alpha0, c1=c1)
    return line_search_fn

#------------------------------------------------------------------------------
# Armijo line and scalar searches (modified from scipy)
# https://github.com/scipy/scipy/blob/master/scipy/optimize/linesearch.py
#------------------------------------------------------------------------------

def line_search_armijo(f, xk, pk, gfk, old_fval, c1=1e-4, alpha0=1, max_backtrack_count=100):
    """
    @brief Minimize over alpha, the function ``f(xk+alpha pk)``.
        Uses the interpolation algorithm (Armijo backtracking) as suggested by
        Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57
    @param f : callable
        Function to be minimized.
    @param xk : array_like
        Current point.
    @param pk : array_like
        Search direction.
    @param gfk : array_like
        Gradient of `f` at point `xk`.
    @param old_fval : float
        Value of `f` at point `xk`.
    @param c1 : float, optional
        Value to control stopping criterion.
    @param alpha0 : scalar, optional
        Value of `alpha` at start of the optimization.
    @param max_backtrack_count: scalar, optional 
        maximum number of trials 
    @return alpha, f_count, f_val_at_alpha
    """
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1*pk)

    if old_fval is None:
        phi0 = phi(0.0)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    derphi0 = gfk.dot(pk)
    alpha, phi1, count = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                       alpha0=alpha0, max_backtrack_count=max_backtrack_count)

    return alpha, fc[0], phi1


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0, max_backtrack_count=10):
    """
    @brief Minimize over alpha, the function ``phi(alpha)``.
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57
    alpha > 0 is assumed to be a descent direction.
    @param phi 
    @param phi0 
    @param derphi0 
    @param c1 
    @param alpha0 
    @param amin 
    @param max_backtrack_count
    @return alpha, phi1
    """
    count = 0

    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, count

    # Otherwise compute the minimizer of a quadratic interpolant:

    # alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    alpha1 = -derphi0.mul(alpha0.pow(2)).div(2.0).div(phi_a0.sub(phi0).sub(derphi0.mul(alpha0)))
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1, count

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    # original assumption: we are assuming alpha>0 is a descent direction
    # Yibo: I change to also handle negative alpha1 
    while alpha1 != amin and count < max_backtrack_count:       
        factor = alpha0.pow(2) * alpha1.pow(2) * (alpha1-alpha0)
        a = alpha0.pow(2) * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1.pow(2) * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0.pow(3) * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1.pow(3) * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        # (-b + np.sqrt(abs(b.pow(2) - 3 * a * derphi0))) / (3.0*a)
        alpha2 = (b.pow(2).sub(a.mul(derphi0).mul(3))).abs().sqrt().sub(b).div( a.mul(3.0) )
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, count

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

        count += 1

    # Failed to find a suitable step length
    return alpha1, phi_a1, count

"""
Use the code from wikipedia 
https://en.wikipedia.org/wiki/Golden-section_search
"""
def build_line_search_fn_golden_section(obj_fn):
    """
    @brief initialization
    @param obj_fn a callable function to evaluate the objective given input parameters 
    """
    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
    def line_search_fn(xk, pk, gfk, fk, alpha_min, alpha_max, tol=1e-1):
        """
        Given a function f with a single local minimum in
        the interval [a,b], gss returns a subset interval
        [c,d] that contains the minimum with d-c <= tol.

        @brief line search 
        @param xk current point  
        @param pk search direction 
        @param gfk gradient of f at xk, can be None  
        @param fk value of f at xk, can be None
        @param alpha_min minimum alpha 
        @param alpha_max maximum alpha 
        @param tol tolerance 
        @return step size 
        """
        def f(lr):
            return obj_fn(xk + pk * lr)

        a = alpha_min 
        b = alpha_max

        (a, b) = (min(a, b), max(a, b))
        h = b - a
        if h <= tol:
            alpha = (a + b) / 2
            return alpha, n-1, f(alpha)

        # Required steps to achieve tolerance
        n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

        c = a + invphi2 * h
        d = a + invphi * h
        yc = f(c)
        yd = f(d)

        for k in range(n-1):
            if yc < yd:
                b = d
                d = c
                yd = yc
                h = invphi * h
                c = a + invphi2 * h
                yc = f(c)
            else:
                a = c
                c = d
                yc = yd
                h = invphi * h
                d = a + invphi * h
                yd = f(d)

        if yc < yd:
            alpha = (a + d) / 2
        else:
            alpha = (c + b) / 2
        return alpha, n-1, f(alpha)
    return line_search_fn
