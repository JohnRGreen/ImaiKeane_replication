def find_M_for_K(K, coefficients):
    b0, b1, b2, b3 = coefficients
    
    a = b2
    b = b1
    c = b0 + b3*K - 1
    
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None  # No real solutions
    elif discriminant == 0:
        # One solution
        return -b / (2*a)
    else:
        # Two solutions
        M1 = (-b + np.sqrt(discriminant)) / (2*a)
        M2 = (-b - np.sqrt(discriminant)) / (2*a)
        return M1, M2