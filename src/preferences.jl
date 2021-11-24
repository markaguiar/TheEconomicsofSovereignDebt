
(::Log)(x) = log(x)
(u::CRRA2)(x) = -1 / x
(u::CRRA)(x) = u.uf(x) 
(u::Power)(x) = x^u.power / u.power

# first derivatives
u_prime(::Log, x) = 1 / x
u_prime(::CRRA2, x) = 1 / x^2
u_prime(u::CRRA, x) = u_prime(u.uf, x)
u_prime(u::Power, x) = x^(u.power - 1)

# second derivatives
u_prime_prime(::Log, x) = - 1 / x^2
u_prime_prime(::CRRA2, x) = - 2 / x^3
u_prime_prime(u::CRRA, x) = u_prime_prime(u.uf, x)
u_prime_prime(u::Power, x) = (u.power - 1) * x^(u.power - 2)


inv_u(::Log, x) = exp(x)
inv_u(::CRRA2, x) = -1 / x
inv_u(u::CRRA, x) = inv_u(u.uf, x)
inv_u(u::Power, x) = (u.power * x)^(u.inv)
