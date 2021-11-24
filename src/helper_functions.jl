
# this file contains utility functions


all_default_at(a, bi, yi::Int) = get_d_pol(a)[bi, yi] <= get_m(a.model).m_min 
all_default_at(a, bi, (yi, yi0)) = get_d_pol(a)[bi, yi, yi0] <= get_m(a.model).m_min 

