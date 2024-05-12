function x_temp = bisectionMethodNew(gradient_omega, v, min_bis, max_bis, error_bisection, tt0, w, l, r, rho, u, maxIterBisection)
% FYI:  gradient_omega = @(omega, v, tt0, w, l, r, rho, u) tt0 + 5*(0.15*tt0/(w^4))*(omega^4) + l + rho*(omega-r*u-v);
% FYI:  omega(iter_omega) = bisectionMethodNew(gradient_omega, vTemp, min_bis, max_bis, error_bisection, tt0Temp, wTemp, l, r, rho, uTemp);
max_iter = maxIterBisection;
x_temp = (min_bis+max_bis)/2;

gradient_omega_value = gradient_omega(x_temp, v, tt0, w, l, r, rho, u);
counter = 0;
% Check gradient at min_bis and max_bis
gradient_temp_min = gradient_omega(min_bis, v, tt0, w, l, r, rho, u);
gradient_temp_max = gradient_omega(max_bis, v, tt0, w, l, r, rho, u);
if gradient_temp_min>0
    x_temp = min_bis;
elseif gradient_temp_max<0
    x_temp = max_bis;
else
%     while abs(max_bis-min_bis)>error_bisection
    while abs(gradient_omega_value)>error_bisection
        if gradient_omega_value < 0
            min_bis = x_temp;
        else
            max_bis = x_temp;
        end
        x_temp = (min_bis+max_bis)/2;
        gradient_omega_value = gradient_omega(x_temp, v, tt0, w, l, r, rho, u);
        counter = counter + 1;
        if counter > max_iter
            break
        elseif x_temp<1
            x_temp = 0;
            break
        end
    end
end