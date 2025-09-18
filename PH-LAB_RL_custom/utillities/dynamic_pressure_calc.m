h_init = 2000
s = 24.2
[T,a,P,rho_init,nu,mu] = atmosisa(h_init)
q_init = 0.5*rho_init*(90^2)
q_alt = q_init * 5.5
rho_alt = q_alt*2/(200^2)
%Cl_init = mass / q
% find altitude that gives the desired density at the desired velocity
rho = 0;
h = h_init
delta_h = 0
while abs(rho-rho_alt) > 1e-5
    eror = abs(rho-rho_alt)
    if rho > rho_alt
        h = h+delta_h;
        [~,~,~,rho,~,~] = atmosisa(h);
    else
        h = h-delta_h;
        [~,~,~,rho,~,~] = atmosisa(h);
    end
    if eror > 1e-2
        delta_h = 10;
    elseif eror > 1e-3
        delta_h = 1;
    else
        delta_h = 0.001;
    end
end