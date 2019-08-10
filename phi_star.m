function[y]=phi_star(X)
y=sum(sum((X/2).*log((1+X)./(1-X))+0.5*log(abs(1-X.^2))));
end