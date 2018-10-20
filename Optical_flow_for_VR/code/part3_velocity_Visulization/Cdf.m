function [C] = Cdf(P)

 l = length(P);
 C = zeros(l, 1);
 C(1) = P(1);
 for i = 2:1:l
        C(i) = P(i)+C(i-1); 
 end
end
 
 