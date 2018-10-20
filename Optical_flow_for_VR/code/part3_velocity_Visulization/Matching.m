function [S] = Matching(cdfr, cdf_bin_s, RImage)
    
    [m,n] = size(RImage);
    l =  length(cdf_bin_s)-1;
    S = zeros(m,n);
    for i = 1:1:m
        for j = 1:1:n
           intensity = RImage(i,j);   
           %intensity
           %floor(cdfr(intensity+1)*l)
           S(i,j) = cdf_bin_s(floor(cdfr(intensity+1)*l)+1);
        end
    end
end