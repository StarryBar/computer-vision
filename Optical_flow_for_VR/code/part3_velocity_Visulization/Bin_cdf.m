function [B] = Bin_cdf(cdf, n)

        B = zeros(n+1,1);
        N = zeros(n+1,1);
        l = length(cdf);
        
        %using the mean value as the mapping intensity
        for i = 1:1:l
            b = floor(cdf(i) * n)+1;          
            B(b) = (B(b) * N(b) + i)/(N(b)+1);
            N(b) = N(b) + 1;
        end
end