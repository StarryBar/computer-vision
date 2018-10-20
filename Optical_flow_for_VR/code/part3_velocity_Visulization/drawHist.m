function [H] = drawHist(GreyMatrix, k)
    [m,n] = size(GreyMatrix);
    H = zeros(ceil(256/k), 1);
    for i = 1:1:m
        for j = 1:1:n
        intensity = GreyMatrix(i,j);
        H(floor(intensity/k)+1) = H(floor(intensity/k)+1) + 1; 
        %H(floor(GreyMatrix(i,j)/k) +1)= H(floor(GreyMatrix(i,j)/k) +1) + 1;
        end
    end

end