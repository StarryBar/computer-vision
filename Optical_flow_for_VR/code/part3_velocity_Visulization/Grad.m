function [Igrad, Iori] = Grad(Ix, Iy)
    [r,c] = size(Ix);
    Igrad = zeros(r,c);
    Iori= zeros(r,c);
    for i=1:1:r
        for j = 1:1:c
            Igrad(i,j) = sqrt(Ix(i,j).*Ix(i,j)+Iy(i,j).*Iy(i,j));
            if Ix(i,j) == 0
                if Iy(i,j) >0
                    Iori(i,j) = 90;
                elseif Iy < 0
                    Iori(i,j) = -90;
                else
                end
            else
                Iori(i,j) = atand(Iy(i,j)/Ix(i,j));
            end
        end
    end
    
end