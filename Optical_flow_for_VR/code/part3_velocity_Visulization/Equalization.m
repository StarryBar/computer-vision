function  [E] = Equalization(n)
    E = zeros(n,1);
    for i = 1:1:n
        E(i) = i/n;   
    end
end