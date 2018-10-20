function [TImage,T]=BackwardTrasformNN(SImage, theata, sx, sy, tx, ty,sv,sh)    
    [m,n] = size(SImage);
    TImage = zeros(m,n);
    %for coordinate 
     Tc = [0,   1,    -n/2;
           -1,   0,    m/2;
           0,   0,     1];
    %for transform
     Rf = [cos(theata), -sin(theata),0;
          sin(theata), cos(theata),  0;
          0,            0,           1
         ];
     Tf = [1,   0,    tx;
           0,   1,    ty;
           0,   0,    1];
     Sf = [sx,  0,     0;
            0,   sy,   0;
            0,   0,    1];
     Shf=  [1,   sh,     0;
            sv,   1,   0;
            0,   0,    1];
      %transform that orign in center of the image
      Trans =Tf*Shf*Rf*Sf;
      T = inv(Tc)*Trans*Tc;
    for i=1:1:m
        for j=1:1:n
            pb = [i;j;1];
            pbb = Tc*pb;%change basis to center of the image
            pff = inv(Trans) * pbb;
            pf =  inv(Tc)*pff;%change basis back
            x = round(pf(1),0);
            y = round(pf(2),0);
            if x<m&x>0&y<n&y>0
                TImage(i,j) = SImage(x,y); 
            end                      
        end
    end

end