function [V,F,TT] = ConstructVFTT(SImage, resulotion)

[m,n]=size(SImage);
V = [];
F = [];
l = floor(max(m,n)/resulotion);

% construct F and V and TT
 for i = 0:resulotion:m
     for j=0:resulotion:n
        p = [i,j];
        V = [V;p];
     end
 end
 
 [nv,cv] = size(V);
 for i=0:1:nv-l-1
        if mod(i,l+1)~=l
        f = [i+1,i+2,l+i+2,l+i+3];
        F = [F;f];
     end
 end

  [frows,fcols] = size(F);
 TT = [];
 for i=1:1:frows
     tt = [];
        left = i-1;
        right = i+1;
        up = i+l;
        down = i-l;
     if mod(i,l)==0
        left = i-1;
        right = 0;
        up = i+l;
        down = i-l;
     end
     
     if mod(i,l)==1
        left = 0;
        right = i+1;
        up = i+l;
        down = i-l;
     end
     
     if left>0 && left <=frows
        tt = [tt,left];
     else
         tt = [tt,-1];
     end
     
     if right>0 && right <=frows
        tt = [tt,right];
     else
         tt = [tt,-1];
     end
     if up>0 && up <=frows
        tt = [tt,up];
     else
         tt = [tt,-1];
     end
     if down>0 && down <frows
        tt = [tt,down];
     else
         tt = [tt,-1];
     end
     TT = [TT;tt];
 end
end