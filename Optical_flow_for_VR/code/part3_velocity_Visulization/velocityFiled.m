function [Qff,Q,F,V,TT,b,vectors] = velocityFiled(SImage, C, resulotion)
%This is used for vector interpolation
%SImage is the input Image
%C is the constraint vector list, each row is [faceIndex, u,v]
%resulotion decided how long is the arrows
clc
[m,n]=size(SImage);
[V,F,TT] = ConstructVFTT(SImage,resulotion);

%construct L matrix
[frows,fcols] = size(F);
Q =zeros(frows,frows);
for f = 1:1:frows
   for ei = 1:1:fcols
       g = TT(f,ei);
       if g~=-1 && f<g
          Q(f,f) = Q(f,f)+1;
          Q(f,g) = Q(f,g)-1;
          Q(g,f) = Q(g,f)-1;
          Q(g,g) = Q(g,g)+1;
       end
       
   end
end


% % construct constraints list and Q
[crows,ccol] = size(C);
Uc = [];
Qcf = [];
Qfc =[];
lcc=[]; lcr = [];
lfc=[1:1:frows];
lfr=lfc';
for i=1:1:crows
    u = C(i,2:3);
    Uc=[Uc;u];
    lcc = [lcc;C(i,1)];
    lcr = [lcr,C(i,1)];
    lfc(C(i,1))=-1;
    lfr(C(i,1))=-1;
 end
lfc = find(lfc~=-1);
lfr = find(lfr~=-1);

Qcc = Q(lcr,lcc);
Qff = Q(lfr,lfc);
Qfc = Q(lfr,lcc);
Qcf = Q(lcr,lfc);
% %interpulation 
 b= -Qfc*Uc;
 U = Qff\b;

 vectors = zeros(frows,2);
 for i=1:1:size(lfr)
     vectors(lfr(i),:)=U(i,:);
 end
 for i=1:1:size(lcc)
     vectors(lcc(i),:)=Uc(i,:);
 end
%  
%  %plot the line on each faces
figure
 imshow(SImage); hold on
 for i=1:1:frows-1
     grids = [F(i,1),F(i,2),F(i,3),F(i,4)];
       c1 = V(grids(1),:);
       c2 = V(grids(2),:);
       c3 = V(grids(3),:);
       c4 = V(grids(4),:);
      orign = (c1+c2+c3+c4)/4;
      quiver(orign(1),orign(2),vectors(i,1)*20*resulotion,vectors(i,2)*20*resulotion); 
 end

end