function [V,F,C,I,J] = CreateContraintsList(UI, VI,resulotion,t)
    [V,F,TT] = ConstructVFTT(VI,resulotion);
    [m,n] = size(UI);
    l = m/resulotion;
    Um = UI - 0.5; Vm = VI - 0.5;
    [Ig,~] = Grad(Um,Vm);
    [frows, fcols]=size(F);
    I1=[];J1=[];I2=[];J2=[];I=[];J=[];
    [i1,j1] = find(Ig>=t);
    [i2,j2] = find(Ig<=-t);
    i2=[]; j2=[];
    I=[i1;i2];
    J = [j1;j2];
    [rows,~] = size(I);
    index= floor((I(1)-1)/resulotion) + floor((J(1)-1)/resulotion)*l+1;
    x = I(1); y = J(1);
    u = UI(x,y); v=VI(x,y); 
    C = [index,u, v,1];

   
     for i=2:1:rows
         index= floor((I(i)-1)/resulotion) + floor((J(i)-1)/resulotion)*l+1;
         if  ~ismember(index,C(:,1))
             x = I(i); y = J(i);
             u = Um(x,y); v=Vm(x,y);
%              [u,v] = norm([u,v]);
             c=[index, u, v,1];
             C = [C;c];
         else
             x = I(i); y = J(i);
             k = find(C(:,1)==index);
             u = Um(x,y); v=Vm(x,y);
             n = C(k,4); 
             mu = (n*C(k,2)+u)/(n+1);
             mv = (n*C(k,3)+v)/(n+1);
             n=n+1;
             c=[index, mu, mv,n];
             C(k,:) = c;
         end
     end
    
end