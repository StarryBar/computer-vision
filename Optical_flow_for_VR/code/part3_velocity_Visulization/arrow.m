function  arrow(P,V,color)
%draw arrow
% P=[x0,y0],V=[a,b]
%P is the orign of arrows
%[a,b]is the vector of arrows

if nargin < 3 
    color = 'b'; 
end
x0 = P(1);y0 = P(2); 
a = V(1); b = V(2); 
l = max(norm(V), eps);
u = [x0 x0+a]; v = [y0 y0+b]; 
hchek = ishold;

plot(u,v,color)
hold on 
h = l - min(.2*l, .2) ;v = min(.2*l/sqrt(3), .2/sqrt(3) );
a1 = (a*h -b*v)/l;
b1 = (b*h +a*v)/l; 

plot([x0+a1, x0+a], [y0+b1, y0+b], color) 

a2 = (a*h +b*v)/l; 
b2 = (b*h -a*v)/l; 

plot([x0+a2, x0+a], [y0+b2, y0+b], color)
if hchek == 0 
    hold off 
end