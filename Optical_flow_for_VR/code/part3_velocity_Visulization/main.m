% UI = imread('TrueResult/line45dg/result/U8.jpg');
% UI=rgb2gray(UI);
% UI=im2double(UI);
% 
% VI = imread('TrueResult/line45dg/result/V8.jpg');
% VI=rgb2gray(VI);
% VI=im2double(VI);
% 
% [vs,fs,C,I,J] = CreateContraintsList(VI,UI,30,0.02);
% [Qff,Q,F,V,TT,b,vectors]=velocityFiled(UI,C,30,'line45');
s=1;e=19
for k=s:1:e
    uname =strcat('TrueResult/ver270dg/result/U',strcat(num2str(k),'.jpg'));
    vname =strcat('TrueResult/ver270dg/result/V',strcat(num2str(k),'.jpg'));
UI = imread(uname);
UI=rgb2gray(UI);
UI=im2double(UI);

VI = imread(vname);
VI=rgb2gray(VI);
VI=im2double(VI);

[vs,fs,C,I,J] = CreateContraintsList(VI,UI,60,0.02);
[Qff,Q,F,V,TT,b,vectors]=velocityFiled(UI,C,30);
end








