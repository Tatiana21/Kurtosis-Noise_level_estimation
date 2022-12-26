% Demonstration example

clear; close all;
level = [5,15];
patchsize = 8;   

% dimg =double( imread('lena_gray.pgm'));
dimg =double( imread('traffic_gray.png'));

for i=1:size(level,2)

    noiseimg = dimg + randn(size(dimg))* level(i);

    tic;
    est     =      NoiseLevelEst(noiseimg,patchsize); 
    toc;
    fprintf('Given STD: %5.2f,  Estimated STD: %5.2f \n ', level(i), est);

end




