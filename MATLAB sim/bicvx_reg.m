function [ noiseSD, estimatedKurt] = bicvx_reg( noiseKurts, noiseVars, nClass, weight)
%Alternating Optimization function for sub-problems

if nargin < 4
     weight = ones(nClass,1)./nClass;
end

weight = sqrt(weight);

noiseKurts = max(noiseKurts - 3,0); % excess kurtosis. 

A = sqrt(noiseKurts);
B = 1./noiseVars;
m = nClass;
n = size(A, 2) ;

MAX_ITERS = 25;
y = mean(noiseVars(:)); % initialize the estimated noise variance
options = optimset('Display','off','Algorithm','interior-point-convex');
warning('off','all');

L = zeros(nClass,nClass) -1;
L = diag(ones(nClass,1)*(nClass)) + L;
beta = 0.01;

R = beta*L;
for iter = 1:MAX_ITERS
        
        if mod(iter,2) == 1
           C = repmat(weight,1,n).*(1-y*B);
           D = repmat(weight,1,n).*A;
           H = diag(sum(C.^2,2));
           H = (H - R);
           f = - sum(C.*D,2);
           x = quadprog(H,f,[],[],[],[],sqrt(mean(noiseKurts,2)),ones(m,1)*Inf,[],options);
        else
           y = sum(sum((repmat(x,1,n) - A).*B.*repmat(x,1,n)))/(sum(sum(B.^2.*repmat(x,1,n).^2)));
        end
end

noiseSD = sqrt(y);
estimatedKurt = x.^2;
end

