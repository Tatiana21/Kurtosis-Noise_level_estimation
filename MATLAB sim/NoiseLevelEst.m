% Estimating the noise level of input single noisy image
%
% [noiseSD, estimatedKurt, label] = NoiseLevelEst(Image,patchsize)
%
%Output parameters
% noiseSD: estimated noise levels. 
% estimatedKurt: estimated kurtosis for each cluster.
% label: the label for each unoverlapping square patch (used for image partition).
% 
%
%
%Input parameters
% Image: input single image
% patchsize: patch size (e.g., 8)
%

function  [noiseSD, estimatedKurt, label] = NoiseLevelEst(Image,patchsize)
    % ==== Parameters ====
    [NN1, NN2] = size(Image);
    nClass     = 3; %number of clusters
    gSize      = patchsize*2;
    Image = Image - mean2(Image); % remove DC 
    
    % ==== To form patch groups ====
    
    gLabelMat = zeros( floor((NN1-patchsize+1)/gSize)*gSize,floor((NN2-patchsize+1)/gSize)*gSize);
    mSize = size(gLabelMat,1)/gSize; 
    gLabelMat = blockproc(gLabelMat,[gSize,gSize], @(x) (floor(x.location(1)/gSize)+floor(x.location(2)/gSize)*mSize+1)*ones(gSize,gSize));
    temp = zeros(size(Image));
    temp(1:size(gLabelMat,1),1:size(gLabelMat,2)) = gLabelMat;
    gLabelMat  = temp;
    gLabelMat(NN1-patchsize+2:end,:) = [];gLabelMat(:,NN2-patchsize+2:end) = [];
    
    
    blkMatrix = im2col(Image,[patchsize,patchsize],'sliding'); % speed up by mex file
    gLabel = im2col(gLabelMat,[1,1],'sliding');
    
    numGroup = max(gLabel);
    patchGroup = cell(numGroup,1);
    
    for i = 1:numGroup
        patchGroup{i} = blkMatrix(:,gLabel==i);
    end


    % ==== Train a PCA basis UPCA ====
        muPCA = mean(blkMatrix,2);
        mBlkMat = bsxfun(@minus, blkMatrix, muPCA);
        [U,S] = eig(mBlkMat*mBlkMat'/(size(mBlkMat,2)-1));
        
        S = diag(S);
        [~,idx] = sort(S,'descend');
        UPCA = U(:,idx);

    % ==== For each patch group, obtain the kurtosis feature ====
    coeffGroup = cell(numGroup,1);
    elecoeff = 1;
    numFreq = patchsize^2 - elecoeff;
    kurtGroup = zeros(numGroup,numFreq);
    for i = 1:numGroup % can be accelrated with 'parfor'
        currGroup = cell2mat(patchGroup(i));
        data = bsxfun(@minus, currGroup, muPCA); 
        tempCoeff = UPCA'*data;
        currCoeff = tempCoeff((elecoeff+1):end,:);
        coeffGroup{i} = currCoeff(1:end,:)'; % each row is a band
         kurtGroup(i,:) = kurtosis(currCoeff(1:end,:),[],2)'; % along each filter band (each row is the kurtosis feature of a group);
    end

    % ==== Clustering patch group based on kurtosis feature via K-means ====
    if nClass == 1
        label = ones(size(kurtGroup,1),1);
    else
        label = litekmeans(kurtGroup,nClass,'Replicates',10); % each row is a sample
    end
    
    nClass = length(unique(label));
    

    % ==== Merge patch groups into different clusters ====
    %      For different clusters, compute  kurtosis and variance
    noiseVars = zeros(nClass, numFreq);
    noiseKurts = zeros(nClass,numFreq);
    for i = 1:nClass
        tempDctGroup= cell2mat(coeffGroup(label==i));
        noiseVars(i,:) = var(tempDctGroup);
        noiseKurts(i,:) = kurtosis(tempDctGroup);
    end
    
    % ==== higher kurtosis will get more weight ===
    allKurts = mean(noiseKurts,2);
    weight = allKurts./sum(allKurts);
    
    % ==== bi-convex problem Optimization ====
    % With the assumption piece-wise constant kurtosis for each cluster,
    % to optimize the objective function using alternating updating method
    [noiseSD, estimatedKurt] = bicvx_reg( noiseKurts, noiseVars, nClass, weight);
    noiseSD = abs(noiseSD);
    
end
