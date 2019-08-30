function saen = samp_entropy(dim,r,data)
%samp_entropy 
%   calculates the sample entropy of a given time-series 

%   dim     : dimension - m, how many of them are we taking for comparing
%   3,5,7
%   r       : tolerance (typically 0.2 * std)
%   data    : time-series data


epsilon = 0.001;
N = length(data);
correl = zeros(1,2);
dataMat = zeros(dim+1,N-dim);
for i = 1:dim+1
    dataMat(i,:) = data(i:N-dim+i-1);
end

for m = dim:dim+1
    count = zeros(1,N-dim);
    tempMat = dataMat(1:m,:);
    
    for i = 1:N-m
        % calculate distance, excluding self-matching case
        dist = max(abs(tempMat(:,i+1:N-dim) - repmat(tempMat(:,i),1,N-dim-i)));
        
        D = (dist < r);
        
        count(i) = sum(D)/(N-dim-1);
    end
    
    correl(m-dim+1) = sum(count)/(N-dim);
end

saen = log((correl(1)+epsilon)/(correl(2)+epsilon));

end
