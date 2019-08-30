function H = perm_entropy(x,n,tau)
% The perm_entropy function calculates the permutation entropy of time-series.
%It computes the permutation entropy H of 
% a scalar vector x, using permutation order n and time lag TAU.
% x      - data vector (Mx1 or 1xM)
% n      - permutation order - how many of them are we taking for
% comparison  ...3,5,7
% tau    - time lag scalar = 1

x = x(:);
M = length(x);

if n*log10(n)>15, error('permutation dimension too high'); end

    shift_mat_ind = reshape(0:tau:(n-1)*tau,[],1) * ones(1,M-(n-1)*tau) +...
        ones(n, 1) * reshape(1:(M-(n-1)*tau),1,[]);
    shift_mat = x(shift_mat_ind);
    

    % sort matrix along rows to build rank orders, equal values retain
    % order of appearance
    [~, sort_ind_mat] = sort(shift_mat,1);
    ind_mat = sort_ind_mat - 1;
    
% assign unique number to each pattern (base-n number system)
ind_vec = n.^(0:n-1) * ind_mat;
br = numel(find(ind_vec == max(ind_vec)));

switch br
    case 1
        ad = 1;
    otherwise
        ad = br;
end

[~,ia,~] = unique(sort(ind_vec), 'first');

permpat_num = diff([ia;ia(end)+ad]);
permpat_num = permpat_num/sum(permpat_num);

% compute permutation entropy
H = -sum(permpat_num .* log2(permpat_num));
%normalization:

%H = H/(n-1);
%H = H/(log2(factorial(n)));
end



