function [shannon_en, renyi_en, tsallis_en]  = sh_ren_ts_entropy(x, a, q)
%% shannon_en - Shannon entropy (it's a special case of Renyi's entropy for a = 1)
%% renyi_en - Renyi entropy 

p = hist(x);

p = p./sum(p);

p = p(p>0); %to exclude log(0)

shannon_en = - sum(p.*log2(p));

renyi_en = log2(sum(p.^a))/(1-a);

tsallis_en = (1-sum(p.^q))/(q-1);


end

