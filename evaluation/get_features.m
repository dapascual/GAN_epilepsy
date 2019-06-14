function X = get_features(fs, signal)

level = 7;
wavelet_type = 'db4';

[C, L] = wavedec(signal, level, wavelet_type);

n1 = 2;

r1 = 0.2;
r2 = 0.35;

d7 = C(L(1,1) + 1 : L(1,1) + L(1,2));
d6 = C(L(1,1) + L(1,2) + 1 : L(1,1) + L(1,2) + L(1,3));
d5 = C(L(1,1) + L(1,2) + L(1,3) + 1 : L(1,1) + L(1,2) + L(1,3) + L(1,4));
d4 = C(L(1,1) + L(1,2) + L(1,3)+ L(1,4) + 1 : L(1,1) + L(1,2) + L(1,3) + L(1,4) + L(1,5));
d3 = C(L(1,1) + L(1,2) + L(1,3) + L (1,4) + L(1,5) + 1 : L(1,1) + L(1,2) + L(1,3) + L(1,4) + L(1,5) + L(1,6));


samp_1_d7_1 = samp_entropy(n1, r1 * std(d7), d7);
samp_1_d6_1 = samp_entropy(n1, r1 * std(d6), d6);

samp_2_d7_1 = samp_entropy(n1, r2 * std(d7), d7);
samp_2_d6_1 = samp_entropy(n1, r2 * std(d6), d6);

perm_d7_3 = perm_entropy(d7, 3, 1);
perm_d7_5 = perm_entropy(d7, 5, 1);
perm_d7_7 = perm_entropy(d7, 7, 1);

perm_d6_3 = perm_entropy(d6, 3, 1);
perm_d6_5 = perm_entropy(d6, 5, 1);
perm_d6_7 = perm_entropy(d6, 7, 1);

perm_d5_3 = perm_entropy(d5, 3, 1);
perm_d5_5 = perm_entropy(d5, 5, 1);
perm_d5_7 = perm_entropy(d5, 7, 1);

perm_d4_3 = perm_entropy(d4, 3, 1);
perm_d4_5 = perm_entropy(d4, 5, 1);
perm_d4_7 = perm_entropy(d4, 7, 1);

perm_d3_3 = perm_entropy(d3, 3, 1);
perm_d3_5 = perm_entropy(d3, 5, 1);
perm_d3_7 = perm_entropy(d3, 7, 1);

a = 2;  q = 2;

[shannon_en_sig, renyi_en_sig, tsallis_en_sig]  = sh_ren_ts_entropy(signal, a, q);
[shannon_en_d7, renyi_en_d7, tsallis_en_d7]  = sh_ren_ts_entropy(d7, a, q);
[shannon_en_d6, renyi_en_d6, tsallis_en_d6]  = sh_ren_ts_entropy(d6, a, q);
[shannon_en_d5, renyi_en_d5, tsallis_en_d5]  = sh_ren_ts_entropy(d5, a, q);
[shannon_en_d4, renyi_en_d4, tsallis_en_d4]  = sh_ren_ts_entropy(d4, a, q);
[shannon_en_d3, renyi_en_d3, tsallis_en_d3]  = sh_ren_ts_entropy(d3, a, q);


p_tot = bandpower(signal);
p_dc = bandpower(signal, fs, [0 0.1]);
p_mov = bandpower(signal, fs, [0.1 0.5]);
p_delta = bandpower(signal, fs, [0.5 4]);
p_theta = bandpower(signal, fs, [4 8]);
p_alfa = bandpower(signal, fs, [8 12]);
p_middle = bandpower(signal, fs, [12 13]);
p_beta = bandpower(signal, fs, [13 30]);
p_gamma = bandpower(signal, fs, [30 45]);

p_dc_rel = p_dc/p_tot;
p_mov_rel = p_mov/p_tot;
p_delta_rel = p_delta/p_tot;
p_theta_rel = p_theta/p_tot;
p_alfa_rel = p_alfa/p_tot;
p_middle_rel = p_middle/p_tot;
p_beta_rel = p_beta/p_tot;
p_gamma_real = p_gamma/p_tot;


X = [samp_1_d7_1 samp_1_d6_1 samp_2_d7_1 samp_2_d6_1 perm_d7_3 perm_d7_5 perm_d7_7 perm_d6_3 perm_d6_5 perm_d6_7 perm_d5_3 perm_d5_5 ...
    perm_d5_7 perm_d4_3 perm_d4_5 perm_d4_7 perm_d3_3 perm_d3_5 perm_d3_7 shannon_en_sig renyi_en_sig tsallis_en_sig shannon_en_d7 renyi_en_d7 tsallis_en_d7 ...
    shannon_en_d6 renyi_en_d6 tsallis_en_d6 shannon_en_d5 renyi_en_d5 tsallis_en_d5 shannon_en_d4 renyi_en_d4 tsallis_en_d4 shannon_en_d3 renyi_en_d3 tsallis_en_d3 ...
    p_tot p_dc p_mov p_delta p_theta p_alfa p_middle p_beta p_gamma p_dc_rel p_mov_rel p_delta_rel p_theta_rel p_alfa_rel p_middle_rel p_beta_rel p_gamma_real];

% X = [p_tot p_dc p_mov p_delta p_theta p_alfa p_middle p_beta];

end

