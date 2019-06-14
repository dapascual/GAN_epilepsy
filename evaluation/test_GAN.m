%Script to test GANs output: compare to baseline
clear

fs = 256;
power_features = [38:54,92:108];

patient = {'pat_1' 'pat_2' 'pat_3' 'pat_4' 'pat_5' 'pat_6' 'pat_7' 'pat_8' 'pat_9' 'pat_10' 'pat_11' 'pat_12' 'pat_13' 'pat_14' 'pat_15' 'pat_16' 'pat_17' 'pat_18' 'pat_19' 'pat_20' 'pat_21' 'pat_22' 'pat_23' 'pat_24' 'pat_25' 'pat_26' 'pat_27' 'pat_28' 'pat_29' 'pat_30'};

%%
gmean_patient = [];
sens_patient = [];
spec_patient = [];
patient_result = [];
gmean_by_patient = [];  
sens_by_patient = [];
spec_by_patient = [];

for p = patient
   
    pat = p{1}
    Features_training = [];
    Labels_traing = [];   
   
    %% Load test set and extract features
    load(['data/RF_testset/' pat '_testRF.mat']);
    X_features_non_seiz = [];
    X_features_seiz = [];
    for i = 1:size(X_seiz,1)
        X_features_seiz = [X_features_seiz; [get_features(fs, X_seiz(i,1:1024)) get_features(fs, X_seiz(i,1025:2048))]];
    end
    for i = 1:size(X_non_seiz,1)
        X_features_non_seiz = [X_features_non_seiz; [get_features(fs, X_non_seiz(i,1:1024)) get_features(fs, X_non_seiz(i,1025:2048))]];
    end

    Features_testing = [X_features_seiz; X_features_non_seiz];
    Labels_testing = [ones(size(X_features_seiz,1),1); zeros(size(X_features_non_seiz,1),1)];
    ix = randperm(size(Labels_testing,1));
    Features_testing = Features_testing(ix,:);
    Labels_testing = Labels_testing(ix,:);
    
    %% Load train set (synthetic seizures and real non seizure)
    load(['data/RF_trainset_nonseiz/' pat '_trainRF_nonseiz.mat']); # Real non seizures
    dir_seiz = ['../test_set_transformed/' pat '/']; # Synthetic seizures
   
    X_features_non_seiz = [];
    X_features_seiz = [];


    for i = 1:size(X_non_seiz,1)
        i_1 = sprintf('%d', i);

        load([dir_seiz 'GAN_seizure_' pat '_GAN_test_' i_1 '.mat']);
              
        X_features_seiz = [X_features_seiz; [get_features(fs, GAN_seiz(1,1:1024)) get_features(fs, GAN_seiz(1,1025:2048))]];
        X_features_non_seiz = [X_features_non_seiz; [get_features(fs, X_non_seiz(i,1:1024)) get_features(fs, X_non_seiz(i,1025:2048))]];              
    end
   
    X_features_non_seiz_den = [];
    
    % Remove noisey non seizures
    median_powE1 = median(X_features_non_seiz(:,38),1);
    mad_powE1 = mad(X_features_non_seiz(:,38),1);
    median_powE2 = median(X_features_non_seiz(:,92),1);
    mad_powE2 = mad(X_features_non_seiz(:,92),1);

    for i = 1:size(X_features_non_seiz,1)
        if X_features_non_seiz(i,38) < median_powE1 + 3*mad_powE1 && X_features_non_seiz(i,38) > median_powE1 - 3*mad_powE1 && X_features_non_seiz(i,92) < median_powE2 + 3*mad_powE2 && X_features_non_seiz(i,92) > median_powE2 - 3*mad_powE2
            X_features_non_seiz_den = [X_features_non_seiz_den; X_features_non_seiz(i,:)];
        end
    end
  
    %% Train and test RF classifier
    gmean_pat = [];
    sens_pat = [];
    spec_pat = [];
    for k = 1:15         
               
        j_perm_non = randperm(size(X_features_non_seiz_den,1));
        nonseiz_size = min(2000,length(j_perm_non));
        X_features_non_seiz_k = X_features_non_seiz_den(j_perm_non(1:nonseiz_size),:);%j_perm_non(1:500),:);

        j_perm = randperm(size(X_features_seiz,1));
        seiz_size = min(2000,nonseiz_size);
        X_features_seiz_k = X_features_seiz(j_perm(1:seiz_size),:);%size(X_features_non_seiz_k,1)/2),:);


        Features_training = [X_features_seiz_k; X_features_non_seiz_k];
        Labels_training = [ones(size(X_features_seiz_k,1),1); zeros(size(X_features_non_seiz_k,1),1)];
        ix = randperm(size(Labels_training,1));
        Features_training = Features_training(ix,:);
        Labels_training = Labels_training(ix,:);
    
        mdl = TreeBagger(500, Features_training, Labels_training, 'Method', 'classification');

        y_predicted = str2double(predict(mdl, Features_testing));

        [sens, spec, gmean, ~] = classification_performance(Labels_testing, y_predicted);
        gmean_pat = [gmean_pat; gmean];
        sens_pat = [sens_pat; sens];
        spec_pat = [spec_pat; spec];
       
    end
    gmean_patient = geomean(gmean_pat)
    gmean_by_patient = [gmean_by_patient; gmean_patient];
    sens_patient = geomean(sens_pat);
    sens_by_patient = [sens_by_patient; sens_patient];
    spec_patient = geomean(spec_pat);
    spec_by_patient = [spec_by_patient; spec_patient];
    patient_result = [patient_result; {p(1) sens_patient spec_patient gmean_patient seiz_size nonseiz_size}];
    save('Results_test_GAN', 'patient_result');
   
end

gmean_total = geomean(gmean_by_patient);
sens_total = geomean(sens_by_patient);
spec_total = geomean(spec_by_patient);

save('Results_test_GAN', 'patient_result', 'sens_total', 'spec_total', 'gmean_total');
