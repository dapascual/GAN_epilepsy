clear

fs = 256;
leave_out_result = [];

patient = {'pat_1' 'pat_2' 'pat_3' 'pat_4' 'pat_5' 'pat_6' 'pat_7' 'pat_8' 'pat_9' 'pat_10' 'pat_11' 'pat_12' 'pat_13' 'pat_14' 'pat_15' 'pat_16' 'pat_17' 'pat_18' 'pat_19' 'pat_20' 'pat_21' 'pat_22' 'pat_23' 'pat_24' 'pat_25' 'pat_26' 'pat_27' 'pat_28' 'pat_29' 'pat_30'};
patient_test = {'pat_1' 'pat_2' 'pat_3' 'pat_4' 'pat_5' 'pat_6' 'pat_7' 'pat_8' 'pat_9' 'pat_10' 'pat_11' 'pat_12' 'pat_13' 'pat_14' 'pat_15' 'pat_16' 'pat_17' 'pat_18' 'pat_19' 'pat_20' 'pat_21' 'pat_22' 'pat_23' 'pat_24' 'pat_25' 'pat_26' 'pat_27' 'pat_28' 'pat_29' 'pat_30'};

%%
dir_train = 'data/RF_trainset_seiz/';
dir_test = 'data/RF_test/';

for pt = patient_test
    pat_pt = pt{1}
    Features_training_all = [];
    Labels_training_all = [];
    % Get seizure features to train RF
    for p = patient  
        if strcmp(pat_pt,p{1}) == 0
            pat = [p{1} '_'];
            load([dir_train pat 'trainRF_seiz.mat']);
            X_features_non_seiz = [];
            X_features_seiz = [];
            for i = 1:size(X_seiz,1) 
                X_features_seiz = [X_features_seiz; [get_features(fs, X_seiz(i,1:1024)) get_features(fs, X_seiz(i,1025:2048))]];
            end
               
            Features_training_all = [Features_training_all; X_features_seiz];%; X_features_non_seiz_k]]; 
        end
    end

    % Get non-seizure features to train RF
    load(['data/RF_trainset_nonseiz/' pat_pt '_trainRF_nonseiz.mat']);
    X_features_non_seiz = [];
    for i = 1:size(X_non_seiz,1) 
        X_features_non_seiz = [X_features_non_seiz; [get_features(fs, X_non_seiz(i,1:1024)) get_features(fs, X_non_seiz(i,1025:2048))]];
    end
    
    % Remove noisy non-seizure samples from the training set
    median_powE1 = median(X_features_non_seiz(:,38),1);
    mad_powE1 = mad(X_features_non_seiz(:,38),1);
    median_powE2 = median(X_features_non_seiz(:,92),1);
    mad_powE2 = mad(X_features_non_seiz(:,92),1);

    X_features_non_seiz_all = [];
    for i = 1:size(X_features_non_seiz,1)
        if X_features_non_seiz(i,38) < median_powE1 + 3*mad_powE1 && X_features_non_seiz(i,38) > median_powE1 - 3*mad_powE1 && X_features_non_seiz(i,92) < median_powE2 + 3*mad_powE2 && X_features_non_seiz(i,92) > median_powE2 - 3*mad_powE2
            X_features_non_seiz_all = [X_features_non_seiz_all; X_features_non_seiz(i,:)];
        end
    end
    

    %% Load test set and extract features
    load([dir_test pat_pt '_testRF.mat']);
    X_features_non_seiz = [];
    X_features_seiz = [];
    
    for i = 1:size(X_non_seiz,1)
        X_features_non_seiz = [X_features_non_seiz; [get_features(fs, X_non_seiz(i,1:1024)) get_features(fs, X_non_seiz(i,1025:2048))]];
    end
    for i = 1:size(X_seiz,1) 
        X_features_seiz = [X_features_seiz; [get_features(fs, X_seiz(i,1:1024)) get_features(fs, X_seiz(i,1025:2048))]];
    end

    Features_testing = [X_features_seiz; X_features_non_seiz];
    Labels_testing = [ones(size(X_features_seiz,1),1); zeros(size(X_features_non_seiz,1),1)];
    ix = randperm(size(Labels_testing,1));
    Features_testing = Features_testing(ix,:);
    Labels_testing = Labels_testing(ix,:);

   
   
    %% Train and test RF classifier shuffling each time the training set
    gmean_pat = [];
    sens_pat = [];
    spec_pat = [];
    for k = 1:15    
        
        j_perm_non = randperm(size(X_features_non_seiz_all,1));
        nonseiz_size = min(2000,length(j_perm_non));
        X_features_non_seiz_k = X_features_non_seiz_all(j_perm_non(1:nonseiz_size),:);%j_perm_non(1:500),:);
    
        ix = randperm(size(Features_training_all,1));
        F = Features_training_all(ix,:);
        Features_training = [F(1:size(X_features_non_seiz_k,1),:); X_features_non_seiz_k];%; X_features_non_seiz_k]];
        Labels_training = [ones(size(X_features_non_seiz_k,1),1); zeros(size(X_features_non_seiz_k,1),1)];

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
    leave_out_result = [leave_out_result; {pt(1) geomean(sens_pat) geomean(spec_pat) geomean(gmean_pat)}];
    geomean(gmean_pat)
    
    save('Results_baseline_GAN.mat', 'leave_out_result');
    
end
