function [TPR, TNR, gmean, err] = classification_performance(Labels_GT, y_predicted)

TPR = sum(y_predicted == 1 & Labels_GT == 1)/sum(Labels_GT); %sensitivity
TNR = sum(y_predicted == 0 & Labels_GT == 0)/numel(find(Labels_GT==0)); %specificity

gmean = sqrt(TPR*TNR);
err = sum(y_predicted ~= Labels_GT)/length(y_predicted);    

end

