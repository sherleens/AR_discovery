

function [pre_label,err_rate] = classify(train_ind,test_ind,label,features)
addpath(genpath('/home/ubuntu/tools/liblinear-1.5-dense-float'));

train_label = label(train_ind);
test_label = label(test_ind);

train_features = features(train_ind,:);
test_features = features(test_ind,:);


disp('Train linear SVM ... ...');
lc = 1; % regularization parameter C
option = ['-s 1 -c ' num2str(lc)];
model = train(train_label, single(train_features), option);

[pre_label,err_rate,~] = predict(test_label, single(test_features), model);
end