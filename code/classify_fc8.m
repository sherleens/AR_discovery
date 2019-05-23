function [pre_label,acc] = classify_fc8(feat_fc8,label)
% this function calculate the accuracy according to fc8 output for 2-class
% inputs:
%       feat_fc8 -- an N*C matrix, N/C is the number of data/class
%       label -- the ground truth of these N samples, e.g. 0 for negative,
%       1 for positive
% outputs:
%       pre_label -- the predicted class for N samples
%       acc -- accuracy (%)

for i = 1:length(feat_fc8)
    pre_label(i,1) = find(feat_fc8(i,:)==max(feat_fc8(i,:)))-1; %range label from 1-2 to 0-1
end
a = find(pre_label==label);
acc = length(a)/length(label)*100
end