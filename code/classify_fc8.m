

function [pre_label,err_rate] = classify_fc8(feat_fc8,label)

for i = 1:length(feat_fc8)
    pre_label(i,1) = find(feat_fc8(i,:)==max(feat_fc8(i,:)))-1;
end
a = find(pre_label==label);
err_rate=length(a)/length(label)*100
end