% ----------------------------------------
% 1. filter out regions
% ----------------------------------------

box_filter = cell(1,length(bbs));
for i = 1:length(bbs)
    tbox=[];
    tbox=bbs{1,i}(:,1:4);
    if length(tbox) > 100
        for j = 1:length(tbox)
            b_w= tbox(j,3)-tbox(j,1);         %width
            b_h = tbox(j,4)-tbox(j,2);        %height
            if ss1/b_h >6 || b_h/b_w>6 || ss1>500 ||b_h>500 || ss1<28 || b_h<28
                box_filter{1,i}(j,1) = 0;
            else
                box_filter{1,i}(j,1) = 1;
            end
        end
    end
end

% ----------------------------------------
% 2. candidate selection
% ----------------------------------------

m = 50; % candidate num
for i = 1:length(bbs)
    tbox = [];
    tbox = bbs{1,i}(:,:);
    tbox_filter = tbox(logical(box_filter{i}),:);
    % calculate W in (1)
    W{i} = boxoverlap(tbox_filter, tbox_filter);
    % cluster
    [idx,~,~,~] = clu_ncut(W{i},m);
    [~,IA]=unique(idx);
    candidate_box{i} = tbox_filter(IA,:);
end

% ----------------------------------------
% 3. combine with ARs
% ----------------------------------------
model_def = '';
model_file = '';
imgname = '';
imsize = '';
path = '';
k = 20;
alpha = 0.6;
beta = 0.3;

[~,box_fc8] = extract_myfeature(model_def,model_file,imgname,imsize,path,candidate_box);
for i = 1:length(imgname)
    ss = (i-1)*m+1;
    ee = i*m;
    box_pre = box_fc8(ss:ee,:);
    % calculate senti_score
    senti_score = 1 + box_pre(:,1).*log(box_pre(:,1))+box_pre(:,2).*log(box_pre(:,2));
    % calculate object_score
    obj_score = candidate_box{i}(:,5);
    senti_score_norm = mapminmax(senti_score',0,1)';
    obj_score_norm = mapminmax(obj_score',0,1)';
    % calculate AR_score
    AR_score = (1-alpha)*senti_score_norm + alpha*obj_score_norm;
    [c,d] = sort(AR_score,'descend');
    global_f = ft_ft8(i,:);
    if k==1
        local_f = (box_pre(d(1:k),:));
    else
        local_f = mean(box_pre(d(1:k),:));
    end
    % sum pooling
    final_f1(i,:) = (1-beta)*global_f + beta*local_f; 
    % concanation
    final_f2(i,:) = [global_f local_f];
    % max pooling
    final_f3(i,1) = max(global_f(1),local_f(1)); 
    final_f3(i,2) = max(global_f(2),local_f(2));
end
[~,acc1] = classify(~test_ind,test_ind,label2,final_f1);
[~,acc2] = classify(~test_ind,test_ind,label2,final_f2);
[~,acc3] = classify(~test_ind,test_ind,label2,final_f3);
result = [acc1 acc2 acc3]
