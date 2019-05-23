% -----------------------------------------------------
% 0. download or extract the objectness bounding box 
% -----------------------------------------------------
clear all;
load('TMM_config/fi_config.mat')
% run('/home/ubuntu2/sdy/objectness-release-v2.2 (1)/objectness-release-v2.2/demo.m')
load('Objectness_bbox/fi_c2_objectness_max2000.mat')
for i=1:length(boxes)
    if size(boxes{i})<100
        error('boxes num < 100')
    end
end

% -----------------------------------------------------
% 1. filter out regions
% -----------------------------------------------------
disp('Step 1: filter out regions');
try 
    load('tmp/fi_c2_candidate_box.mat');
catch  
    bbs = boxes;
    box_filter = cell(1,length(bbs));
    for i = 1:length(bbs)
        tbox=[];
        tbox=bbs{i,1}(:,1:4);
        if length(tbox) > 100
            for j = 1:length(tbox)
                b_w= tbox(j,3)-tbox(j,1);         %width
                b_h = tbox(j,4)-tbox(j,2);        %height
                if b_w/b_h >6 || b_h/b_w>6 || b_w>500 ||b_h>500 || b_w<28 || b_h<28
                    box_filter{1,i}(j,1) = 0;
                else
                    box_filter{1,i}(j,1) = 1;
                end
            end
        end
    end
end

% -----------------------------------------------------
% 2. download or run candidate selection alogrithm
% -----------------------------------------------------
disp('Step 2: candidate selection alogrithm');
m = 20; % candidate num
try 
    load('tmp/fi_c2_candidate_box.mat');
catch
    parfor i = 1:length(bbs)
        tbox = [];
        tbox = bbs{i,1}(:,:);
        tbox_filter = tbox(logical(box_filter{i}),:);
        W{i} = boxoverlap(tbox_filter, tbox_filter);    % calculate W in (1)
        [idx,~,~,~] = clu_ncut(W{i},m);    % cluster
        [~,IA]=unique(idx);
        candidate_box{i} = tbox_filter(IA,:);
    end
    save('tmp/fi_c2_candidate_box','candidate_box');
end

% -----------------------------------------------------
% 3. download or extract global and local features 
% -----------------------------------------------------
disp('Step 3: extract global and local features');
try
    load('/home/ubuntu2/zyj/AR_discovery/code/tmp/fi_c2_boxfc8.mat');
    load('/home/ubuntu2/zyj/AR_discovery/code/tmp/fi_c2_im_ft8.mat');
catch
    imsize = 224; 
    model_def = ' ';
    model_file = ' ';
    path = '';      %dataset path
    [~,box_fc8] = extract_boxfeature(model_def,model_file,imgname(test_ind),imsize,path,candidate_box);
    save('tmp/fi_c2_boxfc8','box_fc8');
    [~,im_ft8] = extract_imfeature(model_def,model_file,imgname(test_ind),imsize,path);
    save('tmp/fi_c2_im_ft8','im_ft8');
end

% -----------------------------------------------------
% 4. predict with ARs
% -----------------------------------------------------
disp('Step 4: prediction');
k = 8; alpha = 0.6; beta = 0.3;     %hyper-parameter
for i = 1:length(imgname(test_ind))
    box_pre = box_fc8((i-1)*m+1:i*m,:); 
    senti_score = 1 + box_pre(:,1).*log(box_pre(:,1))+box_pre(:,2).*log(box_pre(:,2)); % calculate senti_score
    obj_score = candidate_box{i}(:,5); % obtain object_score
    obj_score = obj_score(1:m);
    senti_score_norm = mapminmax(senti_score',0,1)';
    obj_score_norm = mapminmax(obj_score',0,1)';
    AR_score = (1-alpha)*senti_score_norm + alpha*obj_score_norm; % calculate AR_score
    [c,d] = sort(AR_score,'descend');
    global_f = im_ft8(i,:);
    if k==1
        local_f = (box_pre(d(1:k),:));
    else
        local_f = mean(box_pre(d(1:k),:));
    end
    final_f1(i,:) = (1-beta)*global_f + beta*local_f;  % sum pooling
%     final_f2(i,1) = max(global_f(1),local_f(1)); % max pooling
%     final_f2(i,2) = max(global_f(2),local_f(2));
%     final_f3(i,:) = [global_f local_f]; % concanation
end
[~,acc] = classify_fc8(final_f1,label(test_ind));
fid=fopen('result.txt','a+');
fprintf(fid,'%.4f\r\n',acc);   
fclose(fid);
