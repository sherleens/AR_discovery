%-----------------------------------------------
%               extract_boxfeature.m
%-----------------------------------------------
% input:
% -model_def: caffe model deploy
% -model_file: caffemodel
% -imgname: image list [N x 1]
% -imsize: crop size, e.g. 224, 227
% -path: dataset path
% -bbox: candidate box
%
% output:
% -feat_fc7: features from FC7 layer [Nm x 4096]
% -feat_fc8: features from FC8 layer [Nm x 8]
%-----------------------------------------------
function [feat_fc7,feat_fc8]=extract_boxfeature(model_def,model_file,imgname,imsize,path,mbox)
Caffe_path = '/home/ubuntu2/yxx2/caffe-master/matlab';
addpath(Caffe_path);
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(model_file, 'file')
    error('Please download CaffeNet from Model Zoo before you run this demo');
end
if ~exist(model_def, 'file')
    error('Please download deploy before you run this demo');
end
% Initialize a network
net = caffe.Net(model_def, model_file, phase);
batch_size = 20; %
crop_size = imsize;
crop_padding = 16;
crop_mode = 'Square';
d = load([Caffe_path,'/+caffe/imagenet/ilsvrc_2012_mean.mat']);
image_mean = d.mean_data;

for i = 1: length(imgname)
    imgpath{1,i}  = imgname{i};
end
m  = 20; %number of candidate box for each image
N = numel(imgpath) * m;
num_batches = ceil(N / batch_size);
batches = cell(num_batches, 1);
batch_padding = batch_size - mod(N, batch_size);
if mod(N, batch_size) == 0
    batch_padding = 0;
end

% get batch
for batch = 1 : num_batches
    fprintf('Preparing batches %d / %d \n',batch, num_batches);
    batch_start = (batch-1)*batch_size/m+1;
    batch_end = min(numel(imgpath), batch_start+batch_size/m-1);
    ims = zeros(crop_size, crop_size, 3, batch_size, 'single');
    for j = batch_start:batch_end
        try
            im = imread([path,imgpath{j}]);
            
            if size(im,3) ~= 3
                img = zeros(size(im,1), size(im,2), 3);
                img(:, :, 1) = im;
                img(:, :, 2) = im;
                img(:, :, 3) = im;
                im = img;
            end
            im = single(im(:,:,[3 2 1])); % permute channels from RGB to BGR
            try
                bbox = mbox{j};
            catch
                disp('Box read fail!');
                bbox = [1 1 size(im, 1) size(im, 2)];
            end
            for k = 1:m%length(bbox)
                crop = rcnn_im_crop(im, bbox(k,:), crop_mode, crop_size, crop_padding, image_mean);
                % flip width and height to make width the fastest dimension (for caffe)
                ims(:,:,:,(j-batch_start)*m+k) = permute(crop, [2 1 3]);
            end
        catch
           disp('Imread fail!');
        end
    end
    batches{batch} = ims;
    %save(['/home/ubuntu2/zyj/AR_discovery/code/batch/batch_',num2str(batch)],'ims');
end

% compute features for each batch of region images
feat_dim_fc7 = -1;
feat_fc7 = [];
feat_dim_fc8 = -1;
feat_fc8 = [];

for j = 1:length(batches)
    % forward propagate batch of region images
    fprintf('extract feature batch %d / %d \n', j, length(batches));
    net.forward(batches(j));
    %batchj=load(['/home/ubuntu2/zyj/AR_discovery/code/batch/batch_',num2str(j),'.mat']);
    %net.forward(batchj);
    %fc7
    global_f = net.blobs('fc7').get_data();
    global_f = global_f(:);
    %fc8 to prob
    f = net.blobs('prob').get_data();
    f = f(:);
    if j == 1 % first batch, init feat_dim and feat
        feat_dim_fc7 = length(global_f)/batch_size;
        feat_dim_fc8 = length(f)/batch_size;
        feat_fc7=[];
        feat_fc8=[];
    end
    f = reshape(f, [feat_dim_fc8 batch_size]);
    global_f = reshape(global_f, [feat_dim_fc7 batch_size]);
    if j == length(batches)
        if batch_padding > 0
            f = f(:, 1:end-batch_padding);
            global_f = global_f(:, 1:end-batch_padding);
        end
    end
    feat_fc7 = [feat_fc7; global_f'];
    feat_fc8 = [feat_fc8; f'];
end
caffe.reset_all();
end
