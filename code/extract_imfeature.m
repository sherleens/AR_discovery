%-----------------------------------------------
%               extract_imfeature.m
%-----------------------------------------------
% input:
% -model_def: caffe model deploy
% -model_file: caffemodel
% -imgname: image list [N x 1]
% -imsize: crop size, e.g. 224, 227
% -path: dataset path
%
% output:
% -feat_fc7: features from FC7 layer [N x 4096]
% -feat_fc8: features from FC8 layer [N x 8]
%-----------------------------------------------
function [feat_fc7,feat_fc8]=extract_imfeature(model_def,model_file,imgname,imsize,path)

addpath('~/sdy/caffe-master/matlab/');
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
% get batch
batch_size =100;
crop_size = imsize;
for i = 1: length(imgname)
    imgpath{1,i}  = imgname{i};
end
N = numel(imgpath)*10;
num_batches = ceil(N / batch_size);
batches = cell(num_batches, 1);
batch_padding = batch_size - mod(N, batch_size);
if mod(N, batch_size) == 0
    batch_padding = 0;
end

for batch = 1 : num_batches
    fprintf('Preparing batches %d / %d \n',batch, num_batches);
    batch_start = (batch-1)*batch_size/10+1;
    batch_end = min(N/10, batch_start+batch_size/10-1);
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
            ims(:,:,:,(j-batch_start)*10+1:(j-batch_start+1)*10) = prepare_image(im,imsize);
        catch
            disp('imread fail');
        end
    end
    batches{batch} = ims;
end

% compute features for each batch of region images
feat_dim = -1;
feat_fc7 = [];
feat_dim_g = -1;
feat_fc8 = [];

for j = 1:length(batches)
    % forward propagate batch of region images
    fprintf('extract feature batch %d / %d \n', j, length(batches));
    net.forward(batches(j));
    %fc7
    global_f = net.blobs('fc7').get_data(); 
    global_f = global_f(:);
    %fc8
    f = net.blobs('my_fc8').get_data(); 
    f = f(:);
    if j == 1 % first batch, init feat_dim and feat
        feat_dim = length(f)/batch_size;
        feat_dim_g = length( global_f)/batch_size;
        feat_fc8 = zeros(size(imgpath, 2), feat_dim, 'single');
        feat_fc7 = zeros(size(imgpath, 2), feat_dim_g, 'single');
    end
    f = reshape(f, [feat_dim batch_size]);
    global_f = reshape(global_f, [feat_dim_g batch_size]);
    if j == length(batches)
        if batch_padding > 0
            f = f(:, 1:end-batch_padding);
            global_f = global_f(:, 1:end-batch_padding);
        end
    end
    tmp1=[];tmp2=[];
    for kk = 1: 10 : size(f,2)
        tmp1 = [tmp2;mean(global_f(:,kk:kk+9),2)'];
        tmp2 = [tmp1;mean(f(:,kk:kk+9),2)'];
    end
    feat_fc7 = [feat_fc7; tmp1];
    feat_fc8 = [feat_fc8; tmp2];
end
caffe.reset_all();
end

% ------------------------------------------------------------------------
function crops_data = prepare_image(im,imsize)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
d = load('~/caffe-master/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;
IMAGE_DIM = 256;
CROPPED_DIM = imsize;

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

% oversample (4 corners, center, and their x-axis flips)
crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
n = 1;
for i = indices
    for j = indices
        crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
        crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
        n = n + 1;
    end
end
center = floor(indices(2) / 2) + 1;
crops_data(:,:,:,5) = ...
    im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
end
