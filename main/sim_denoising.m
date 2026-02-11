% =========================================================================
% Introduction
% =========================================================================
% This code performs video denoising via spatiotemporal total variation
% regularization.
%
% Author: Yunhui Gao (gyh21@mails.tsinghua.edu.cn)
% =========================================================================
%%
% =========================================================================
% Load simulation dataset
% =========================================================================
clear all;
close all;
clc;

foldername = 're2500_weInf_surfElev_first2089_B1024_rayTrace_D3pi_png';

% get all png files in folder
files = dir(fullfile(foldername, '*.png'));
assert(~isempty(files), 'No PNG files found.');

% sort by filename (important for time order)
[~, idx] = sort({files.name});
files = files(idx);

K = numel(files);

% read first image to get size
img = im2double(imread(fullfile(foldername, files(1).name)));
[n1,n2] = size(img);

x = zeros(n1,n2,K);
x(:,:,1) = img;

% read remaining frames
for k = 2:K
    img = im2double(imread(fullfile(foldername, files(k).name)));
    x(:,:,k) = img;
end

% % add additive white gaussian noise
% rng(0)
% snr_val = 10;
% y = awgn(x,snr_val);

y = x;

% figure
% for k = 1:K
%     ax=subplot(1,2,1);imshow(x(:,:,k),[0,1]);
%     title(ax,'Ground-truth video')
%     ax=subplot(1,2,2);imshow(y(:,:,k),[0,1]);
%     title(ax,'Noisy video')
%     drawnow;
% end

snr_val = 10;

cache_path_name = fullfile('cache', foldername, num2str(snr_val));

if ~isfolder(cache_path_name)
    mkdir(cache_path_name)
end

save([cache_path_name,'/measurement.mat'],'y','x')

[n1,n2,n3] = size(y);

% set regularization parameters
lam_s = 0.8e-1;     % spatial regularization coefficient
lam_t = 3e-1;       % temporal regularization coefficient

n_iters = 1;      % number of iterations

% define auxilary variables
w_est = zeros(n1,n2,n3,3);
v_est = zeros(n1,n2,n3,3);

y = single(y); v_est = single(v_est); w_est = single(w_est);

% --- choose GPU if available, otherwise fall back to CPU
gpu = true;

if gpu
    try
        device = gpuDevice();   % will error if no supported GPU
        reset(device);
        y     = gpuArray(y);
        v_est = gpuArray(v_est);
        w_est = gpuArray(w_est);
        fprintf("Using GPU: %s\n", device.Name);
    catch ME
        warning("GPU disabled (%s). Falling back to CPU.");
        gpu = false;
    end
end

% initialize GPU
if gpu
    y     = gather(y);
    w_est = gather(w_est);
    reset(device);
end

% main loop
for iter = 1:n_iters
    fprintf('iter: %04d / %04d\n', iter, n_iters)
    w_next = v_est + 1/12*Df(y-DTf(v_est));
    w_next(:,:,:,1) = min(abs(w_next(:,:,:,1)),lam_s).*exp(1i*angle(w_next(:,:,:,1)));
    w_next(:,:,:,2) = min(abs(w_next(:,:,:,2)),lam_s).*exp(1i*angle(w_next(:,:,:,2)));
    w_next(:,:,:,3) = min(abs(w_next(:,:,:,3)),lam_t).*exp(1i*angle(w_next(:,:,:,3)));

    v_est = w_next + iter/(iter+3)*(w_next-w_est);
    w_est = w_next;
end

% gather data from GPU
if gpu
    y       = gather(y);
    w_est   = gather(w_est);
    reset(device);
end

% calculate primal optimum from dual optimum
x_est = real(y - DTf(w_est));

%%
% =========================================================================
% Visualize results (export as .gif files)
% =========================================================================
fps = 10;
fig = figure;
for k = 1:size(x_est,3)
    u = x_est(:,:,k);
    imshow(imresize(u,size(u)*3,'nearest'),[0,1],'border','tight');drawnow;
    F = getframe(fig);
    I = frame2im(F);
    [I,map] = rgb2ind(I,256);
    if k == 1
        imwrite(I,map,[cache_path_name,'/3dtv.gif'],'gif','Loopcount',inf,'DelayTime',1/fps);
    else
        imwrite(I,map,[cache_path_name,'/3dtv.gif'],'gif','WriteMode','append','DelayTime',1/fps);
    end
end

save([cache_path_name,'/3dtv.mat'],'x_est');

%%
% =========================================================================
% Auxiliary functions
% =========================================================================

function w = Df(x)
% =========================================================================
% Calculate the 3D gradient (finite difference) of an input 3D datacube.
% -------------------------------------------------------------------------
% Input:    - x  : The input 3D datacube.
% Output:   - w  : The gradient (4D array).
% =========================================================================
if size(x,3) > 1
    w = cat(4, x(1:end,:,:) - x([2:end,end],:,:), ...
               x(:,1:end,:) - x(:,[2:end,end],:), ...
               x(:,:,1:end) - x(:,:,[2:end,end]));
else
    w = cat(4, x(1:end,:,:) - x([2:end,end],:,:), ...
               x(:,1:end,:) - x(:,[2:end,end],:), ...
               zeros(size(x(:,:,1))));
end
end


function u = DTf(w)
% =========================================================================
% Calculate the transpose of the gradient operator.
% -------------------------------------------------------------------------
% Input:    - w  : 4D array.
% Output:   - x  : 3D array.
% =========================================================================
u1 = w(:,:,:,1) - w([end,1:end-1],:,:,1);
u1(1,:,:) = w(1,:,:,1);
u1(end,:,:) = -w(end-1,:,:,1);

u2 = w(:,:,:,2) - w(:,[end,1:end-1],:,2);
u2(:,1,:) = w(:,1,:,2);
u2(:,end,:) = -w(:,end-1,:,2);

if size(w,3) > 1
    u3 = w(:,:,:,3) - w(:,:,[end,1:end-1],3);
    u3(:,:,1) = w(:,:,1,3);
    u3(:,:,end) = -w(:,:,end-1,3);
else
    u3 = 0;
end

u = u1 + u2 + u3;
end

