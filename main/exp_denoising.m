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
% Load and pre-process experimental data
% =========================================================================
clear all;
close all;
clc;

% load calibrated raw data
grp_num = 2;
load(['../data/experiment/G',num2str(grp_num),'/data.mat'],'vid')

% create path to save results
cache_path_name = ['cache/G',num2str(grp_num)];
if ~isfolder(cache_path_name)
    mkdir(cache_path_name)
end

% select the region of interest
if grp_num == 1
    c1 = 327;
    c2 = 259;
    w  = 250;
    y  = vid(c1-w+1:c1+w,c2-w+1:c2+w,:);
else
    y = vid;
end

% display the noisy video
figure
for k = 1:300
    imshow(y(:,:,k),[0,0.02]);
    title('Captured noisy video')
    drawnow;
end

% normalization
ymin = prctile(y(:), 0.5);
ymax = prctile(y(:),99.5);
y = (y - ymin)./(ymax - ymin);

% save pre-processed noisy video
save([cache_path_name,'/measurement.mat'],'y')

%%
% =========================================================================
% Run video denoising algorithm batch by batch
% =========================================================================

gpu = true;

K = 100;        % batch size (# of images)

frame_start = 1;
frame_step = 50;
frame_end = 300;

for frame = frame_start:frame_step:frame_end-K+1
    
    load([cache_path_name,'/measurement.mat'],'y')
    y = y(:,:,frame:frame+K-1);
    
    [n1,n2,n3] = size(y);
    
    % set regularization parameters
    lam_s = 5e-2;   % spatial regularization coefficient
    lam_t = 15e-2;  % temporal regularization coefficient

    n_iters = 200;  % number of iterations
    
    % define auxilary variables
    w_est = zeros(n1,n2,n3,3);
    v_est = zeros(n1,n2,n3,3);

    % initialize GPU
    if gpu
        device  = gpuDevice();
        reset(device)
        y       = gpuArray(y);
        v_est   = gpuArray(v_est);
        w_est   = gpuArray(w_est);
    end
    
    % main loop
    for iter = 1:n_iters
        fprintf('frame: %04d -> %04d | iter: %04d / %04d\n', frame, frame+K-1, iter, n_iters)
        w_next = v_est + 1/12*Df(y-DTf(v_est));
        w_next(:,:,:,1) = min(abs(w_next(:,:,:,1)),lam_s).*exp(1i*angle(w_next(:,:,:,1)));
        w_next(:,:,:,2) = min(abs(w_next(:,:,:,2)),lam_s).*exp(1i*angle(w_next(:,:,:,2)));
        w_next(:,:,:,3) = min(abs(w_next(:,:,:,3)),lam_t).*exp(1i*angle(w_next(:,:,:,3)));
    
        v_est = w_next + iter/(iter+3)*(w_next-w_est);  % Nesterov extrapolation
        w_est = w_next;
    end
    
    % gather data from GPU
    if gpu
        y       = gather(y);
        w_est   = gather(w_est);
        reset(device);
    end
    
    % calculate primal optimum from dual optimum
    vid_denoised = real(y - DTf(w_est));

    figure
    for k = 1:size(vid_denoised,3)
        imshow(vid_denoised(:,:,k),[0.3,0.55]);
        title('Denoised video')
        drawnow;
    end
    close
    
    % save results
    save([cache_path_name,'/',num2str(frame),'_',num2str(frame+99),'.mat'],'vid_denoised');
end

%%
% =========================================================================
% Post-process denoised videos
% =========================================================================

K = 100;    % batch size
frame_start = 1;
frame_step = 50;
frame_end = 300;

vid = zeros(size(y));

% merge batches into an entire video clip
flag = true;
for frame = frame_start:frame_step:frame_end-K+1
    fprintf('frame: %04d -> %04d \n', frame, frame+K-1)
    load([cache_path_name,'/',num2str(frame),'_',num2str(frame+99),'.mat'],'vid_denoised');
    if flag
        vid(:,:,frame-frame_start+1:frame-frame_start+K) = vid_denoised;
        flag = false;
    else
        a = zeros(1,1,K/2);
        a(:,:,1:end) = linspace(0,1,K/2);
        vid(:,:,frame-frame_start+1:frame-frame_start+K/2) = (1-a).*vid(:,:,frame-frame_start+1:frame-frame_start+K/2) + a.*vid_denoised(:,:,1:K/2);
        vid(:,:,frame-frame_start+K/2+1:frame-frame_start+K) = vid_denoised(:,:,K/2+1:K);
    end
end
        
save([cache_path_name,'/vid_denoised.mat'],'vid')

%%
% =========================================================================
% Visualize results (export as .gif files)
% =========================================================================
vmin = prctile(vid(:), 0.5);
vmax = prctile(vid(:),99.5);

fps = 10;
fig = figure;
set(gcf,'unit','normalized','position',[0.3,0.3,0.5,0.4])
for i = 1:size(vid,3)
    img = real(vid(:,:,i));
    img = medfilt2(img);    % median filtering to suppress outliers
    imshow(imresize(img,[size(img,1)*2,size(img,2)*2],'nearest'),[vmin,vmax],'border','tight');
    drawnow;
    F = getframe(fig);
    I = frame2im(F);
    [I,map] = rgb2ind(I,256);
    if i == 1
        imwrite(I,map,[cache_path_name,'/denoised.gif'],'gif','Loopcount',inf,'DelayTime',1/fps);
    else
        imwrite(I,map,[cache_path_name,'/denoised.gif'],'gif','WriteMode','append','DelayTime',1/fps);
    end
end

load([cache_path_name,'/measurement.mat'],'y')

fps = 10;
fig = figure;
set(gcf,'unit','normalized','position',[0.3,0.3,0.5,0.4])
for i = 1:300
    imshow(imresize(y(:,:,i),[size(y,1)*2,size(y,2)*2],'nearest'),[vmin,vmax],'border','tight');
    drawnow;
    F = getframe(fig);
    I = frame2im(F);
    [I,map] = rgb2ind(I,256);
    if i == 1
        imwrite(I,map,[cache_path_name,'/noisy.gif'],'gif','Loopcount',inf,'DelayTime',1/fps);
    else
        imwrite(I,map,[cache_path_name,'/noisy.gif'],'gif','WriteMode','append','DelayTime',1/fps);
    end
end

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

