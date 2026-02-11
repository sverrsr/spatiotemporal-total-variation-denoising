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

%% My part
foldername = 'C:\Users\sverr\Documents\NTNU\Prosjekt\Experiments\imageAndVideoConversion\re2500_weInf_surfElev_first2089_B1024_rayTrace_D3pi_png';
cache_path_name = fullfile('cache_raw_try_all','png_input');

gpu = false;

files = dir(fullfile(foldername,'*.png'));
assert(~isempty(files),'No PNG files found.');

% sort by filename (important for time order)
[~,idx] = sort({files.name});
files = files(idx);

K_total = numel(files);

% read first image to get size
img = im2double(imread(fullfile(foldername,files(1).name)));
if size(img,3) == 3
    img = rgb2gray(img);   % force grayscale if needed
end

[n1,n2] = size(img);
y = zeros(n1,n2,K_total,'single');
y(:,:,1) = single(img);

% read remaining frames
for k = 2:K_total
    img = imread(fullfile(foldername,files(k).name));
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    y(:,:,k) = single(im2double(img));
end

%% END

% % % display the noisy video
% % figure
% % for k = 1:49
% %     imshow(imadjust(y(:,:,k)),[0,1]);
% %     title('Captured noisy video')
% %     drawnow;
% % end

% Global normalization
ymin = prctile(y(:), 0.5);
ymax = prctile(y(:),99.5);
y = (y - ymin)./(ymax - ymin);



if ~isfolder(cache_path_name)
    mkdir(cache_path_name)
end

% save pre-processed noisy video
save([cache_path_name,'/measurement.mat'],'y')

%%
% =========================================================================
% Run video denoising algorithm batch by batch
% =========================================================================


K = 100;        % batch size (# of images)

frame_start = 1;
frame_step = 50;
frame_end = K_total;

for frame = frame_start:frame_step:frame_end-K+1
    
    load([cache_path_name,'/measurement.mat'],'y')
    y = y(:,:,frame:frame+K-1);
    
    [n1,n2,n3] = size(y);
    
    % set regularization parameters

    lam_s = 0.05;
    lam_t = 0.15;
    n_iters = 200;

    % % lam_s = 5e-2;   % spatial regularization coefficient 0.05
    % % lam_t = 15e-2;  % temporal regularization coefficient 0.15
    % % 
    % % n_iters = 200;  % number of iterations
    
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
        % % w_next(:,:,:,1) = min(abs(w_next(:,:,:,1)),lam_s).*exp(1i*angle(w_next(:,:,:,1)));
        % % w_next(:,:,:,2) = min(abs(w_next(:,:,:,2)),lam_s).*exp(1i*angle(w_next(:,:,:,2)));
        % % w_next(:,:,:,3) = min(abs(w_next(:,:,:,3)),lam_t).*exp(1i*angle(w_next(:,:,:,3)));

        a = abs(w_next(:,:,:,1)); w_next(:,:,:,1) = w_next(:,:,:,1) .* min(1, lam_s ./ max(a, 1e-8));
        a = abs(w_next(:,:,:,2)); w_next(:,:,:,2) = w_next(:,:,:,2) .* min(1, lam_s ./ max(a, 1e-8));
        a = abs(w_next(:,:,:,3)); w_next(:,:,:,3) = w_next(:,:,:,3) .* min(1, lam_t ./ max(a, 1e-8));

    
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
    
    %% Lagre bilder
    % create output folder (once per batch)
    png_folder = fullfile(cache_path_name, 'denoised_png');
    if ~isfolder(png_folder)
        mkdir(png_folder);
    end
    
    % save each frame as PNG
    for k = 1:size(vid_denoised,3)
        frame_img = vid_denoised(:,:,k);
    
        % optional: clip to [0,1] for safety
        frame_img = max(min(frame_img,1),0);
    
        imwrite(frame_img, ...
            fullfile(png_folder, sprintf('frame_%04d.png', frame + k - 1)));
    end
    %% Slutt lagre bilder

    % % figure
    % % for k = 1:size(vid_denoised,3)
    % %     imshow(vid_denoised(:,:,k),[0.3,0.55]);
    % %     title('Denoised video')
    % %     drawnow;
    % % end
    % % close
    
    % save results
    save([cache_path_name,'/',num2str(frame),'_',num2str(frame+99),'.mat'],'vid_denoised');
end

%%
% =========================================================================
% Post-process denoised videos
% =========================================================================


K = 100;        % batch size (# of images)

frame_start = 1;
frame_step = 50;
frame_end = K_total;

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
% % fig = figure;
% % set(gcf,'unit','normalized','position',[0.3,0.3,0.5,0.4])
% % for i = 1:size(vid,3)
% %     img = real(vid(:,:,i));
% %     img = medfilt2(img);    % median filtering to suppress outliers
% %     imshow(imresize(img,[size(img,1)*2,size(img,2)*2],'nearest'),[vmin,vmax],'border','tight');
% %     drawnow;
% %     F = getframe(fig);
% %     I = frame2im(F);
% %     [I,map] = rgb2ind(I,256);
% %     if i == 1
% %         imwrite(I,map,[cache_path_name,'/denoised.gif'],'gif','Loopcount',inf,'DelayTime',1/fps);
% %     else
% %         imwrite(I,map,[cache_path_name,'/denoised.gif'],'gif','WriteMode','append','DelayTime',1/fps);
% %     end
% % end

load([cache_path_name,'/measurement.mat'],'y')

fps = 10;
% % fig = figure;
% % set(gcf,'unit','normalized','position',[0.3,0.3,0.5,0.4])
% % for i = 1:49
% %     imshow(imresize(y(:,:,i),[size(y,1)*2,size(y,2)*2],'nearest'),[vmin,vmax],'border','tight');
% %     drawnow;
% %     F = getframe(fig);
% %     I = frame2im(F);
% %     [I,map] = rgb2ind(I,256);
% %     if i == 1
% %         imwrite(I,map,[cache_path_name,'/noisy.gif'],'gif','Loopcount',inf,'DelayTime',1/fps);
% %     else
% %         imwrite(I,map,[cache_path_name,'/noisy.gif'],'gif','WriteMode','append','DelayTime',1/fps);
% %     end
% % end

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

