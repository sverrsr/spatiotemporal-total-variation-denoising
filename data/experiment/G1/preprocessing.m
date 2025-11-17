% =========================================================================
% Introduction
% =========================================================================
% This code performs data pre-processing (dark-frame calibration) to the 
% captured raw image sequences under extremely low-light conditions.
%
% Author: Yunhui Gao (gyh21@mails.tsinghua.edu.cn)
% =========================================================================
%%
% =========================================================================
% Dark-frame calculation
% =========================================================================
clear all;
close all;
clc;

prefix = 'VimbaImage_';

foldername = 'dark_frames';

img = imrotate(im2double(imread([foldername,'/',prefix,'1.bmp'])),90);
[n1,n2] = size(img);

img_dark = zeros(n1,n2);
figure
nt = 1000;
for i = 1:nt
    disp([num2str(i,'%04d'),'/',num2str(nt,'%04d')])
    img = imrotate(im2double(imread([foldername,'/',prefix,num2str(i),'.bmp'])),90);
    img_dark = img_dark + img/nt;
    imshow(img,[0,0.05])
    drawnow;
end

% display the averaged dark frame
figure,imshow(img_dark,[0.,0.05])

%%
% =========================================================================
% Dark-frame compensation
% =========================================================================

prefix = 'VimbaImage_';

foldername = 'test_frames';

img = imrotate(im2double(imread([foldername,'/',prefix,'1.bmp'])),90);
[n1,n2] = size(img);
nt = 300;

vid = zeros(n1,n2,nt);

figure
for i = 1:nt
    disp([num2str(i,'%04d'),'/',num2str(nt,'%04d')])
    img = imrotate(im2double(imread([foldername,'/',prefix,num2str(i),'.bmp'])),90);
    vid(:,:,i) = img-img_dark;
    imshow(img-img_dark,[0,0.02])
    drawnow;
end

% save calibrated data
save('data.mat','vid')
