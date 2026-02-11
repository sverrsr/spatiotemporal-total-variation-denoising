foldername = "re2500_weInf_surfElev_first2089_B1024_rayTrace_D3pi_png";

files = dir(fullfile(foldername, "*.png"));
assert(~isempty(files), "No PNGs found in folder");

% sort by name so time is correct
[~, idx] = sort({files.name});
files = files(idx);

K = numel(files);

img = im2double(imread(fullfile(foldername, files(1).name)));
[n1,n2] = size(img);

x = zeros(n1,n2,K);
x(:,:,1) = img;

for k = 2:K
    x(:,:,k) = im2double(imread(fullfile(foldername, files(k).name)));
end

%%
snr_val = 10;
y = awgn(x, snr_val);
[n1,n2,n3] = size(y);
