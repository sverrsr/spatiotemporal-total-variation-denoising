% Side-by-side video from two folders of 256x256 grayscale PNGs
clear; clc;

folderA = "C:\Users\sverr\Documents\NTNU\Prosjekt\Experiments\imageAndVideoConversion\re2500_weInf_surfElev_first2089_B1024_rayTrace_D3pi_png";
folderB = "C:\Users\sverr\Documents\NTNU\Prosjekt\Experiments\spatiotemporal-total-variation-denoising\cache_raw_try_all\png_input\denoised_png";
outFile = "side_by_side.mp4";
fps = 10;

filesA = dir(fullfile(folderA, "*.png"));
filesB = dir(fullfile(folderB, "*.png"));

% Sort by name to keep consistent ordering
[~, ia] = sort({filesA.name}); filesA = filesA(ia);
[~, ib] = sort({filesB.name}); filesB = filesB(ib);

n = min(numel(filesA), numel(filesB));
if n == 0
    error("No PNGs found in one or both folders.");
end

vw = VideoWriter(outFile, "MPEG-4");
vw.FrameRate = fps;
open(vw);

for k = 1:900
    imgA = imread(fullfile(folderA, filesA(k).name));
    imgB = imread(fullfile(folderB, filesB(k).name));

    % Ensure grayscale 2D
    if ndims(imgA) == 3, imgA = rgb2gray(imgA); end
    if ndims(imgB) == 3, imgB = rgb2gray(imgB); end

    % Ensure same size (in case)
    if ~isequal(size(imgA), [256 256])
        imgA = imresize(imgA, [256 256], "nearest");
    end
    if ~isequal(size(imgB), [256 256])
        imgB = imresize(imgB, [256 256], "nearest");
    end
    

    % Side-by-side: 256x512
    frameGray = [imadjust(im2uint8(imgA)), im2uint8(imgB)];

    % VideoWriter expects RGB for many players -> convert to 3-channel
    frameRGB = repmat(frameGray, 1, 1, 3);

    writeVideo(vw, frameRGB);
end

close(vw);
fprintf("Wrote %d frames to %s\n", n, outFile);
