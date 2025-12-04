clc; clear; close all;

% --------------------------------------------------
% 1. Membaca dataset image dengan imageDatastore
% --------------------------------------------------
imds = imageDatastore('sample', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% --------------------------------------------------
% 2. Resize semua gambar agar seragam
% --------------------------------------------------
inputSize = [128 128];

% --------------------------------------------------
% 3. Ekstraksi fitur (mean RGB + entropy)
% --------------------------------------------------
numImages = numel(imds.Files);
features = zeros(numImages, 4);  
labels = imds.Labels;

for i = 1:numImages
    img = readimage(imds, i);
    img = imresize(img, inputSize);
    imgRGB = im2double(img);

    % Mean RGB
    R = mean(mean(imgRGB(:,:,1)));
    G = mean(mean(imgRGB(:,:,2)));
    B = mean(mean(imgRGB(:,:,3)));

    % Entropy
    gray = rgb2gray(imgRGB);
    E = entropy(gray);

    features(i,:) = [R G B E];
end

disp('Ekstraksi fitur selesai!');

% Split 80/20
cv = cvpartition(labels,'HoldOut',0.2);

Xtrain = features(training(cv), :);
Ytrain = labels(training(cv));

Xtest  = features(test(cv), :);
Ytest  = labels(test(cv));

% --------------------------------------------------
% 4. Train model KNN
% --------------------------------------------------
knnModel = fitcknn(Xtrain, Ytrain, ...
    'NumNeighbors', 5, ...
    'Standardize', true);

disp('Model KNN berhasil dibuat!');

% Simpan model
save('KNN.mat', 'knnModel', 'inputSize');
disp('Model disimpan sebagai KNN.mat');

Ypred = predict(knnModel, Xtest);

% Akurasi
akurasi = sum(Ypred == Ytest) / numel(Ytest) * 100;
fprintf('Akurasi KNN: %.2f%%\n', akurasi);

% Confusion matrix
figure;
confusionchart(Ytest, Ypred);
title('Confusion Matrix KNN');