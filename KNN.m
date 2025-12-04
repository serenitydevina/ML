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
% 3. Ekstraksi fitur (warna + tekstur)
% --------------------------------------------------
numImages = numel(imds.Files);
features = zeros(numImages, 4);  % fitur: [meanR meanG meanB entropy]
labels = imds.Labels;

for i = 1:numImages
    img = readimage(imds, i);
    
    img = imresize(img, inputSize);
    
    % Konversi ke RGB
    imgRGB = im2double(img);
    
    % Fitur Warna (rata-rata RGB)
    R = mean(mean(imgRGB(:,:,1)));
    G = mean(mean(imgRGB(:,:,2)));
    B = mean(mean(imgRGB(:,:,3)));
    
    % Fitur Tekstur (Entropy)
    gray = rgb2gray(imgRGB);
    E = entropy(gray);

    features(i,:) = [R G B E];
end

disp('Ekstraksi fitur selesai!');

% Membagi data 80% train, 20% test
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

Ypred = predict(knnModel, Xtest);

% Akurasi
akurasi = sum(Ypred == Ytest) / numel(Ytest) * 100;
fprintf('Akurasi KNN: %.2f%%\n', akurasi);

% Confusion matrix
figure;
confusionchart(Ytest, Ypred);
title('Confusion Matrix KNN');

% Load gambar yang ingin diprediksi
img = imread('immature1.jpg');
img = imresize(img, inputSize);
imgRGB = im2double(img);

% Fitur warna
R = mean(mean(imgRGB(:,:,1)));
G = mean(mean(imgRGB(:,:,2)));
B = mean(mean(imgRGB(:,:,3)));

% Fitur tekstur
E = entropy(rgb2gray(imgRGB));

fiturUji = [R G B E];

hasil = predict(knnModel, fiturUji);

fprintf('Prediksi tingkat kematangan: %s\n', hasil);

