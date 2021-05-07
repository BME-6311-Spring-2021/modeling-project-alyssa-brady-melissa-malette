% Lyme disease neural network 

% Part 1: Create a datastore to the images 
imds = imageDatastore("LymeTraining");

% Part 2: Get the known classifications from a file and use these as the
% image labels 
groundtruth = readtable("LymeTrainingData.csv");
imds.Labels = categorical(groundtruth.Status);

% Part 3: View the first few images
%figure 
%imshow(readimage(imds,1))
%figure
%imshow(readimage(imds,2))
%figure
%imshow(readimage(imds,3))

% Part 4: Divide the data into training (60%) and testing (40%) sets 
%[trainImgs,testImgs] = splitEachLabel(imds,0.6,"randomized");

%% Put in the test data and change test ds 

imds3= imageDatastore("LymeTestSet");
groundtruth3= readtable("LymeTestDataNEW.csv");
imds3.Labels= categorical(groundtruth3.Status);


% Part 5: Create augmented image datastores to preprocess the images 
trainds = augmentedImageDatastore([224 224],imds,"ColorPreprocessing","gray2rgb");
testds = augmentedImageDatastore([224 224],imds3,"ColorPreprocessing","gray2rgb");

% Part 6: Build a network using pretrained GoogLeNet 
net = googlenet;
lgraph = layerGraph(net);

% Part 7: Take the CNN layer graph and replace the output layers 
newFc = fullyConnectedLayer(2,"Name","new_fc")
lgraph = replaceLayer(lgraph,"loss3-classifier",newFc)
newOut = classificationLayer("Name","new_out")
lgraph = replaceLayer(lgraph,"output",newOut)
%%
% Part 8: Set training options and import validation set
imds2 = imageDatastore("LymeValidation2");
groundtruth2 = readtable("LymeValidation2.csv");
imds2.Labels = categorical(groundtruth2.Status);

%figure 
%imshow(readimage(imds2,2))

[trainImgs2,testImgs2] = splitEachLabel(imds2,0.6,"randomized");

trainds2 = augmentedImageDatastore([224 224],trainImgs2,"ColorPreprocessing","gray2rgb");
testds2 = augmentedImageDatastore([224 224],testImgs2,"ColorPreprocessing","gray2rgb");


options = trainingOptions("sgdm","InitialLearnRate", 0.01,'MaxEpochs',20,'ValidationData',imds2,'ValidationFrequency',2,'Plots','training-progress');
%%
% Part 9: Train the network 
lymesnet = trainNetwork(trainds,lgraph,options)

% Part 10: Evaluate network on test data
preds = classify(lymesnet,testds);
truetest = imds3.Labels;
accuracy=nnz(preds == truetest)/numel(preds)

% Part 11: Confusion matrix
figure
confusionchart(truetest,preds)

% Part 12: View first incorrect classificaiton 
idx = find(preds~=truetest)
imshow(readimage(imds3,idx(1)))
title(truetest(idx(1)))

% Part 13: View second incorrect classificaiton 
imshow(readimage(imds3,idx(2)))
title(truetest(idx(2)))