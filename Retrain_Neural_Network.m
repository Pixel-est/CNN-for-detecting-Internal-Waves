% Retrain IW classifier and save model + metadata for downstream scripts.

rng(42, "twister");

% User settings
folderName = "IW_data";
modelFile = "IW_net.mat";
trainRatio = 0.70;
valRatio = 0.15;
testRatio = 0.15; %#ok<NASGU>
showNetworkGraph = false;

% Build datastore
imds = imageDatastore(folderName, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

classNames = categories(imds.Labels);
numClasses = numel(classNames);
fprintf("Classes: %s\n", strjoin(string(classNames), ", "));
fprintf("Total images: %d\n", numel(imds.Files));

% Preview random training samples
idx = randperm(numel(imds.Labels), min(16, numel(imds.Labels)));
figure;
imshow(imtile(imds, Frames=idx));
title("Random training examples");

% Stratified split
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, trainRatio, valRatio, "randomized");
fprintf("Train/Val/Test: %d / %d / %d\n", numel(imdsTrain.Files), numel(imdsValidation.Files), numel(imdsTest.Files));

% Create transfer-learning model
net = imagePretrainedNetwork("squeezenet", NumClasses=numClasses);
inputSize = networkInputSize(net);
[headLayerName, ~] = networkHead(net);
net = freezeNetwork(net, LayerNamesToIgnore=headLayerName);

if showNetworkGraph
    analyzeNetwork(net);
end

% Data augmentation
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandXTranslation=[-30 30], ...
    RandYTranslation=[-30 30], ...
    RandRotation=[-12 12]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, DataAugmentation=augmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Training options
validationFrequency = max(5, floor(numel(imdsTrain.Files) / 16));
options = trainingOptions("adam", ...
    MaxEpochs=12, ...
    MiniBatchSize=32, ...
    InitialLearnRate=3e-4, ...
    L2Regularization=1e-4, ...
    Shuffle="every-epoch", ...
    ValidationData=augimdsValidation, ...
    ValidationFrequency=validationFrequency, ...
    ValidationPatience=5, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

% Train
net = trainnet(augimdsTrain, net, "crossentropy", options);

% Compute scores
scoresVal = minibatchpredict(net, augimdsValidation);
scoresVal = iEnsureObsByClass(scoresVal, numel(imdsValidation.Files));
scoresTest = minibatchpredict(net, augimdsTest);
scoresTest = iEnsureObsByClass(scoresTest, numel(imdsTest.Files));

% Determine IW class and optimize threshold on validation set for F1
[iwClassName, iwClassIdx] = iFindIWClass(classNames);
[bestThreshold, bestValPrecision, bestValRecall, bestValF1] = iBestF1Threshold( ...
    scoresVal(:, iwClassIdx), imdsValidation.Labels, iwClassName);

% Evaluate on test data
YTestArgmax = scores2label(scoresTest, classNames);
TTest = imdsTest.Labels;
accArgmax = mean(YTestArgmax == TTest);

isIWTrueTest = string(TTest) == iwClassName;
isIWPredTest = scoresTest(:, iwClassIdx) >= bestThreshold;
[testPrecision, testRecall, testF1] = iBinaryMetrics(isIWTrueTest, isIWPredTest);
[rocFPR, rocTPR, rocThresholds, testAUC] = iComputeROC(isIWTrueTest, scoresTest(:, iwClassIdx));

fprintf("\nArgmax test accuracy: %.4f\n", accArgmax);
fprintf("Validation-tuned threshold for %s: %.3f\n", iwClassName, bestThreshold);
fprintf("Test precision/recall/F1 at threshold: %.4f / %.4f / %.4f\n", ...
    testPrecision, testRecall, testF1);
fprintf("Test ROC AUC (%s vs not %s): %.4f\n", iwClassName, iwClassName, testAUC);

figure;
confusionchart(TTest, YTestArgmax);
title("Test confusion matrix (argmax)");

figure;
plot(rocFPR, rocTPR, "LineWidth", 1.5);
grid on;
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title(sprintf("Test ROC (%s), AUC = %.4f", iwClassName, testAUC));

% Save model and metadata for inference scripts
modelInfo = struct();
modelInfo.classNames = string(classNames);
modelInfo.inputSize = inputSize;
modelInfo.iwClassName = iwClassName;
modelInfo.iwClassIdx = iwClassIdx;
modelInfo.decisionThreshold = bestThreshold;
modelInfo.validationF1 = bestValF1;
modelInfo.validationPrecision = bestValPrecision;
modelInfo.validationRecall = bestValRecall;
modelInfo.testArgmaxAccuracy = accArgmax;
modelInfo.testThresholdPrecision = testPrecision;
modelInfo.testThresholdRecall = testRecall;
modelInfo.testThresholdF1 = testF1;
modelInfo.testAUC = testAUC;
modelInfo.testROC_FPR = rocFPR;
modelInfo.testROC_TPR = rocTPR;
modelInfo.testROC_Thresholds = rocThresholds;
modelInfo.trainingDate = datetime("now");

save(modelFile, "net", "modelInfo");
fprintf("Saved model to %s\n", modelFile);

% Optional single-image sanity check (if file exists)
quickTestFile = "IW_dataTest.jpg";
if isfile(quickTestFile)
    im = imread(quickTestFile);
    im = iEnsure3Channels(im);
    im = imresize(im, inputSize(1:2));
    X = single(im);
    if canUseGPU
        X = gpuArray(X);
    end
    s = predict(net, X);
    s = iEnsureObsByClass(s, 1);
    iwScore = s(1, iwClassIdx);
    isIW = iwScore >= bestThreshold;
    quickLabel = iLabelFromDecision(isIW, classNames, iwClassName);

    figure;
    imshow(im);
    title(sprintf("%s | IW score %.3f (thr %.3f)", quickLabel, iwScore, bestThreshold));
end

function scores = iEnsureObsByClass(scores, numObs)
scores = iToCPU(scores);
if size(scores, 1) ~= numObs && size(scores, 2) == numObs
    scores = scores.';
end
end

function [iwClassName, iwClassIdx] = iFindIWClass(classNames)
classStrings = string(classNames);
iwClassIdx = find(strcmpi(classStrings, "Present_IW"), 1);
if isempty(iwClassIdx)
    iwClassIdx = find(contains(lower(classStrings), "present"), 1);
end
if isempty(iwClassIdx)
    notAbsent = ~contains(lower(classStrings), "absent");
    candidates = find(notAbsent);
    if ~isempty(candidates)
        iwClassIdx = candidates(1);
    end
end
if isempty(iwClassIdx)
    iwClassIdx = min(2, numel(classStrings));
end
iwClassName = classStrings(iwClassIdx);
end

function [bestThr, bestPrec, bestRec, bestF1] = iBestF1Threshold(iwScores, trueLabels, iwClassName)
iwScores = double(iToCPU(iwScores));
trueIW = string(trueLabels) == iwClassName;

thresholds = 0:0.01:1;
bestThr = 0.50;
bestPrec = 0;
bestRec = 0;
bestF1 = -inf;

for t = thresholds
    predIW = iwScores >= t;
    [p, r, f1] = iBinaryMetrics(trueIW, predIW);
    if f1 > bestF1
        bestF1 = f1;
        bestThr = t;
        bestPrec = p;
        bestRec = r;
    end
end
end

function [precision, recall, f1] = iBinaryMetrics(truePositiveClass, predictedPositiveClass)
tp = sum(truePositiveClass & predictedPositiveClass);
fp = sum(~truePositiveClass & predictedPositiveClass);
fn = sum(truePositiveClass & ~predictedPositiveClass);

precision = tp / max(tp + fp, 1);
recall = tp / max(tp + fn, 1);
f1 = (2 * precision * recall) / max(precision + recall, eps);
end

function [fpr, tpr, thresholds, auc] = iComputeROC(truePositiveClass, positiveScores)
yTrue = logical(iToCPU(truePositiveClass));
yScore = double(iToCPU(positiveScores));
[fpr, tpr, thresholds, auc] = perfcurve(yTrue, yScore, true);
end

function out = iToCPU(x)
out = x;
if isa(out, "dlarray")
    out = extractdata(out);
end
if isa(out, "gpuArray")
    out = gather(out);
end
end

function label = iLabelFromDecision(isIW, classNames, iwClassName)
classStrings = string(classNames);
if isIW
    label = char(iwClassName);
    return;
end

notIW = classStrings(classStrings ~= iwClassName);
if isempty(notIW)
    label = "Not_IW";
else
    label = char(notIW(1));
end
end
