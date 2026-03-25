% Summarize IW seasonality by month using tile-level detections per image.
% An image is labeled Present_IW if enough 227x227 tiles exceed iwThreshold.

modelFile = "IW_net.mat";
imagePath = "S1_Images";                 % Folder or single image
trainingDataFolder = "IW_data";          % Fallback if model metadata missing
outputCsvFile = "IW_monthly_presence_summary.csv";
saveCsv = true;

tileSize = [227 227];
iwThreshold = [];                        % [] = use threshold from modelInfo
minPositiveTiles = 1;                    % Image is Present_IW if >= this many tiles are positive

if ~isfile(modelFile)
    error("Model file not found: %s", modelFile);
end

imageFiles = iResolveImagePaths(imagePath);
if isempty(imageFiles)
    error("No images found at: %s", imagePath);
end

S = load(modelFile);
if ~isfield(S, "net")
    error("Model file %s does not contain variable 'net'.", modelFile);
end
net = S.net;

classNames = iGetClassNamesFromModelOrData(S, net, trainingDataFolder);
[iwClassName, iwClassIdx] = iFindIWClass(classNames);

if isempty(iwThreshold)
    iwThreshold = iGetThresholdFromModelInfo(S);
end

numImages = numel(imageFiles);
imageDecision = strings(numImages,1);
maxTileIWScore = zeros(numImages,1);
meanTileIWScore = zeros(numImages,1);
numPositiveTiles = zeros(numImages,1);
numTiles = zeros(numImages,1);
acqDate = NaT(numImages, 1);
acqMonth = zeros(numImages, 1);

fprintf("Processing %d image(s)...\n\n", numImages);

for k = 1:numImages
    imageFile = imageFiles{k};
    im = imread(imageFile);
    im = iEnsure3Channels(im);

        [H, W, ~] = size(im);
        yStarts = iTileStarts(H, tileSize(1));
        xStarts = iTileStarts(W, tileSize(2));
        nRows = numel(yStarts);
        nCols = numel(xStarts);
        thisNumTiles = nRows * nCols;
        numTiles(k) = thisNumTiles;

    if thisNumTiles == 0
        imageDecision(k) = "TooSmall";
        maxTileIWScore(k) = NaN;
        meanTileIWScore(k) = NaN;
    else
        tileImages = zeros(tileSize(1), tileSize(2), 3, thisNumTiles, "single");
        idx = 1;
        for r = 1:nRows
            for c = 1:nCols
                y1 = yStarts(r);
                y2 = y1 + tileSize(1) - 1;
                x1 = xStarts(c);
                x2 = x1 + tileSize(2) - 1;
                tileImages(:,:,:,idx) = single(im(y1:y2, x1:x2, :));
                idx = idx + 1;
            end
        end

        if canUseGPU
            tileImages = gpuArray(tileImages);
        end

        scores = predict(net, tileImages);
        scores = iEnsureObsByClass(scores, thisNumTiles);
        iwScores = double(scores(:, iwClassIdx));

        numPositiveTiles(k) = sum(iwScores >= iwThreshold);
        maxTileIWScore(k) = max(iwScores);
        meanTileIWScore(k) = mean(iwScores);

        if numPositiveTiles(k) >= minPositiveTiles
            imageDecision(k) = string(iwClassName);
        else
            imageDecision(k) = iGetNegativeClassLabel(classNames, iwClassName);
        end
    end

    thisDate = iExtractDateFromFilename(imageFile);
    acqDate(k) = thisDate;
    if ~isnat(thisDate)
        acqMonth(k) = month(thisDate);
    end
end

results = table(string(imageFiles(:)), imageDecision, maxTileIWScore, meanTileIWScore, ...
    numPositiveTiles, numTiles, repmat(iwThreshold, numImages, 1), ...
    repmat(minPositiveTiles, numImages, 1), acqDate, acqMonth, ...
    'VariableNames', ["ImageFile","ThresholdDecision","MaxTileIWScore","MeanTileIWScore", ...
    "NumPositiveTiles","NumTiles","Threshold","MinPositiveTiles","AcquisitionDate","AcquisitionMonth"]);

isPresent = results.ThresholdDecision == string(iwClassName);
nPresent = sum(isPresent);
nAbsent = sum(results.ThresholdDecision ~= "TooSmall" & ~isPresent);
nUnknownDate = sum(isnat(results.AcquisitionDate));

fprintf("Total images: %d\n", height(results));
fprintf("Present_IW: %d\n", nPresent);
fprintf("Absent_IW: %d\n", nAbsent);
fprintf("Images with unparsed date: %d\n", nUnknownDate);
fprintf("Threshold used: %.3f\n", iwThreshold);
fprintf("Minimum positive tiles per image: %d\n", minPositiveTiles);

validPresent = isPresent & ~isnat(results.AcquisitionDate);
monthCounts = accumarray(month(results.AcquisitionDate(validPresent)), 1, [12 1], @sum, 0);

figure("Name", "IW present count per month");
bar(1:12, monthCounts, "FaceColor", [0.2 0.5 0.8]);
grid on;
xlabel("Month");
ylabel("IW-present image count");
title("Internal Wave Presence by Month");
xticks(1:12);
xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]);

if saveCsv
    writetable(results, outputCsvFile);
    fprintf("Saved per-image summary to %s\n", outputCsvFile);
end

function value = iToCPU(value)
if isa(value, "dlarray")
    value = extractdata(value);
end
if isa(value, "gpuArray")
    value = gather(value);
end
end

function scores = iEnsureObsByClass(scores, numObs)
scores = iToCPU(scores);
if size(scores,1) ~= numObs && size(scores,2) == numObs
    scores = scores.';
end
end

function imageFiles = iResolveImagePaths(imagePath)
if isfile(imagePath)
    imageFiles = {char(imagePath)};
    return;
end

if isfolder(imagePath)
    imds = imageDatastore(imagePath, ...
        IncludeSubfolders=true, ...
        FileExtensions={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'});
    imageFiles = imds.Files;
    return;
end

error("Path does not exist: %s", imagePath);
end

function starts = iTileStarts(imageLength, tileLength)
if imageLength < tileLength
    starts = [];
    return;
end

starts = 1:tileLength:(imageLength - tileLength + 1);
lastStart = imageLength - tileLength + 1;
if starts(end) ~= lastStart
    starts(end+1) = lastStart; %#ok<AGROW>
end
end

function im = iEnsure3Channels(im)
if ndims(im) == 2
    im = repmat(im, 1, 1, 3);
elseif size(im, 3) == 1
    im = repmat(im, 1, 1, 3);
elseif size(im, 3) > 3
    im = im(:,:,1:3);
end
end

function classNames = iGetClassNamesFromModelOrData(S, net, trainingDataFolder)
classNames = [];

if isfield(S, "modelInfo") && isfield(S.modelInfo, "classNames")
    classNames = cellstr(string(S.modelInfo.classNames));
end

if isempty(classNames) && isfolder(trainingDataFolder)
    imds = imageDatastore(trainingDataFolder, ...
        IncludeSubfolders=true, LabelSource="foldernames");
    classNames = categories(imds.Labels);
end

if isempty(classNames) && isprop(net, "Classes") && ~isempty(net.Classes)
    netClasses = net.Classes;
    if iscategorical(netClasses)
        classNames = categories(netClasses);
    elseif isstring(netClasses)
        classNames = cellstr(netClasses);
    elseif iscellstr(netClasses)
        classNames = netClasses;
    end
end

if isempty(classNames)
    classNames = {'Absent_IW','Present_IW'};
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

function thr = iGetThresholdFromModelInfo(S)
thr = 0.50;
if isfield(S, "modelInfo") && isfield(S.modelInfo, "decisionThreshold")
    thisThr = double(S.modelInfo.decisionThreshold);
    if isfinite(thisThr) && thisThr >= 0 && thisThr <= 1
        thr = thisThr;
    end
end
end

function label = iGetNegativeClassLabel(classNames, iwClassName)
classStrings = string(classNames);
notIW = classStrings(classStrings ~= iwClassName);
if isempty(notIW)
    label = "Absent_IW";
else
    label = notIW(1);
end
end

function dt = iExtractDateFromFilename(imageFile)
dt = NaT;
[~, nameOnly, ~] = fileparts(imageFile);

token = regexp(nameOnly, '(?<date>\d{4}-\d{2}-\d{2})', 'names', 'once');
if ~isempty(token)
    try
        dt = datetime(token.date, "InputFormat", "yyyy-MM-dd");
        return;
    catch
    end
end

token2 = regexp(nameOnly, '(?<date>\d{8})', 'names', 'once');
if ~isempty(token2)
    try
        dt = datetime(token2.date, "InputFormat", "yyyyMMdd");
    catch
    end
end
end
