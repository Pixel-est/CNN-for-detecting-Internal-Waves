% Detect internal waves in tiled 227x227 patches from Sentinel-1 image(s).
% This script:
% 1) Loads a trained network from IW_net.mat
% 2) Reads one image file or all images in a folder
% 3) Splits each image into full 227x227 tiles (no edge padding)
% 4) Classifies each tile as IW present/absent
% 5) Saves per-image outputs and one combined detections table

modelFile = "IW_net.mat";
imagePath = "S1_Images";             % File path or folder (all images are processed)
trainingDataFolder = "IW_data";      % Fallback only if model metadata missing
tileSize = [227 227];
iwThreshold = 0.60;                    % [] = use threshold from modelInfo; else numeric [0,1]
maxShownTiles = 100;

outputDir = "IW_Detections";
saveCombinedDetectionsCsv = true;
combinedDetectionsCsvFile = fullfile(outputDir, "IW_tile_detections_all.csv");
savePerImageDetectionsCsv = true;
saveOverlayImage = true;
saveProbabilityMapImage = true;

showOverlayFigure = true;
showProbabilityMapFigure = true;
showTopTilesMontage = true;
showOnlyIWPositiveFigures = true;    % Only show/save figures for images with IW-positive tiles

if ~isfile(modelFile)
    error("Model file not found: %s", modelFile);
end

if ~isfolder(outputDir)
    mkdir(outputDir);
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

imageFiles = iResolveImagePaths(imagePath);
if isempty(imageFiles)
    error("No image files found at: %s", imagePath);
end

allTileInfo = table();

fprintf("Processing %d image(s)...\n\n", numel(imageFiles));
for imageIdx = 1:numel(imageFiles)
    imageFile = imageFiles{imageIdx};
    [~, imageBase, ~] = fileparts(imageFile);
    safeBase = iSafeName(imageBase);

    im = imread(imageFile);
    imOrig = im;
    im = iEnsure3Channels(im);

    % Use full tiles across the image, including extra edge-aligned tiles.
    [H, W, ~] = size(im);
    yStarts = iTileStarts(H, tileSize(1));
    xStarts = iTileStarts(W, tileSize(2));
    nRows = numel(yStarts);
    nCols = numel(xStarts);
    nTiles = nRows * nCols;

    if nTiles == 0
        fprintf("Skipping %s (smaller than one %dx%d tile)\n\n", ...
            imageFile, tileSize(1), tileSize(2));
        continue;
    end

    usedH = yStarts(end) + tileSize(1) - 1;
    usedW = xStarts(end) + tileSize(2) - 1;
    overlapH = max(0, nRows * tileSize(1) - H);
    overlapW = max(0, nCols * tileSize(2) - W);

    tileImages = zeros(tileSize(1), tileSize(2), 3, nTiles, "single");
    tileInfo = table('Size', [nTiles 12], ...
        'VariableTypes', {'string','string','double','double','double','double','double','double','double','string','logical','double'}, ...
        'VariableNames', {'ImageFile','ImageName','TileRow','TileCol','Y1','X1','Y2','X2','IWScore','Decision','IsIW','ImageIndex'});

    idx = 1;
    for r = 1:nRows
        for c = 1:nCols
            y1 = yStarts(r);
            y2 = y1 + tileSize(1) - 1;
            x1 = xStarts(c);
            x2 = x1 + tileSize(2) - 1;

            patch = im(y1:y2, x1:x2, :);
            tileImages(:,:,:,idx) = single(patch);

            tileInfo.ImageFile(idx) = string(imageFile);
            tileInfo.ImageName(idx) = string(imageBase);
            tileInfo.TileRow(idx) = r;
            tileInfo.TileCol(idx) = c;
            tileInfo.Y1(idx) = y1;
            tileInfo.X1(idx) = x1;
            tileInfo.Y2(idx) = y2;
            tileInfo.X2(idx) = x2;
            tileInfo.ImageIndex(idx) = imageIdx;
            idx = idx + 1;
        end
    end

    tileImagesForDisplay = tileImages;
    if canUseGPU
        tileImages = gpuArray(tileImages);
    end

    scores = predict(net, tileImages);
    scores = iEnsureObsByClass(scores, nTiles);
    iwScores = double(scores(:, iwClassIdx));
    probMap = reshape(iwScores, [nCols nRows]).';
    xCenters = xStarts + (tileSize(2) - 1) / 2;
    yCenters = yStarts + (tileSize(1) - 1) / 2;

    isIW = iwScores >= iwThreshold;
    tileInfo.IWScore = iwScores;
    tileInfo.IsIW = isIW;
    tileInfo.Decision(:) = "Absent_IW";
    tileInfo.Decision(isIW) = string(iwClassName);

    iwTiles = tileInfo(isIW, :);

    fprintf("[%d/%d] Image: %s\n", imageIdx, numel(imageFiles), imageFile);
    fprintf("Original size: %d x %d\n", size(imOrig,1), size(imOrig,2));
    fprintf("Tile grid: %d rows x %d cols (%d tiles)\n", nRows, nCols, nTiles);
    fprintf("Edge-aligned coverage overlap: bottom=%d px, right=%d px\n", overlapH, overlapW);
    fprintf("IW threshold: %.3f\n", iwThreshold);
    fprintf("IW-positive tiles: %d\n\n", height(iwTiles));

    shouldRenderOverlay = (showOverlayFigure || saveOverlayImage) && ...
        (~showOnlyIWPositiveFigures || ~isempty(iwTiles));
    if shouldRenderOverlay
        figOverlay = figure('Visible', iOnOff(showOverlayFigure), ...
            'Name', sprintf('IW detections: %s', imageBase));
        imshow(imOrig);
        title(sprintf("%s | IW-positive tiles: %d (threshold %.2f)", ...
            imageBase, height(iwTiles), iwThreshold), 'Interpreter', 'none');
        hold on;
        for k = 1:height(iwTiles)
            x1 = iwTiles.X1(k);
            y1 = iwTiles.Y1(k);
            x2 = iwTiles.X2(k);
            y2 = iwTiles.Y2(k);
            rectangle('Position', [x1 y1 (x2-x1+1) (y2-y1+1)], ...
                'EdgeColor', 'g', 'LineWidth', 1.5);
            text(x1, max(1, y1-8), sprintf("%.2f", iwTiles.IWScore(k)), ...
                'Color', 'y', 'FontSize', 8, 'FontWeight', 'bold', 'BackgroundColor', 'k');
        end
        hold off;

        if saveOverlayImage
            overlayImageFile = fullfile(outputDir, sprintf("%s_overlay.png", safeBase));
            saveas(figOverlay, overlayImageFile);
        end
        if ~showOverlayFigure
            close(figOverlay);
        end
    end

    if showProbabilityMapFigure || saveProbabilityMapImage
        figProb = figure('Visible', iOnOff(showProbabilityMapFigure), ...
            'Name', sprintf('IW probability map: %s', imageBase));
        imagesc(xCenters, yCenters, probMap);
        axis image;
        set(gca, 'YDir', 'reverse');
        colormap(gca, turbo);
        clim([0 1]);
        colorbar;
        xlabel('X pixel');
        ylabel('Y pixel');
        title(sprintf('%s | IW probability map', imageBase), 'Interpreter', 'none');
        xlim([1 usedW]);
        ylim([1 usedH]);

        if saveProbabilityMapImage
            probabilityMapFile = fullfile(outputDir, sprintf("%s_probability_map.png", safeBase));
            saveas(figProb, probabilityMapFile);
        end
        if ~showProbabilityMapFigure
            close(figProb);
        end
    end

    shouldShowMontage = showTopTilesMontage && ...
        (~showOnlyIWPositiveFigures || ~isempty(iwTiles));
    if shouldShowMontage && ~isempty(iwTiles)
        [~, order] = sort(iwTiles.IWScore, 'descend');
        iwTilesSorted = iwTiles(order, :);
        nShow = min(height(iwTilesSorted), maxShownTiles);
        shownImages = cell(1, nShow);

        for k = 1:nShow
            r = iwTilesSorted.TileRow(k);
            c = iwTilesSorted.TileCol(k);
            linearIdx = (r - 1) * nCols + c;
            shownImages{k} = iToDisplayPatch(tileImagesForDisplay(:,:,:,linearIdx));
        end

        figure('Name', sprintf('IW-positive tiles: %s', imageBase));
        montage(shownImages, 'Size', [ceil(sqrt(nShow)) ceil(sqrt(nShow))], ...
            'BorderSize', [2 2], 'BackgroundColor', 'white');
        title(sprintf("%s | Top %d IW-positive tiles (227x227)", imageBase, nShow), ...
            'Interpreter', 'none');
    end

    if savePerImageDetectionsCsv
        perImageCsv = fullfile(outputDir, sprintf("%s_tile_detections.csv", safeBase));
        writetable(tileInfo, perImageCsv);
    end

    allTileInfo = [allTileInfo; tileInfo]; %#ok<AGROW>
end

if isempty(allTileInfo)
    warning("No valid tiles were processed from %s", imagePath);
else
    if saveCombinedDetectionsCsv
        writetable(allTileInfo, combinedDetectionsCsvFile);
        fprintf("Saved combined tile-level detection table to %s\n", combinedDetectionsCsvFile);
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
    t = double(S.modelInfo.decisionThreshold);
    if isfinite(t) && t >= 0 && t <= 1
        thr = t;
    end
end
end

function scores = iEnsureObsByClass(scores, numObs)
scores = iToCPU(scores);
if size(scores,1) ~= numObs && size(scores,2) == numObs
    scores = scores.';
end
end

function x = iToCPU(x)
if isa(x, "dlarray")
    x = extractdata(x);
end
if isa(x, "gpuArray")
    x = gather(x);
end
end

function out = iToDisplayPatch(patch)
patch = iToCPU(patch);
if isa(patch, "single") || isa(patch, "double")
    out = im2uint8(mat2gray(patch));
else
    out = patch;
end
end

function out = iSafeName(in)
out = regexprep(string(in), '[^A-Za-z0-9_.-]+', '_');
out = char(out);
end

function out = iOnOff(flag)
if flag
    out = 'on';
else
    out = 'off';
end
end
