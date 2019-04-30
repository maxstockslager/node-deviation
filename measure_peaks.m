function measurePeaks

clear all, close all
addpath('helper-functions');

% Peak detection settings (little effect on fit) 
detSettings.quantileWidth = 100; % use 250-1500 depending on signal, magnitude of fluid density changes
detSettings.quantile = 75; 
detSettings.sgOrder = 3; 
detSettings.sgWidth = 7;
detSettings.decimationFactor = 10;
detSettings.datarate = 1000; 
detSettings.expectingFloaters = false; 
detSettings.detectionThreshold = -10; % all -25 for PC3s, -4 for L1210s
detSettings.tripletTime = 1600;  % antinode peaks must be within this gap of one another (data pts)
detSettings.gapTime = 1200; % minimum datapoints in between triplets 
detSettings.maxHeightDiff = 3; % max difference between antinode peaks (hz) --- 50 for PC3s, 5 for L1210s

% Peak fitting settings 
fitSettings.baselineStartFactor = 0.35; % distance from each antinode (as multiple of the
    % separation between antinodes) where the baseline starts
fitSettings.baselineContinueFactor = 1; % distance (as multiple of the
    % separation between antinodes) that the baseline extends on either side
fitSettings.baselineFitOrder = 2; % 0 for constant, 1 for linear, 2 for quadratic, etc
fitSettings.peakFitOrder = 4; % for fitting antinodes + nodes. best if even
fitSettings.peakFitFactor = 0.1; % controls how much signal we fit on
    % either side of the nodes and antinodes

% Peak approval/QC settings 
qcSettings.antinodeMismatchLimit = 0.5;
qcSettings.NDMismatchLimit = 0.50;
qcSettings.minTransitTime = 150; % ms (make sure datarate is entered correctly)
qcSettings.bandwidth = 200; % Hz
qcSettings.largewindowSize = 2e6; % points to plot per window for quality control
qcSettings.smallWindowSize = 1200; 
    
% LOAD IN DATA
% directory = uigetdir;
directory = 'C:\Users\Max\Documents\Program Data\MATLAB\ND\ND test';
data.frequencySignal = readFrequencySignal([directory, '\c00.bin']);
data.frequencyTime = transpose(1:length(data.frequencySignal)) / detSettings.datarate;

% FILTERING
data.lowpass = sgolayfilt(data.frequencySignal, detSettings.sgOrder, detSettings.sgWidth);
data.baseline = computeBaseline(data.lowpass, detSettings);
data.bandpass = data.lowpass - data.baseline; 
% data.highPass = data.frequencySignal-data.baseline;
% data.bandpass = sgolayfilt(data.highPass, detSettings.sgOrder, detSettings.sgWidth);

if detSettings.expectingFloaters
    data.bandpass = -data.bandpass;
end

% PEAK DETECTION
    % Find continuous segments of data below the threshold - these are how
    % we define "peaks". 
peaks.indicesBelowThreshold = find(data.bandpass < detSettings.detectionThreshold);
peaks.peakStartIndices = peaks.indicesBelowThreshold([1; find(diff(peaks.indicesBelowThreshold)>1)+1]);
peaks.peakEndIndices = peaks.indicesBelowThreshold([find(diff(peaks.indicesBelowThreshold) > 1); length(peaks.indicesBelowThreshold)]);
[peaks.heights, peaks.centers] = minimumWithinRange(peaks.peakStartIndices, peaks.peakEndIndices, data.bandpass);

% TRIPLET MATCHING
    % Find sets of peaks that meet these criteria:
        % Three peaks within a specified time (~200 ms)
        % No other peaks within a specified time (~200 ms)
    % These criteria are how we define second-mode "triplets".
    % The list "tripletPeakIndices" gives the index of the center peak,
    % corresponding to the tip of the cantilever. 

peaks.tripletPeakIndices = findTripletPeakIndices(peaks.centers, peaks.heights, detSettings);

% PEAK MEASUREMENT
    % Take the set of three peaks and measure the distance between the two
    % antinodes and the baseline. Again, the list "tripletPeakIndices"
    % gives the indices of the center peaks, which correspond to the cell
    % reaching the end of the cantilever. 

peaks.baselineRange = {}; % cell
peaks.leftAntinode = [];
peaks.rightAntinode = [];
peaks.baselineSlope = [];
peaks.baselineInt = []; 
peaks.leftAntinode = []; 
peaks.rightAntinode = []; 
peaks.antinodeRange1 = {}; % cell
peaks.antinodeRange2 = {}; % cell
peaks.quadfit1 = {}; % cell
peaks.quadfit2 = {}; % cell
peaks.nodeRange1 = {}; % cell
peaks.nodeRange2 = {}; % cell
peaks.nodeFit1 = {}; % cell
peaks.nodeFit2 = {}; % cell
peaks.node1Index = [];   
peaks.node2Index = [];
peaks.leftNode = [];
peaks.rightNode = [];
peaks.tBaseline = {};

for cellNumber = 1:length(peaks.tripletPeakIndices) % for each node peak 
   
    currentPeakFit = getPeakFit(peaks.tripletPeakIndices(cellNumber), ...
        peaks, data, fitSettings); % structure w/ all peak fit data
    
    
    
    
    
%      [tripletHeight, baselineRange, baselineSlope, baselineInt, leftAntinode, rightAntinode, antinodeRange1, antinodeRange2, quadfit1, quadfit2, ...
%         nodeRange1, nodeRange2, nodeFit1, nodeFit2, node1Index, node2Index, leftNode, rightNode, tBaseline] = ...
%         findAverageTripletHeight(peakNumber, peaks.centers, data.frequencySignal, fitSettings.sidepointsFraction, settings);

    % assign to data structure
    peaks.tripletHeights(ii) = tripletHeight;
    peaks.baselineRange{end+1} = baselineRange;
    peaks.baselineSlope(end+1) = baselineSlope; 
    peaks.baselineInt(end+1) = baselineInt;
    peaks.leftAntinode(end+1) = leftAntinode;
    peaks.rightAntinode(end+1) = rightAntinode;
    peaks.antinodeRange1{end+1} = antinodeRange1;
    peaks.antinodeRange2{end+1} = antinodeRange2;
    peaks.quadfit1{end+1} = quadfit1;
    peaks.quadfit2{end+1} = quadfit2;    
    peaks.nodeRange1{end+1} = nodeRange1;
    peaks.nodeRange2{end+1} = nodeRange2;
    peaks.nodeFit1{end+1} = nodeFit1;
    peaks.nodeFit2{end+1} = nodeFit2; 
    peaks.node1Index(end+1) = node1Index;
    peaks.node2Index(end+1) = node2Index;
    peaks.leftNode(end+1) = leftNode;
    peaks.rightNode(end+1) = rightNode; 
    peaks.tBaseline{end+1} = tBaseline;
end

    peaks.tripletHeights = peaks.tripletHeights'; 
    
    % Get times that the peaks occur.
    peaks.tripletTimes = peaks.centers(peaks.tripletPeakIndices);
    

% FLIP BACK SIGNAL FOR CANTILEVERS WITH FLOATERS
    if detSettings.expectingFloaters
        data.bandpass = -data.c1.bandPass;
        peaks.tripletHeights = -peaks.tripletHeights; 
    end
       
%% QUALITY CONTROL
% Check peak detection.
figure
plot(data.bandpass)
hold on
plot(peaks.tripletTimes, peaks.tripletHeights, 'r.', 'MarkerSize', 15)


% Check peak fitting (spot check).
peakData = peaks; % have to do this because there happens to be a MATLAB function called peaks... oops
c0Mask = checkPeaks(data, peakData, 'c0');

close all

peaks.acceptedTripletTimes = transpose(peaks.tripletTimes(c0Mask));

peaks.acceptedLeftAntinodes = peaks.leftAntinode(c0Mask);
peaks.acceptedRightAntinodes = peaks.rightAntinode(c0Mask);
peaks.acceptedPeakHeights = (peaks.acceptedLeftAntinodes + peaks.acceptedRightAntinodes)/2;
peaks.acceptedPeakHeightMismatch = peaks.acceptedRightAntinodes - peaks.acceptedLeftAntinodes;

peaks.acceptedLeftNodes = peaks.leftNode(c0Mask);
peaks.acceptedRightNodes = peaks.rightNode(c0Mask);
peaks.acceptedNodeDev = (peaks.acceptedLeftNodes + peaks.acceptedRightNodes)/2;
peaks.acceptedNormND = peaks.acceptedNodeDev ./ abs(peaks.acceptedPeakHeights);
peaks.acceptedNodeMismatch = peaks.acceptedRightNodes - peaks.acceptedLeftNodes; 

peaks.acceptedNodeDistance = peaks.node2Index(c0Mask) - peaks.node1Index(c0Mask);
peaks.acceptedBaselineSlope = peaks.baselineSlope(c0Mask);
peaks.acceptedTransitTime = 3*peaks.acceptedNodeDistance/detSettings.datarate*1000; % in ms

% Set up ability to reject peaks based on whatever criteria.
rejPeakNumbers = [];

% Plot stuff to check peak fit quality. 
figure

% Plot ND mismatch vs. mass. 
subplot(3, 2, 1)

NDWithinLimitMask = abs(peaks.acceptedNodeMismatch) < qcSettings.NDMismatchLimit;
NDWithinLimitIndices = find(NDWithinLimitMask);
NDOutsideLimitIndices = find(~NDWithinLimitMask);
nCellsWithinLimit = numel(find(NDWithinLimitMask));
plot(abs(peaks.acceptedPeakHeights(NDWithinLimitMask)), peaks.acceptedNodeMismatch(NDWithinLimitMask), ...
    'k.', 'MarkerSize', 12); hold on
plot(abs(peaks.acceptedPeakHeights(~NDWithinLimitMask)), peaks.acceptedNodeMismatch(~NDWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), qcSettings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qcSettings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
fprintf('%d/%d peaks have ND mismatch within threshold of %.2f Hz\n', nCellsWithinLimit, numel(peaks.acceptedPeakHeights), ...
    qcSettings.NDMismatchLimit);
xlabel('Buoyant mass (Hz)');
ylabel('Node mismatch (Hz)');
set(gca, 'YLim', [-1 1]);
rejPeakNumbers = unique([rejPeakNumbers, NDOutsideLimitIndices]);

% Plot antinode mismatch vs. mass. 
subplot(3, 2, 2)

antinodeWithinLimitMask = abs(peaks.acceptedPeakHeightMismatch) < qcSettings.antinodeMismatchLimit;
antinodeWithinLimitIndices = find(antinodeWithinLimitMask);
antinodeOutsideLimitIndices = find(~antinodeWithinLimitMask);

nCellsWithinAntinodeLimit = numel(find(antinodeWithinLimitMask));
plot(abs(peaks.acceptedPeakHeights(antinodeWithinLimitMask)), peaks.acceptedPeakHeightMismatch(antinodeWithinLimitMask), ...
    'k.', 'MarkerSize', 12); hold on
plot(abs(peaks.acceptedPeakHeights(~antinodeWithinLimitMask)), peaks.acceptedPeakHeightMismatch(~antinodeWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), qcSettings.antinodeMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qcSettings.antinodeMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
fprintf('%d/%d peaks have antinode mismatch within threshold of %.2f Hz\n', nCellsWithinAntinodeLimit, numel(peaks.acceptedPeakHeights), ...
    qcSettings.antinodeMismatchLimit);
xlabel('Buoyant mass (Hz)');
ylabel('Antinode mismatch (Hz)');
set(gca, 'YLim', [-1 1]);
rejPeakNumbers = unique([rejPeakNumbers, antinodeOutsideLimitIndices]);

% Plot ND mismatch vs. transit time 
subplot(3, 2, 3)

transitWithinLimitMask = peaks.acceptedTransitTime > qcSettings.minTransitTime; % both in ms
transitWithinLimitIndices = find(transitWithinLimitMask);
transitOutsideLimitIndices = find(~transitWithinLimitMask);
nCellsWithinTransitLimit = numel(find(transitWithinLimitMask));
fprintf('%d/%d peaks have transit times of at least %.0f ms\n', nCellsWithinTransitLimit, numel(peaks.acceptedPeakHeights), ...
    qcSettings.minTransitTime);

plot(peaks.acceptedTransitTime(transitWithinLimitMask), peaks.acceptedNodeMismatch(transitWithinLimitMask), ...
    'k.', 'MarkerSize', 12), hold on
plot(peaks.acceptedTransitTime(~transitWithinLimitMask), peaks.acceptedNodeMismatch(~transitWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
xlabel('Transit time (ms)');
ylabel('Node mismatch (Hz)');
plot(get(gca, 'XLim'), qcSettings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qcSettings.NDMismatchLimit*[1 1], 'k--');
plot(qcSettings.minTransitTime*[1 1], get(gca, 'YLim'), 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
set(gca, 'YLim', [-1 1]);
rejPeakNumbers = unique([rejPeakNumbers, transitOutsideLimitIndices]);

% Plot antinode mismatch vs. transit time
subplot(3, 2, 4)
plot(peaks.acceptedTransitTime(transitWithinLimitMask), peaks.acceptedPeakHeightMismatch(transitWithinLimitMask), ...
    'k.', 'MarkerSize', 12), hold on
plot(peaks.acceptedTransitTime(~transitWithinLimitMask), peaks.acceptedPeakHeightMismatch(~transitWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
xlabel('Transit time (ms)');
ylabel('Antinode mismatch (Hz)');
plot(get(gca, 'XLim'), qcSettings.antinodeMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qcSettings.antinodeMismatchLimit*[1 1], 'k--');
plot(qcSettings.minTransitTime*[1 1], get(gca, 'YLim'), 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
set(gca, 'YLim', [-1 1]);

    % Are these resolved by our bandwidth?
    avgTime = 3*mean(peaks.acceptedNodeDistance)/detSettings.datarate; % in sec
    bandwidthTime = 1/qcSettings.bandwidth; 
    fprintf('Mean peak duration %d ms (~%.0f times the bandwidth limit of %.0f ms)\n', floor(avgTime*1000), avgTime/bandwidthTime, bandwidthTime*1000), 

% After rejecting peaks, convert into a mask.
accPeaksMask = logical(ones(1, numel(peaks.acceptedPeakHeights)));
accPeaksMask(rejPeakNumbers) = false;
rejPeaksMask = ~accPeaksMask;

% (New figure) Plot ND mismatch versus baseline slope. 
subplot(3, 2, 5)
plot(peaks.acceptedBaselineSlope(accPeaksMask), peaks.acceptedNodeMismatch(accPeaksMask), 'k.', 'MarkerSize', 12);
hold on
plot(peaks.acceptedBaselineSlope(rejPeaksMask), peaks.acceptedNodeMismatch(rejPeaksMask), 'r.', 'MarkerSize', 12);
xlabel('Baseline slope'), ylabel('Node mismatch (Hz)');
plot(get(gca, 'XLim'), qcSettings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qcSettings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
plot([0 0], get(gca, 'YLim'), 'k');
set(gca, 'YLim', [-1 1]);

subplot(3, 2, 6)
plot(peaks.acceptedBaselineSlope(accPeaksMask), peaks.acceptedNodeDev(accPeaksMask), 'k.', 'MarkerSize', 12);
hold on
plot(peaks.acceptedBaselineSlope(rejPeaksMask), peaks.acceptedNodeDev(rejPeaksMask), 'r.', 'MarkerSize', 12);
xlabel('Baseline slope'), ylabel('ND (Hz)');
plot(get(gca, 'XLim'), [0 0], 'k');
plot([0 0], get(gca, 'YLim'), 'k');

% (New figure) plot ND and normalized ND vs. mass, for both accepted and rejected peaks.
figure
subplot(2, 2, 1)
plot(abs(peaks.acceptedPeakHeights(accPeaksMask)), peaks.acceptedNormND(accPeaksMask), 'k.', 'MarkerSize', 12), hold on
plot(abs(peaks.acceptedPeakHeights(rejPeaksMask)), peaks.acceptedNormND(rejPeaksMask), 'r.', 'MarkerSize', 12)
plot(get(gca, 'XLim'), [0 0], 'k');
xlabel('Buoyant mass (Hz)');
ylabel('Normalized ND');

subplot(2, 2, 2)
plot(abs(peaks.acceptedPeakHeights(accPeaksMask)), peaks.acceptedNodeDev(accPeaksMask), 'k.', 'MarkerSize', 12), hold on
plot(abs(peaks.acceptedPeakHeights(rejPeaksMask)), peaks.acceptedNodeDev(rejPeaksMask), 'r.', 'MarkerSize', 12)
plot(get(gca, 'XLim'), [0 0], 'k');
xlabel('Buoyant mass (Hz)');
ylabel('Node deviation (Hz)');

subplot(2, 2, 3)
myBoxPlot(peaks.acceptedNormND(accPeaksMask), 1, 'k'); % just accepted peaks
myBoxPlot(peaks.acceptedNormND, 2, 'r'); % all peaks
plot(get(gca, 'XLim'), [0 0], 'k--');
ylabel('Normalized ND');
set(gca, 'XTick', [1 2]);
set(gca, 'XTickLabel', {'Accepted peaks', 'All peaks'});

subplot(2, 2, 4)
myBoxPlot(peaks.acceptedNodeDev(accPeaksMask), 1, 'k'); % just accepted peaks
myBoxPlot(peaks.acceptedNodeDev, 2, 'r'); % all peaks
plot(get(gca, 'XLim'), [0 0], 'k--');
ylabel('ND (Hz)');
set(gca, 'XTick', [1 2]);
set(gca, 'XTickLabel', {'Accepted peaks', 'All peaks'});




  
%% SAVE DATA
    % write to output structure, measuredPeaks
measuredpeaks.peakIndex = transpose(peaks.acceptedTripletTimes(accPeaksMask));
measuredpeaks.leftAntinode = transpose(peaks.acceptedLeftAntinodes(accPeaksMask));
measuredpeaks.rightAntinode = transpose(peaks.acceptedRightAntinodes(accPeaksMask));
measuredpeaks.antinode = transpose(peaks.acceptedPeakHeights(accPeaksMask));
measuredpeaks.antinodeMismatch = transpose(peaks.acceptedPeakHeightMismatch(accPeaksMask));

measuredpeaks.leftNode = transpose(peaks.acceptedLeftNodes(accPeaksMask));
measuredpeaks.rightNode = transpose(peaks.acceptedRightNodes(accPeaksMask));
measuredpeaks.nodeDev = transpose(peaks.acceptedNodeDev(accPeaksMask));
measuredpeaks.nodeMismatch = transpose(peaks.acceptedNodeMismatch(accPeaksMask));

measuredpeaks.normalizedND = measuredpeaks.nodeDev ./ abs(measuredpeaks.antinode);
measuredpeaks.buoyantMass = -measuredpeaks.antinode;

%% Calculate ND offset for accepted cells. 
p = polyfit(measuredpeaks.buoyantMass, measuredpeaks.normalizedND, 1); % fit line. p = [slope, int]
measuredpeaks.NDoffset = measuredpeaks.normalizedND - p(1) * measuredpeaks.buoyantMass;

figure
subplot(2, 2, 1);
plot(measuredpeaks.buoyantMass, measuredpeaks.normalizedND, 'k.', 'MarkerSize', 12); hold on
xAxisLimits = [0, max(get(gca, 'XLim'))];
plot(xAxisLimits, [p(2), p(2) + p(1)*(xAxisLimits(2)-xAxisLimits(1))], 'r--');
plot(get(gca, 'XLim'), [0 0], 'k');
xlabel('Buoyant mass');
ylabel('Normalized ND');


subplot(2, 2, 2);
plot(measuredpeaks.buoyantMass, measuredpeaks.NDoffset, 'k.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), [0 0], 'k');
xlabel('Buoyant mass');
ylabel('ND offset');

subplot(2, 2, 3);
plot(measuredpeaks.nodeDev, measuredpeaks.NDoffset, 'k.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), [0 0], 'k');
plot([0 0], get(gca, 'YLim'), 'k');
xlabel('Node deviation');
ylabel('ND offset');


%% Export data
% output format: .mat file, 'measuredPeaks'
if input('Save data?');
    fprintf('Saving... \n')
    outputFilename = [directory, '\measuredPeaks.mat'];
    save(outputFilename, 'measuredPeaks');
    
    %secondary output format: xls file
    %columns: idx, peak time, peak height, for c2-c1-c0
%     spreadsheetFilename = [directory, '\measuredPeaks.xls'];
%      xlswrite(spreadsheetFilename, [measuredpeaks.peakIndex, measuredpeaks.leftAntinode, measuredpeaks.rightAntinode, ...
%         measuredpeaks.antinode, measuredpeaks.antinodeMismatch, measuredpeaks.leftNode, measuredpeaks.rightNode, ...
%         measuredpeaks.nodeDev, measuredpeaks.nodeMismatch]);

    csvFilename = [directory, '\measuredPeaks.csv'];
     csvwrite(csvFilename, [measuredpeaks.peakIndex, measuredpeaks.leftAntinode, measuredpeaks.rightAntinode, ...
        measuredpeaks.antinode, measuredpeaks.antinodeMismatch, measuredpeaks.leftNode, measuredpeaks.rightNode, ...
        measuredpeaks.nodeDev, measuredpeaks.nodeMismatch, measuredpeaks.normalizedND, measuredpeaks.buoyantMass, ...
        measuredpeaks.NDoffset]);
    

    % export figures
    saveOpenFigures('Peaks', directory);
    
    fprintf('Data saved. \n');
    
    % export parameters
    paramsFilename = [directory, '\parameters.csv'];
    fid = fopen(paramsFilename, 'w');
    fprintf(fid, '%.0f cells after approval + quality control\n', numel(measuredpeaks.nodeDev));
    fprintf(fid, 'Mean node deviation: %.2f\n', mean(measuredpeaks.nodeDev));
    fprintf(fid, 'Mean node mismatch: %.2f\n', mean(measuredpeaks.nodeMismatch));
    fprintf(fid, 'Mean absolute node mismatch: %.2f\n', mean(abs(measuredpeaks.nodeMismatch)));
    fprintf(fid, 'baselineDistance: %.0f\n', fitSettings.baselineDistance);
    fprintf(fid, 'baselineContinueDistance: %.0f\n', fitSettings.baselineContinueDistance);
    fprintf(fid, 'baselineSkipDistance: %.0f\n', fitSettings.baselineSkipDistance);
    fprintf(fid, 'datarate: %.0f\n', detSettings.datarate);
    fprintf(fid, 'bandwidth: %.0f\n', qcSettings.bandwidth);
    fclose(fid);
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%% 
function [averageTripletHeight, baselineRange, baselineSlope, baselineInt, leftAntinode, rightAntinode, ...
antinodeRange1, antinodeRange2, quadfit1, quadfit2, nodeRange1, nodeRange2, nodeFit1, nodeFit2, node1Index, node2Index, leftNode, rightNode, ...
tBaseline] = findAverageTripletHeight(peakNumber, peakCenters, frequencySignal, sidepointsFraction, settings)

derivThreshold = 0.05; 
sidePoints = round(sidepointsFraction*(peakCenters(peakNumber+1)-peakCenters(peakNumber)));

baselineSkipDistance = fitSettings.baselineSkipDistance; 
baselineContinueDistance = fitSettings.baselineContinueDistance;
baselineDistance = fitSettings.baselineDistance; 

leftDerivRange(1) = peakCenters(peakNumber-1) - baselineDistance;
leftDerivRange(2) = peakCenters(peakNumber-1);

rightDerivRange(1) = peakCenters(peakNumber+1);
rightDerivRange(2) = peakCenters(peakNumber+1) + baselineDistance;

% limit to length of signal
leftDerivRange(leftDerivRange < 1) = 1;
leftDerivRange(leftDerivRange > length(frequencySignal)) = length(frequencySignal);
rightDerivRange(rightDerivRange < 1) = 1;
rightDerivRange(rightDerivRange > length(frequencySignal)) = length(frequencySignal);

% filter frequency signal in each region, pick where it crosses slope
% threshold
leftDerivBaseline_filtered = filterSignal(frequencySignal(leftDerivRange(1) : leftDerivRange(2)));
rightDerivBaseline_filtered = filterSignal(frequencySignal(rightDerivRange(1) : rightDerivRange(2)));

% decide where the derivative crosses a threshold
leftDiff = diff(leftDerivBaseline_filtered);
rightDiff = diff(rightDerivBaseline_filtered);


leftThresholdIndex = leftDerivRange(1) + min(find(leftDiff < (-derivThreshold)));
rightThresholdIndex = rightDerivRange(1) + max(find(rightDiff > derivThreshold));

if isempty(leftThresholdIndex) | isempty(rightThresholdIndex)
    leftThresholdIndex = leftDerivRange(2);
    rightThresholdIndex = rightDerivRange(1);
end

% get the baseline range from these threshold crossings

leftBaselineRange(1) = leftThresholdIndex - (baselineSkipDistance + baselineContinueDistance);
leftBaselineRange(2) = leftThresholdIndex - baselineSkipDistance;
rightBaselineRange(1) = rightThresholdIndex + baselineSkipDistance;
rightBaselineRange(2) = rightThresholdIndex + (baselineSkipDistance + baselineContinueDistance);

tBaseline = [leftBaselineRange(1):leftBaselineRange(2), rightBaselineRange(1):rightBaselineRange(2)];
    
% make sure tBaseline stays within length of signal (for very early/late peaks)
tBaseline(tBaseline > length(frequencySignal)) = [];
tBaseline(tBaseline < 1) = [];  
baselineRegion = frequencySignal(tBaseline);
baselineRange = [min(tBaseline), max(tBaseline)]; % just for plotting/to export

% second attempt: fit a straight line to the baseline region
Y = baselineRegion;
Y = reshape(Y, length(Y), 1);
X = [ones(length(baselineRegion), 1), reshape(tBaseline, length(tBaseline), 1)];
B = X\Y;
baselineInt = B(1);
baselineSlope = B(2);
frequencySignalFit = baselineInt + baselineSlope * (1:length(frequencySignal));
frequencySignalFit = frequencySignalFit';
frequencySignal = frequencySignal - frequencySignalFit; 

% range of indices corresponding to the first antinode
leftBound1 = max([1, peakCenters(peakNumber-1) - sidePoints]);
rightBound1 = min([length(frequencySignal), peakCenters(peakNumber-1) + sidePoints]);

% range of indices corresponding to the second antinode
leftBound2 = max([1, peakCenters(peakNumber+1) - sidePoints]);
rightBound2 = min([length(frequencySignal),peakCenters(peakNumber+1) + sidePoints]); 
antinodeRegion1 = frequencySignal(leftBound1:rightBound1);
antinodeRegion2 = frequencySignal(leftBound2:rightBound2);

% set up regions to estimate node deviation
centerPeakIndex = peakCenters(peakNumber);
[~, node1Index] =  max(frequencySignal(rightBound1 : centerPeakIndex));
[~, node2Index] = max(frequencySignal(centerPeakIndex : leftBound2));

if isempty(node1Index) % hack 
    node1Index = 1;
end
if isempty(node1Index)
    node2Index = 1; 
end

node1Index = node1Index + rightBound1 - 1;
node2Index = node2Index + centerPeakIndex - 1;




nodeLeftBound1 = max([1, node1Index - sidePoints]);
nodeRightBound1 = min([length(frequencySignal), node1Index + sidePoints]);
nodeLeftBound2 = max([1, node2Index - sidePoints]);
nodeRightBound2 = min([length(frequencySignal), node2Index + sidePoints]);
nodeRegion1 = frequencySignal(nodeLeftBound1:nodeRightBound1);
nodeRegion2 = frequencySignal(nodeLeftBound2:nodeRightBound2);
        
    % fit a fourth order polynomial to each antinode. use an arbitrary time
    % vector t to index the antinode regions
    t1 = transpose(1:length(antinodeRegion1));
    t2 = transpose(1:length(antinodeRegion2));
    model1 = polyval(polyfit(t1, antinodeRegion1, 4), t1);
    model2 = polyval(polyfit(t2, antinodeRegion2, 4), t2);
    antinode1 = min(model1);
    antinode2 = min(model2);
    averageTripletHeight = (antinode1+antinode2)/2;

% fit fourth order polynomial to each node. use arbitrary time vector t to
% index the node regions.
t1 = transpose(1:length(nodeRegion1));
t2 = transpose(1:length(nodeRegion2));
nodeModel1 = polyval(polyfit(t1, nodeRegion1, 4), t1);
nodeModel2 = polyval(polyfit(t2, nodeRegion2, 4), t2);
node1 = max(nodeModel1);
node2 = max(nodeModel2);
nodes = [node1, node2];
averageNodeDev = (node1+node2)/2;

% to export: (add the baseline back to plot on the orig. signal). 
quadfit1 = transpose(model1 + (baselineInt + baselineSlope * transpose(leftBound1 : rightBound1)));
quadfit2 = transpose(model2 + (baselineInt + baselineSlope * transpose(leftBound2 : rightBound2)));
antinodeRange1 = leftBound1:rightBound1;
antinodeRange2 = leftBound2:rightBound2; 

nodeFit1 = transpose(nodeModel1 + (baselineInt + baselineSlope * transpose(nodeLeftBound1 : nodeRightBound1)));
nodeFit2 = transpose(nodeModel2 + (baselineInt + baselineSlope * transpose(nodeLeftBound2 : nodeRightBound2)));
nodeRange1 = nodeLeftBound1:nodeRightBound1;
nodeRange2 = nodeLeftBound2:nodeRightBound2;

leftAntinode = antinode1;
rightAntinode = antinode2;
leftNode = node1;
rightNode = node2; 
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function tripletPeakIndices = findTripletPeakIndices(peakCenters, peakHeights, detSettings)
    timeBetweenPeaks = diff(peakCenters);
        % this is indexed by the set of peaks, not the raw data!
        % these are really indices, not time 
    timeToPrev = [1e4; timeBetweenPeaks]; % set a long time before the first peak
    timeToNext = [timeBetweenPeaks; 1e4]; % set a long time after the last peak
        % the jth element of timeToNext gives the time after the jth peak
        % before the next peak is found. length(timeToNext = nPeaks
    prevHeight = [1e5; peakHeights(1:end-1)]; % first number doesn't matter - the first peak in the dataset can't be a peak center
    nextHeight = [peakHeights(2:end); 1e5]; % last number doesn't matter for the same reason 
    
    timeBeforePrev = [1e4; timeToPrev(1:end-1)]; 
    timeAfterNext = [timeToNext(2:end); 1e5];
    
    tripletPeakIndices = find(timeToPrev < detSettings.tripletTime & ...
                                                timeToNext < detSettings.tripletTime & ...
                                               timeBeforePrev > detSettings.gapTime & ...
                                               timeAfterNext > detSettings.gapTime & ...
                                               abs(prevHeight - nextHeight) < detSettings.maxHeightDiff);
end
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function frequencySignal = readFrequencySignal(filename)

fileID = fopen(filename, 'r', 'b');
frequencySignal = fread(fileID, 'uint32');
frequencySignal(1:129:end) = [];
frequencySignal = frequencySignal * (12.5e6/2^32); 
fclose(fileID);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function baseline = computeBaseline(signal, detSettings)

if detSettings.expectingFloaters
    detSettings.quantile = 100-detSettings.quantile; 
end

signal_downsampled = signal(1:detSettings.decimationFactor:end);
baseline_downsampled = runningPercentile(signal_downsampled, detSettings.quantileWidth, ...
    detSettings.quantile);
baseline = repelem(baseline_downsampled, detSettings.decimationFactor, 1);
baseline = baseline(:);
baseline = baseline(1:length(signal));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
function [minVal, minInd] = minimumWithinRange(i1, i2, x)
    % Returns the minimum value of the vector x between indices i1 and i2.
    % i1 and i2 can be vectors, as long as they are the same length. Also
    % returns the index corresponding to this value 
    
    minVal = zeros(length(i1), 1);
    minInd = zeros(length(i1), 1);
    
    for ind = 1:length(i1);
        [minVal(ind), minInd(ind)] = min(x(min([i1(ind), i2(ind)]) : max([i1(ind), i2(ind)])));
        minInd(ind) = minInd(ind) + min([i1(ind), i2(ind)]);
    end
    
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function y = runningPercentile(x, win, p, varargin)
%RUNNING_PERCENTILE Median or percentile over a moving window.
%   Y = RUNNING_PERCENTILE(X,WIN,P) returns percentiles of the values in 
%   the vector Y over a moving window of size win, where p and win are
%   scalars. p = 0 gives the rolling minimum, and p = 100 the maximum.
%
%   running_percentile(X,WIN,P,THRESH) includes a threshold where NaN will be
%   returned in areas where the number of NaN values exceeds THRESH. If no
%   value is specified the default is the window size / 2.
%
%   The definition used is the same as for MATLAB's prctile: if the
%   window is too small for a given percentile, the max or min will be
%   returned. Otherwise linear interpolation is used on the sorted data.
%   Sorting is retained while updating the window, making this faster than
%   repeatedly calling prctile. The edges are handled by duplicating nearby
%   data.
%
%   Examples:
%      y = running_percentile(x,500,10); % 10th percentile of x with
%      a 500 point moving window

%   Author: Levi Golston, 2014


% Check inputs
N = length(x);
if win > N || win < 1
    error('window size must be <= size of X and >= 1')
end
if length(win) > 1
    error('win must be a scalar')
end
if p < 0 || p > 100
    error('percentile must be between 0 and 100')
end
if ceil(win) ~= floor(win)
    error('window size must be a whole number')
end
if ~isvector(x)
    error('x must be a vector')
end

if nargin == 4
    NaN_threshold = varargin{1};
else
    NaN_threshold = floor(win/2);
end

% pad edges with data and sort first window
if iscolumn(x)
    x = [x(ceil(win/2)-1 : -1 : 1); x; x(N : -1 : N-floor(win/2)+1); NaN];
else
    x = [x(ceil(win/2)-1 : -1 : 1), x, x(N : -1 : N-floor(win/2)+1), NaN];
end
tmp = sort(x(1:win));
y = NaN(N,1);

offset = length(ceil(win/2)-1 : -1 : 1) + floor(win/2);
numnans = sum(isnan(tmp));

% loop
for i = 1:N
	% Percentile levels given by equation: 100/win*((1:win) - 0.5);
	% solve for desired index
	pt = p*(win-numnans)/100 + 0.5;
    if numnans > NaN_threshold;   % do nothing
    elseif pt < 1        % not enough points: return min
		y(i) = tmp(1);
	elseif pt > win-numnans     % not enough points: return max
		y(i) = tmp(win - numnans);
	elseif floor(pt) == ceil(pt);  % exact match found
		y(i) = tmp(pt);
	else             % linear interpolation
		pt = floor(pt);
		x0 = 100*(pt-0.5)/(win - numnans);
		x1 = 100*(pt+0.5)/(win - numnans);
		xfactor = (p - x0)/(x1 - x0);
		y(i) = tmp(pt) + (tmp(pt+1) - tmp(pt))*xfactor;
    end
    
	% find index of oldest value in window
	if isnan(x(i))
		ix = win;  						  % NaN are stored at end of list
		numnans = numnans - 1;
	else
		ix = find(tmp == x(i),1,'first');
	end
	
	% replace with next item in data
	newnum = x(offset + i + 1);
	tmp(ix) = newnum;
	if isnan(newnum)
		numnans = numnans + 1;
	end
	tmp = sort(tmp);

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function plotSinglePeak(data, peaks, cantilever, peakNumber)

buffer = 200; % extra points to plot on either side
peakTimeRange = (peaks.baselineRange{peakNumber}(1) - buffer) : (peaks.baselineRange{peakNumber}(2) + buffer);
peakTimeRange(peakTimeRange > length(data.frequencySignal)) = [];
peakTimeRange(peakTimeRange < 1) = []; 
set(gca, 'XLim', [peakTimeRange(1), peakTimeRange(end)])
hold on

% plot the *raw* data corresponding to this peak

plot(peakTimeRange, data.frequencySignal(peakTimeRange), 'k.');

% plot the baseline fit
baselineY = peaks.baselineInt(peakNumber) + peaks.baselineSlope(peakNumber) * peakTimeRange;
plot(peakTimeRange, baselineY, 'b--', 'LineWidth', 1.5);

% plot the raw data to which the baseline was fit in a different color
tBaseline = peaks.tBaseline{peakNumber};
plot(tBaseline, data.frequencySignal(tBaseline), 'g.');

% plot the quadratic fit to each antinode
antinodeRange1 = peaks.antinodeRange1{peakNumber};
antinodeRange2 = peaks.antinodeRange2{peakNumber};
quadfit1 = peaks.quadfit1{peakNumber};
quadfit2 = peaks.quadfit2{peakNumber};

plot(antinodeRange1, quadfit1, 'g', 'LineWidth', 2);
plot(antinodeRange2, quadfit2, 'g', 'LineWidth', 2);

% plot the PEAKS (baseline + peak height)
    antinode1Index = peaks.centers(peaks.tripletPeakIndices(peakNumber) - 1);
    antinode2Index = peaks.centers(peaks.tripletPeakIndices(peakNumber) + 1);
    antinode1Baseline = baselineY(antinode1Index - peakTimeRange(1));
    antinode2Baseline = baselineY(antinode2Index - peakTimeRange(1));
    plot([antinode1Index, antinode2Index], [antinode1Baseline + peaks.leftAntinode(peakNumber), ...
        antinode2Baseline + peaks.rightAntinode(peakNumber)], 'r.', 'MarkerSize', 15);


% plot the quadratic fit to each node
nodeRange1 = peaks.nodeRange1{peakNumber};
nodeRange2 = peaks.nodeRange2{peakNumber};
nodeFit1 = peaks.nodeFit1{peakNumber};
nodeFit2 = peaks.nodeFit2{peakNumber};
plot(nodeRange1, nodeFit1, 'g', 'LineWidth', 2);
plot(nodeRange2, nodeFit2, 'g', 'LineWidth', 2);

% plot the node locations
node1Index = peaks.node1Index(peakNumber);
node2Index = peaks.node2Index(peakNumber);
node1Baseline = baselineY(node1Index - peakTimeRange(1));
node2Baseline = baselineY(node2Index - peakTimeRange(2));
plot([node1Index, node2Index], [node1Baseline + peaks.leftNode(peakNumber), ...
    node2Baseline + peaks.rightNode(peakNumber)], 'r.', 'MarkerSize', 15);

    set(gca, 'XLim', [peakTimeRange(1), peakTimeRange(end)])
hold off

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 function mask = checkPeaks(data, peakData, cantilever)
peaksChecked = 0;
rejectPeakNumbers = [];
peakCheckSize = [5 5];

           while peaksChecked < length(peakData.tripletHeights)
                firstPeak = peaksChecked + 1;
                lastPeak = min(length(peakData.tripletHeights), peaksChecked + peakCheckSize(1)*peakCheckSize(2));
                figure

            % plot up to 64 peaks
                for peakNumber = firstPeak : lastPeak; % 1:64, or 65: 128, ...
                      subplot(peakCheckSize(1), peakCheckSize(2), peakNumber-peaksChecked);
                        hold on
                        plotSinglePeak(data, peakData, 'c0', peakNumber)
                        set(gca, 'xticklabel', {})
%                         set(gca, 'yticklabel', {})
                        title(num2str(peakNumber))
                end

                % ask for a vector of the "bad" peaks 
                inputString = input('Peak numbers to reject (enter as row vector) or type ''skip'' to approve all: ');
                if strcmp(inputString, 'skip')
                    break
                else
                    rejectPeakNumbers = [rejectPeakNumbers, input('Peak numbers to reject (enter as row vector): ')];
                    peaksChecked = peaksChecked + (lastPeak - firstPeak + 1);
                  end
            end % end while loop

            mask = 1:length(peakData.tripletHeights);
            mask(rejectPeakNumbers) = [];
end % end checkPeaks

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yf = filterSignal(y)
yf = sgolayfilt(y, 3, 51);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function peakFit = getPeakFit(tripletPeakIndex, peaks, data, fitSettings) % structure w/ all peak fit data

% Get basic useful numbers
peakFit.leftPeakIndex = peaks.centers(tripletPeakIndex-1);
peakFit.centerPeakIndex = peaks.centers(tripletPeakIndex);
peakFit.rightPeakIndex = peaks.centers(tripletPeakIndex+1);

antinodeSeparation = peakFit.rightPeakIndex - peakFit.leftPeakIndex; 

peakFit.leftBaselineStart = peakFit.leftPeakIndex ...
    - round(fitSettings.baselineStartFactor * antinodeSeparation) ...
    - round(fitSettings.baselineContinueFactor * antinodeSeparation);
peakFit.leftBaselineEnd = peakFit.leftPeakIndex ...
    - round(fitSettings.baselineStartFactor * antinodeSeparation);
peakFit.rightBaselineStart = peakFit.rightPeakIndex ...
    + round(fitSettings.baselineStartFactor * antinodeSeparation);
peakFit.rightBaselineEnd = peakFit.rightBaselineStart ...
    + round(fitSettings.baselineContinueFactor * antinodeSeparation);

% Pull out the lowpass-filtered data corresponding to this peak
peakFit.baselineIndices = transpose(...
    [peakFit.leftBaselineStart : peakFit.leftBaselineEnd, ...
     peakFit.rightBaselineStart : peakFit.rightBaselineEnd]);
peakFit.baselineLowpassSignal = data.lowpass(peakFit.baselineIndices);

peakFit.peakIndices = transpose(...
    [(peakFit.leftBaselineEnd+1) : (peakFit.rightBaselineStart-1)]);
peakFit.peakLowpassSignal = data.lowpass(peakFit.peakIndices);

peakFit.totalIndices = transpose(peakFit.leftBaselineStart : peakFit.rightBaselineEnd);
peakFit.totalSignal = data.lowpass(peakFit.totalIndices);

% Fit a curve to the baseline, evaluate over entire range 
[p, ~, mu] = polyfit(peakFit.baselineIndices, ...
    peakFit.baselineLowpassSignal, fitSettings.baselineFitOrder);
peakFit.baselineFit = polyval(p, (peakFit.totalIndices-mu(1))/mu(2));

% Subtract baseline from signal
peakFit.baselineSubtractedSignal = peakFit.totalSignal - peakFit.baselineFit;     

% Get indices corresponding to antinodes
peakFit.leftAntinodeStart = peakFit.leftPeakIndex ...
    - round(fitSettings.peakFitFactor/2 * antinodeSeparation);
peakFit.leftAntinodeEnd = peakFit.leftPeakIndex ...
    + round(fitSettings.peakFitFactor/2 * antinodeSeparation);
peakFit.rightAntinodeStart = peakFit.rightPeakIndex ...
    - round(fitSettings.peakFitFactor/2 * antinodeSeparation);
peakFit.rightAntinodeEnd = peakFit.rightPeakIndex ...
    + round(fitSettings.peakFitFactor/2 * antinodeSeparation);
peakFit.leftAntinodeAbsoluteRange = transpose(peakFit.leftAntinodeStart : ...
    peakFit.leftAntinodeEnd);
peakFit.rightAntinodeAbsoluteRange = transpose(peakFit.rightAntinodeStart : ...
    peakFit.rightAntinodeEnd); 
peakFit.leftAntinodeRelativeRange = peakFit.leftAntinodeAbsoluteRange ...
    - peakFit.leftBaselineStart;
peakFit.rightAntinodeRelativeRange = peakFit.rightAntinodeAbsoluteRange ...
    - peakFit.leftBaselineStart; 

% Fit polynomial to antinodes
peakFit.leftAntinodeSignal = peakFit.baselineSubtractedSignal(peakFit.leftAntinodeRelativeRange);
peakFit.rightAntinodeSignal = peakFit.baselineSubtractedSignal(peakFit.rightAntinodeRelativeRange);
t1 = transpose(1:length(peakFit.leftAntinodeSignal));
t2 = transpose(1:length(peakFit.rightAntinodeSignal));
peakFit.leftAntinodeFit = polyval(polyfit(t1, peakFit.leftAntinodeSignal, fitSettings.peakFitOrder), t1);
peakFit.rightAntinodeFit = polyval(polyfit(t2, peakFit.rightAntinodeSignal, fitSettings.peakFitOrder), t2);
[peakFit.leftAntinodeValue, leftAntinodeRelativePosition] = min(peakFit.leftAntinodeFit);
[peakFit.rightAntinodeValue, rightAntinodeRelativePosition] = min(peakFit.rightAntinodeFit);

% Get approximate node indices by finding maximum of frequency signal 
[~, tempLeftNodeIndex] = max(data.lowpass(...
    peakFit.leftPeakIndex : peakFit.centerPeakIndex));
[~, tempRightNodeIndex] = max(data.lowpass(...
    peakFit.centerPeakIndex : peakFit.rightPeakIndex)); 

peakFit.leftNodeIndex = peakFit.leftPeakIndex + tempLeftNodeIndex;
peakFit.rightNodeIndex = peakFit.centerPeakIndex + tempRightNodeIndex; 

% Get the signal corresponding to each node
fitWindowSize = 2*round(fitSettings.peakFitFactor * antinodeSeparation/2);
peakFit.leftNodeRelativeRange = transpose(((peakFit.leftNodeIndex - fitWindowSize/2) ...
    : (peakFit.leftNodeIndex + fitWindowSize/2)) - peakFit.leftBaselineStart); 
peakFit.rightNodeRelativeRange = transpose(((peakFit.rightNodeIndex - fitWindowSize/2) ...
    : (peakFit.rightNodeIndex + fitWindowSize/2)) - peakFit.leftBaselineStart);
peakFit.leftNodeAbsoluteRange = peakFit.leftNodeRelativeRange + ...
    peakFit.leftBaselineStart;
peakFit.rightNodeAbsoluteRange = peakFit.rightNodeRelativeRange + ...
    peakFit.leftBaselineStart; 
peakFit.leftNodeSignal = peakFit.baselineSubtractedSignal(peakFit.leftNodeRelativeRange);
peakFit.rightNodeSignal = peakFit.baselineSubtractedSignal(peakFit.rightNodeRelativeRange);

% Fit polynomial to each node, using an arbitrary time vector (t) to
% index the node regions.
t1 = transpose(1:length(peakFit.leftNodeSignal));
t2 = transpose(1:length(peakFit.rightNodeSignal));
peakFit.leftNodeFit = polyval(polyfit(t1, peakFit.leftNodeSignal, fitSettings.peakFitOrder), t1);
peakFit.rightNodeFit = polyval(polyfit(t2, peakFit.rightNodeSignal, fitSettings.peakFitOrder), t2);
[peakFit.leftNodeValue, leftNodeRelativePosition] = max(peakFit.leftNodeFit);
[peakFit.rightNodeValue, rightNodeRelativePosition] = max(peakFit.rightNodeFit);


% Get other data segments that are just useful for plotting
peakFit.leftAntinodeBaselineSignal = peakFit.baselineFit(peakFit.leftAntinodeRelativeRange);
peakFit.rightAntinodeBaselineSignal = peakFit.baselineFit(peakFit.rightAntinodeRelativeRange);
peakFit.leftNodeBaselineSignal = peakFit.baselineFit(peakFit.leftNodeRelativeRange);
peakFit.rightNodeBaselineSignal = peakFit.baselineFit(peakFit.rightNodeRelativeRange);

peakFit.leftAntinodeAbsolutePosition = peakFit.leftAntinodeAbsoluteRange(1) + leftAntinodeRelativePosition - 1; 
peakFit.rightAntinodeAbsolutePosition = peakFit.rightAntinodeAbsoluteRange(1) + rightAntinodeRelativePosition - 1; 
peakFit.leftNodeAbsolutePosition = peakFit.leftNodeAbsoluteRange(1) + leftNodeRelativePosition - 1; 
peakFit.rightNodeAbsolutePosition = peakFit.rightNodeAbsoluteRange(1) + rightNodeRelativePosition - 1; 
end

function plotPeakFit(peakFit)
plot(peakFit.totalIndices, peakFit.totalSignal, 'k.');
hold on
plot(peakFit.baselineIndices, peakFit.baselineLowpassSignal, 'b.');
plot(peakFit.totalIndices, peakFit.baselineFit, 'b--', 'LineWidth', 2);

% plot node + antinode fits. first, get baseline values in each region
plot(peakFit.leftAntinodeAbsoluteRange, peakFit.leftAntinodeBaselineSignal+peakFit.leftAntinodeSignal, 'g', 'LineWidth', 2)
plot(peakFit.rightAntinodeAbsoluteRange, peakFit.rightAntinodeBaselineSignal+peakFit.rightAntinodeSignal, 'g', 'LineWidth', 2);
plot(peakFit.leftNodeAbsoluteRange, peakFit.leftNodeBaselineSignal+peakFit.leftNodeSignal, 'g', 'LineWidth', 2);
plot(peakFit.rightNodeAbsoluteRange, peakFit.rightNodeBaselineSignal+peakFit.rightNodeSignal, 'g', 'LineWidth', 2);

% plot fits to nodes + antinodes
plot(peakFit.leftNodeAbsolutePosition , max(peakFit.leftNodeBaselineSignal+peakFit.leftNodeSignal), 'r.', 'MarkerSize', 15);
plot(peakFit.rightNodeAbsolutePosition , max(peakFit.rightNodeBaselineSignal+peakFit.rightNodeSignal), 'r.', 'MarkerSize', 15);
plot(peakFit.leftAntinodeAbsolutePosition , min(peakFit.leftAntinodeBaselineSignal+peakFit.leftAntinodeSignal), 'r.', 'MarkerSize', 15);
plot(peakFit.rightAntinodeAbsolutePosition , min(peakFit.rightAntinodeBaselineSignal+peakFit.rightAntinodeSignal), 'r.', 'MarkerSize', 15);
set(gca, 'XLim', [min(peakFit.totalIndices), max(peakFit.totalIndices)]);
end