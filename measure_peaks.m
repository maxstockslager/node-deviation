function measurePeaks

clear all, close all
addpath('helper-functions');

%%
% Peak detection settings (little effect on fit) 
detection_settings = struct(...
    'quantileWidth', 100, ...
    'quantile', 75, ...
    'sgOrder', 3, ...
    'sgWidth', 7, ...
    'decimationFactor', 10, ...
    'datarate', 1000, ...
    'detectionThreshold', -4, ...
    'tripletTime', 1600, ...
    'gapTime', 1200, ...
    'maxHeightDiff', 3 ...
);

fit_settings = struct(...
    'baselineStartFactor', 0.35, ... % distance from each antinode (as multiple of the separation between antinodes) where the baseline starts
    'baselineContinueFactor', 1, ... % distance (as multiple of the separation between antinodes) that the baseline extends on either side
    'baselineFitOrder', 1, ... % 0 for constant, 1 for linear, 2 for quadratic, etc
    'peakFitOrder', 4, ... % for fitting antinodes + nodes. best if even
    'peakFitFactor', 0.1 ... % controls how much signal we fit on either side of the nodes and antinodes
); 
 

% Peak approval/QC settings 
qc_settings = struct(...
    'antinodeMismatchLimit', 0.5, ... % Hz
    'NDMismatchLimit', 0.50, ... % Hz
    'minTransitTime', 150, ... % ms
    'bandwidth', 200, ...
    'largeWindowSize', 2e6, ...
    'smallWindowSize', 1200 ...
);

% LOAD IN DATA
% directory = uigetdir;
directory = 'C:\Users\Max\Documents\Program Data\MATLAB\ND\ND test';
data.frequencySignal = readFrequencySignal([directory, '\c00.bin']);
data.frequencyTime = transpose(1:length(data.frequencySignal)) / detection_settings.datarate;

% FILTERING
data.lowpass = sgolayfilt(data.frequencySignal, detection_settings.sgOrder, detection_settings.sgWidth);
data.baseline = computeBaseline(data.lowpass, detection_settings);
data.bandpass = data.lowpass - data.baseline; 


% PEAK DETECTION
% Find continuous segments of data below the threshold - these are how
% we define "peaks". 
peaks.indicesBelowThreshold = find(data.bandpass < detection_settings.detectionThreshold);
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

peaks.tripletPeakIndices = findTripletPeakIndices(peaks.centers, peaks.heights, detection_settings);
peakFits = [];
for peakNumber = 1:length(peaks.tripletPeakIndices) % for each node peak 
   
    currentPeakFit = getPeakFit(peaks.tripletPeakIndices(peakNumber), ...
        peaks, data, fit_settings, detection_settings); % structure w/ all peak fit data
    if isempty(peakFits)
        peakFits = currentPeakFit;
    else
        peakFits = [peakFits, currentPeakFit];
    end
end

%% Extract summary data from peakfits
summary = struct(...
    'peakHeight', [peakFits.peakHeight], ...
    'nodeDev', [peakFits.nodeDev], ...
    'nodeMismatch', [peakFits.nodeMismatch], ...
    'antinodeMismatch', [peakFits.antinodeMismatch], ...
    'transitTime', [peakFits.transitTime] ...
);

       
%% QUALITY CONTROL
% Check peak detection.
figure
plot(data.bandpass)
hold on
plot([peakFits.centerPeakIndex], data.bandpass([peakFits.centerPeakIndex]), 'r.', 'MarkerSize', 15)

figure
subplot_dimensions = [6 8];
for ii = 1:(subplot_dimensions(1)*subplot_dimensions(2))
   subplot(subplot_dimensions(1), subplot_dimensions(2), ii)
   peak_number = randi([1, numel(peakFits)], 1);
   plotPeakFit(peakFits(peak_number));
end



%%
% Set up ability to reject peaks based on whatever criteria.
rejPeakNumbers = [];

% Plot stuff to check peak fit quality. 
figure

% Plot ND mismatch vs. mass. 
subplot(3, 2, 1)

NDWithinLimitMask = abs(summary.nodeMismatch) < qc_settings.NDMismatchLimit;
NDOutsideLimitIndices = find(~NDWithinLimitMask);
nCellsWithinLimit = numel(find(NDWithinLimitMask));
plot(abs(summary.peakHeight(NDWithinLimitMask)), summary.nodeMismatch(NDWithinLimitMask), ...
    'k.', 'MarkerSize', 12); hold on
plot(abs(summary.peakHeight(~NDWithinLimitMask)), summary.nodeMismatch(~NDWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), qc_settings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qc_settings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
fprintf('%d/%d peaks have ND mismatch within threshold of %.2f Hz\n', nCellsWithinLimit, numel(summary.peakHeight), ...
    qc_settings.NDMismatchLimit);
xlabel('Buoyant mass (Hz)');
ylabel('Node mismatch (Hz)');
set(gca, 'YLim', [-1 1]);
rejPeakNumbers = unique([rejPeakNumbers, NDOutsideLimitIndices]);

%%
% Plot antinode mismatch vs. mass. 
subplot(3, 2, 2)

antinodeWithinLimitMask = abs(summary.antinodeMismatch) < qc_settings.antinodeMismatchLimit;
antinodeOutsideLimitIndices = find(~antinodeWithinLimitMask);

nCellsWithinAntinodeLimit = numel(find(antinodeWithinLimitMask));
plot(abs(summary.peakHeight(antinodeWithinLimitMask)), summary.antinodeMismatch(antinodeWithinLimitMask), ...
    'k.', 'MarkerSize', 12); hold on
plot(abs(summary.peakHeight(~antinodeWithinLimitMask)),summary.antinodeMismatch(~antinodeWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), qc_settings.antinodeMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qc_settings.antinodeMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
fprintf('%d/%d peaks have antinode mismatch within threshold of %.2f Hz\n', nCellsWithinAntinodeLimit, numel(peakFits), ...
    qc_settings.antinodeMismatchLimit);
xlabel('Buoyant mass (Hz)');
ylabel('Antinode mismatch (Hz)');
set(gca, 'YLim', [-1 1]);
rejPeakNumbers = unique([rejPeakNumbers, antinodeOutsideLimitIndices]);

% Plot ND mismatch vs. transit time 
subplot(3, 2, 3)

transitWithinLimitMask = 1000*summary.transitTime > qc_settings.minTransitTime; % both in ms
transitOutsideLimitIndices = find(~transitWithinLimitMask);
nCellsWithinTransitLimit = numel(find(transitWithinLimitMask));
fprintf('%d/%d peaks have transit times of at least %.0f ms\n', nCellsWithinTransitLimit, numel(peakFits), ...
    qc_settings.minTransitTime);

plot(1000*summary.transitTime(transitWithinLimitMask), summary.nodeMismatch(transitWithinLimitMask), ...
    'k.', 'MarkerSize', 12), hold on
plot(1000*summary.transitTime(~transitWithinLimitMask), summary.nodeMismatch(~transitWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
xlabel('Transit time (ms)');
ylabel('Node mismatch (Hz)');
plot(get(gca, 'XLim'), qc_settings.NDMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qc_settings.NDMismatchLimit*[1 1], 'k--');
plot(qc_settings.minTransitTime*[1 1], get(gca, 'YLim'), 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
set(gca, 'YLim', [-1 1]);
rejPeakNumbers = unique([rejPeakNumbers, transitOutsideLimitIndices]);

% Plot antinode mismatch vs. transit time
subplot(3, 2, 4)
plot(1000*summary.transitTime(transitWithinLimitMask), summary.antinodeMismatch(transitWithinLimitMask), ...
    'k.', 'MarkerSize', 12), hold on
plot(1000*summary.transitTime(~transitWithinLimitMask), summary.antinodeMismatch(~transitWithinLimitMask), ...
    'r.', 'MarkerSize', 12); hold on
xlabel('Transit time (ms)');
ylabel('Antinode mismatch (Hz)');
plot(get(gca, 'XLim'), qc_settings.antinodeMismatchLimit*[1 1], 'k--');
plot(get(gca, 'XLim'), -qc_settings.antinodeMismatchLimit*[1 1], 'k--');
plot(qc_settings.minTransitTime*[1 1], get(gca, 'YLim'), 'k--');
plot(get(gca, 'XLim'), [0 0], 'k');
set(gca, 'YLim', [-1 1]);

    % Are these resolved by our bandwidth?
    avgTime = mean(summary.transitTime); % in sec
    bandwidthTime = 1/qc_settings.bandwidth; 
    fprintf('Mean peak duration %d ms (~%.0f times the bandwidth limit of %.0f ms)\n', floor(avgTime*1000), avgTime/bandwidthTime, bandwidthTime*1000), 

% After rejecting peaks, convert into a mask.
accPeaksMask = logical(ones(1, numel(peakFits)));
accPeaksMask(rejPeakNumbers) = false;
rejPeaksMask = ~accPeaksMask;

%%
peakFits_approved = peakFits(accPeaksMask);
summary_approved = struct(...
    'peakHeight', [peakFits_approved.peakHeight]', ...
    'nodeDev', [peakFits_approved.nodeDev]', ...
    'nodeMismatch', [peakFits_approved.nodeMismatch]', ...
    'antinodeMismatch', [peakFits_approved.antinodeMismatch]', ...
    'transitTime', [peakFits_approved.transitTime]' ...
);

%%   
output_filename = [directory, '\peaks.xlsx'];
fprintf('writing to %s\n', output_filename);
writetable(struct2table(summary_approved), output_filename);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function tripletPeakIndices = findTripletPeakIndices(peakCenters, peakHeights, detection_settings)
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
    
    tripletPeakIndices = find(timeToPrev < detection_settings.tripletTime & ...
                                                timeToNext < detection_settings.tripletTime & ...
                                               timeBeforePrev > detection_settings.gapTime & ...
                                               timeAfterNext > detection_settings.gapTime & ...
                                               abs(prevHeight - nextHeight) < detection_settings.maxHeightDiff);
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
function baseline = computeBaseline(signal, detection_settings)

signal_downsampled = signal(1:detection_settings.decimationFactor:end);
baseline_downsampled = runningPercentile(signal_downsampled, detection_settings.quantileWidth, ...
    detection_settings.quantile);
baseline = repelem(baseline_downsampled, detection_settings.decimationFactor, 1);
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
function peakFit = getPeakFit(tripletPeakIndex, peaks, data, fit_settings, detection_settings) % structure w/ all peak fit data

% Get basic useful numbers
peakFit.leftPeakIndex = peaks.centers(tripletPeakIndex-1);
peakFit.centerPeakIndex = peaks.centers(tripletPeakIndex);
peakFit.rightPeakIndex = peaks.centers(tripletPeakIndex+1);

antinodeSeparation = peakFit.rightPeakIndex - peakFit.leftPeakIndex; 

peakFit.leftBaselineStart = peakFit.leftPeakIndex ...
    - round(fit_settings.baselineStartFactor * antinodeSeparation) ...
    - round(fit_settings.baselineContinueFactor * antinodeSeparation);
peakFit.leftBaselineEnd = peakFit.leftPeakIndex ...
    - round(fit_settings.baselineStartFactor * antinodeSeparation);
peakFit.rightBaselineStart = peakFit.rightPeakIndex ...
    + round(fit_settings.baselineStartFactor * antinodeSeparation);
peakFit.rightBaselineEnd = peakFit.rightBaselineStart ...
    + round(fit_settings.baselineContinueFactor * antinodeSeparation);

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
    peakFit.baselineLowpassSignal, fit_settings.baselineFitOrder);
peakFit.baselineFit = polyval(p, (peakFit.totalIndices-mu(1))/mu(2));

% Subtract baseline from signal
peakFit.baselineSubtractedSignal = peakFit.totalSignal - peakFit.baselineFit;     

% Get indices corresponding to antinodes
peakFit.leftAntinodeStart = peakFit.leftPeakIndex ...
    - round(fit_settings.peakFitFactor/2 * antinodeSeparation);
peakFit.leftAntinodeEnd = peakFit.leftPeakIndex ...
    + round(fit_settings.peakFitFactor/2 * antinodeSeparation);
peakFit.rightAntinodeStart = peakFit.rightPeakIndex ...
    - round(fit_settings.peakFitFactor/2 * antinodeSeparation);
peakFit.rightAntinodeEnd = peakFit.rightPeakIndex ...
    + round(fit_settings.peakFitFactor/2 * antinodeSeparation);
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
peakFit.leftAntinodeFit = polyval(polyfit(t1, peakFit.leftAntinodeSignal, fit_settings.peakFitOrder), t1);
peakFit.rightAntinodeFit = polyval(polyfit(t2, peakFit.rightAntinodeSignal, fit_settings.peakFitOrder), t2);
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
fitWindowSize = 2*round(fit_settings.peakFitFactor * antinodeSeparation/2);
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
peakFit.leftNodeFit = polyval(polyfit(t1, peakFit.leftNodeSignal, fit_settings.peakFitOrder), t1);
peakFit.rightNodeFit = polyval(polyfit(t2, peakFit.rightNodeSignal, fit_settings.peakFitOrder), t2);
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

% summarize
peakFit.peakHeight = 0.5*(peakFit.leftAntinodeValue+peakFit.rightAntinodeValue);
peakFit.nodeDev = 0.5*(peakFit.leftNodeValue+peakFit.rightNodeValue);
peakFit.nodeMismatch = peakFit.rightNodeValue - peakFit.leftNodeValue;
peakFit.antinodeMismatch = peakFit.rightAntinodeValue - peakFit.leftAntinodeValue;
peakFit.transitTime = 2*(peakFit.rightPeakIndex - peakFit.leftPeakIndex)/detection_settings.datarate;
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