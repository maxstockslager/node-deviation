function NDcalculations

% This function loads in a measuredPeaks file, performs calculations, and
% exports a new spreadsheet with the calculations appended. 

clear all, close all
directory = uigetdir;
inputFilename = strcat(directory, '\measuredPeaks.mat');
load(inputFilename);



%% Calculate ND offset for accepted cells. 
p = polyfit(measuredPeaks.c0.buoyantMass, measuredPeaks.c0.normalizedND, 1); % fit line. p = [slope, int]
measuredPeaks.c0.NDoffset = measuredPeaks.c0.normalizedND - p(1) * measuredPeaks.c0.buoyantMass;

figure
subplot(2, 2, 1);
plot(measuredPeaks.c0.buoyantMass, measuredPeaks.c0.normalizedND, 'k.', 'MarkerSize', 12); hold on
xAxisLimits = [0, max(get(gca, 'XLim'))];
plot(xAxisLimits, [p(2), p(2) + p(1)*(xAxisLimits(2)-xAxisLimits(1))], 'r--');
plot(get(gca, 'XLim'), [0 0], 'k');
xlabel('Buoyant mass');
ylabel('Normalized ND');


subplot(2, 2, 2);
plot(measuredPeaks.c0.buoyantMass, measuredPeaks.c0.NDoffset, 'k.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), [0 0], 'k');
xlabel('Buoyant mass');
ylabel('ND offset');

subplot(2, 2, 3);
plot(measuredPeaks.c0.nodeDev, measuredPeaks.c0.NDoffset, 'k.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), [0 0], 'k');
plot([0 0], get(gca, 'YLim'), 'k');
xlabel('Node deviation');
ylabel('ND offset');

subplot(2, 2, 4)
plot(measuredPeaks.c0.normalizedND, measuredPeaks.c0.NDoffset, 'k.', 'MarkerSize', 12); hold on
plot(get(gca, 'XLim'), [0 0], 'k');
plot([0 0], get(gca, 'YLim'), 'k');
xlabel('Normalized ND');
ylabel('ND offset');


%% Export data
fprintf('Saving...\n');
outputFilename = strcat(directory, '\measuredPeaks_2.mat');
save(outputFilename, 'measuredPeaks');

csvFilename = [directory, '\measuredPeaks_2.csv'];
 csvwrite(csvFilename, [measuredPeaks.c0.peakIndex, measuredPeaks.c0.leftAntinode, measuredPeaks.c0.rightAntinode, ...
    measuredPeaks.c0.antinode, measuredPeaks.c0.antinodeMismatch, measuredPeaks.c0.leftNode, measuredPeaks.c0.rightNode, ...
    measuredPeaks.c0.nodeDev, measuredPeaks.c0.nodeMismatch, measuredPeaks.c0.normalizedND, measuredPeaks.c0.buoyantMass, ...
    measuredPeaks.c0.NDoffset]);

saveOpenFigures('Calculations', directory);

end
