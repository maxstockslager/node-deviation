clear all, close all

filename = 'C:\Users\Max\Dropbox (MIT)\SMR data\ND data\20180606 HSC samples\data_nonsorted_scatterhist.xlsx';
num = xlsread(filename);
labels = num(:, 1);
data = num(:, [2 3]); 

figure
h = scatterhist(data(:, 1), data(:, 2), 'Group', labels, ...
    'Location', 'SouthWest', 'Marker', 'o', 'MarkerSize', [3 3], ...
    'Kernel', 'on');
xlabel('Volume (pL)');
ylabel('ND/volume (Hz/pL)');
% legend('+dox, sorted', '-dox, sorted');
legend('+dox, nonsorted', '-dox, nonsorted');

%     % Plot data with marginal histograms
%     figure
%     h = scatterhist(currentLDA.data(:, 1), currentLDA.data(:, 2), ...
%         'Group', currentLDA.labels, 'Location', 'SouthWest', ...
%         'Color', colors, 'Marker', 'o', 'MarkerSize', [3 3]);
%     xlabel('Mass (pg)');
%     ylabel('MAR (pg/hr)');
%     hold on
%     clr = get(h(1), 'colororder');
%     boxplot(h(2), currentLDA.data(:, 1), currentLDA.labels, ...
%         'orientation', 'horizontal', 'color',clr, 'symbol', '.', ...
%         'boxstyle', 'outline'); %'label', {'',''}
%     hold(h(2), 'on')
%     plot(h(2), mean(data(1).mass), 1, 's', 'Color', colors(1, :), 'MarkerSize', 4);
%     plot(h(2), mean(data(2).mass), 2, 's', 'Color', colors(2, :), 'MarkerSize', 4);
%     boxplot(h(3), currentLDA.data(:, 2), currentLDA.labels, ...
%         'orientation', 'horizontal', 'color',clr, 'symbol', '.', ...
%         'boxstyle', 'outline'); %'label', {'',''}
%     hold(h(3), 'on');
%     plot(h(3), mean(currentData(1).MAR), 1, 's', 'Color', colors(1, :), 'MarkerSize', 4);
%     plot(h(3), mean(currentData(2).MAR), 2, 's', 'Color', colors(2, :), 'MarkerSize', 4);
%     % set(h(2:3), 'XTickLabel', '');
%     view(h(3),[270,90]);
%     % axis(h(1),'auto');
%     if useManualAxisLimits, axis(h(1), manualAxisLimits); end
%     hold off
%     title(exptLabels{exptNumber});  