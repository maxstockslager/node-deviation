function saveOpenFigures(name, directory)

 h = get(0, 'children');
 for ii = 1:length(h)
     saveas(h(ii), strcat(directory, '\', name, '_Figure_', num2str(length(h) + 1 - ii), '.fig'), 'fig');
%      saveas(h(ii), strcat(directory, '\', name, '_Figure_', num2str(length(h) + 1 - ii), '.svg'), 'svg');
     saveas(h(ii), strcat(directory, '\', name, '_Figure_', num2str(length(h) + 1 - ii), '.png'), 'png');
end

