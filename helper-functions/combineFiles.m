clear all, close all

prefix = 'C:\Users\Max\Dropbox (MIT)\SMR data\ND data\20170926 BAF3 DMSO control\';
foldernames = {'1 t0';
                            '4 t2h restart';
                            '5 t4h';
                            '6 t6h'};
suffix = '\measuredPeaks.csv';
outputName = 'combinedData.xlsx';


timeGaps = [0, 30, 14, 3] * 60 * 4000; % minutes --> indices to add between files. index 1 = before file 1, etc.
combinedArray = [];

for ii = 1:numel(foldernames)
    if ~isempty(combinedArray)
        previousTime = max(combinedArray(:, 2));
    else
        previousTime = 0;
    end
    
    filename = strcat(prefix, foldernames{ii}, suffix);
    currentArray = csvread(filename);
    newTimeVector =  previousTime + currentArray(:, 1) + timeGaps(ii);
    currentArray = [currentArray(:, 1), newTimeVector, currentArray(:, 2:end)];
    combinedArray = [combinedArray; currentArray];
end

outputFilename = strcat(prefix, outputName);
xlswrite(outputFilename, combinedArray);