function convertToDouble

[fn, pn] = uigetfile;
filename = strcat(pn, fn);

fileID = fopen(filename, 'r', 'b');
frequencySignal = fread(fileID, 'uint32');
% frequencySignal(1:129:end) = [];
frequencySignal = frequencySignal * (12.5e6/2^32); 
fclose(fileID);

outputFilename = 'c2';
save(outputFilename, 'frequencySignal');

end

