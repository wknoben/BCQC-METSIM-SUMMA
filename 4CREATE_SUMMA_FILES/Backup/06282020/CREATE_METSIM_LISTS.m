% % % % % % % % % % % % % % % % % % 
% % WRITE METSIM CONTROL FILES  % %
% % % % % % % % % % % % % % % % % % 
% mkdir('LISTS1') % make dir to write lists into
cd('/Users/cjh458/Desktop/BCQC_QC/4gap_filling/LISTS')    % cd into this dir
ID1=num2cell(ID1);  %Convert matrix to cell
ID=num2cell(ID1);   %Convert matrix to cell

% load('/Users/cjh458/Desktop/BCQC_QC/4gap_filling/LISTS/LISTS1.mat');  %load a mat file that contains a list of the snoTEL station names
clear domain state forcing bcqc metsim wind swe station
for i=1:length(ID1)
    domain{i,1}=strcat('bcqc_',num2str(ID1(i)),'_domain.nc'); % create output list for domain-MetSim files
    state{i,1}=strcat('bcqc_',num2str(ID1(i)),'_state.nc');  % create output list for state-MetSim files
    forcing{i,1}=strcat('bcqc_',num2str(ID1(i)),'_forcing.nc');  % create output list for forcing-MetSim files
    bcqc{i,1}=strcat('bcqc_',num2str(ID1(i)),'.csv');    % create a file list used to read bcqc data
    metsim{i,1}=strcat('snotel_',num2str(ID1(i)),'_20170401-20180830.nc');   % create a list of all the output files from MetSim
    wind{i,1}=strcat('snotel_',num2str(ID1(i)),'_20170401-20180830.nc.wind.nc'); % create a list to read wind data downloaded from ERA5
    swe{i,1}=strcat('snotel_',num2str(ID1(i)),'_swe.csv');   % create file used to go back and read origional swe data
    station{i,1}=strcat(num2str(ID1(i)));
end

% % % % % % % % % % % % Convert these lists into tables to write csvs
d=cell2table(domain);
s=cell2table(state);
f=cell2table(forcing);
b=cell2table(bcqc);
m=cell2table(metsim);
w=cell2table(wind);
swe1=cell2table(swe);
station=cell2table(station);

% % % % % % % % % % % Write these tables to file
writetable(d,'LIST_DOMAIN.csv','WriteVariableNames',0)
writetable(s,'LIST_STATE.csv','WriteVariableNames',0)
writetable(f,'LIST_FORCING.csv','WriteVariableNames',0)
writetable(b,'LIST_STATION.csv','WriteVariableNames',0)
writetable(m,'LIST_METSIM.csv','WriteVariableNames',0)
writetable(w,'LIST_WIND.csv','WriteVariableNames',0)
writetable(swe1,'LIST_SWE.csv','WriteVariableNames',0)
writetable(station,'LIST_STATIONnum.csv','WriteVariableNames',0)

LL=array2table(LLE1(1:length(ID1),:));
LL.Properties.VariableNames(1:4) = {'lat','lon','elev','sid'}; % Set table header names
writetable(LL,'LIST_LLES.csv'); % write this table to file 
cd('/Users/cjh458/Desktop/BCQC_QC/4gap_filling/dataCSV_bcqc');
writetable(LL,'LLE.csv'); % write this table to file 

sbatch submit_CONFIG.sh
sbatch submit_DOMAIN.sh
sbatch submit_FORCING.sh
sbatch submit_STATE.sh
sbatch submit_SUBMIT.sh
sbatch submit_RUN.sh

% % % % % % % % % % % % % Go through files to find appropriate hru info

ID1=num2cell(names);
% LISTLLES=LLE;
LLES(length(names),5)=0;
for i=1:length(names) % Iterate through the ID values in this file, (the first file didn't work, due to bad data)
    for j=1:length(LISTLLES)
        str1=strcmp(num2str(names(i,1)),num2str(LISTLLES(j,4))); % Compare Lat strings to information from the snoTEL sites%  
        if (str1==1)
            
            LLES(i,1:4)=LISTLLES(j,1:4); % attach hru info to the LLE list
            
        end
    end
end

% % Write this Lat Lon elev station hruid to file
LL=array2table(LLES(1:length(ID1),:));
LL.Properties.VariableNames(1:4) = {'lat','lon','elev','sid'}; % Set table header names
writetable(LL,'LIST_LLES.csv'); % write this table to file 
writetable(LL,'LLE.csv'); % write this table to file 


% pause; 
% 
% % % %  Make another set of lists that removes the filed MetSim simulations
% mkdir('LISTS2')
% cd('LISTS2')
% %  Created names var by copying file names from output files 
% %  Failed runs were idenified and removed manullly from list


% % Repeat origional produre of creating varibles converting them to tables
% % and writing to csv
clear domain state forcing bcqc metsim wind swe1
for i=1:length(ID1)
    
    domain{i,1}=strcat('bcqc_',num2str(ID1{i}),'_domain.nc');
    state{i,1}=strcat('bcqc_',num2str(ID1{i}),'_state.nc');
    forcing{i,1}=strcat('bcqc_',num2str(ID1{i}),'_forcing.nc');
    bcqc{i,1}=strcat('bcqc_',num2str(ID1{i}),'.csv');
    metsim{i,1}=strcat('bcqc_',num2str(ID1{i}),'_20170401-20180830.nc');
    wind{i,1}=strcat('snotel_',num2str(ID1{i}),'_20170401-20180830.nc.wind.nc');
    swe1{i,1}=strcat('snotel_',num2str(ID1{i}),'_swe.csv');

end

d=cell2table(domain);
s=cell2table(state);
f=cell2table(forcing);
b=cell2table(bcqc); 
m=cell2table(metsim);
w=cell2table(wind);
swe=cell2table(swe1);
station=cell2table(ID1);
 
writetable(d,'LIST_DOMAIN.csv','WriteVariableNames',0)
writetable(s,'LIST_STATE.csv','WriteVariableNames',0)
writetable(f,'LIST_FORCING.csv','WriteVariableNames',0)
writetable(b,'LIST_BCQC.csv','WriteVariableNames',0)
writetable(m,'LIST_METSIM.csv','WriteVariableNames',0)
writetable(w,'LIST_WIND.csv','WriteVariableNames',0)
writetable(swe,'LIST_SWE.csv','WriteVariableNames',0)
writetable(b,'LIST_STATION.csv','WriteVariableNames',0)
writetable(station,'LIST_STATIONnum.csv','WriteVariableNames',0)

% /home/cjh458/summa/summa-22-10-2019/build/bin/summa.exe -g 1 1 -r never -m /home/cjh458/summa/summa-22-10-2019/settings/SETTINGS-snoTEL_04-30/01_SnowEval_201704_201812/fileManager_plato.txt
% 
% /home/cjh458/summa/summa-22-10-2019/settings/SETTINGS-snoTEL_03-31/01_SnowEval_201704_201812

% % CREATE BCQC DATA % % 
mkdir('/Users/cjh458/Desktop/BCQC_QC/1read_BCQC/bcqc_dataMat/bcqcData');
cd('/Users/cjh458/Desktop/BCQC_QC/1read_BCQC/bcqc_dataMat/bcqcData');
load('/Users/cjh458/Desktop/BCQC_QC/1read_BCQC/bcqc_dataMat/files.mat')
for i =1:length(files)
    clearvars -except i files
    load(files{i});
    T=array2table(data);
    T.Properties.VariableNames{1} = 'time';                                                   % Write headers to the table
    T.Properties.VariableNames{2} = 'prec';                                                  % Write headers to the table
    T.Properties.VariableNames{3} = 'swe';                                                  % Write headers to the table
    T.Properties.VariableNames{4} = 'empty';                                            % Write headers to the table
    T.Properties.VariableNames{5} = 'tmin';                                                   % Write headers to the table
    T.Properties.VariableNames{6} = 'tmax';                                                  % Write headers to the table
    T.Properties.VariableNames{7} = 'lat';                                                  % Write headers to the table
    T.Properties.VariableNames{8} = 'lon'; 
    T.Properties.VariableNames{9} = 'elev'; 
    T.Properties.VariableNames{9} = 'siteID'; 
    outfile=strcat('bcqc_',num2str(data(1,10)),'.csv');
    writetable(T,outfile) 
end

       


