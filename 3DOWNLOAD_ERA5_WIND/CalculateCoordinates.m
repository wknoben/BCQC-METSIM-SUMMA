% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% %  FINDS the ERA lat and lon HRU that corresponds to the snotel sites % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
clearvars %clear the workspace

load('/Users/cjh458/Desktop/BCQC-METSIM-SUMMA/3DOWNLOAD_ERA5_WIND/LISTS/SNOTEL_LL.mat'); % Import snoTEL Lat Lon
load('/Users/cjh458/Desktop/BCQC-METSIM-SUMMA/3DOWNLOAD_ERA5_WIND/LISTS/ERA5_LL.mat'); % Import ERA5 Lat Lon
% 

for i = 1:length(stationID) % iterate values form 1 to the total # of suitable stations
    
    count=0;    % initaite a count variable == 0
    Distance(1:length(lat),1:length(lon))=0; % initiate a Distance variable thats the size of lats and lons
    
    for j = 1:length(lat) % iterate through ghcnd latitude values

        for k = 1:length(lon) % iterate through ghcnd longitude values
            
            count = count + 1; % increase count variable by one
            Distance(j,k) = sqrt(((lat{j})-stationID(i,1)).^2 + ((lon{k})-stationID(i,2)).^2); % calculate the distance between ghcnd lats and lons and snoTEL values

        end   % end itration loop 3

    end  % end iteration loop 2
    
    minimum=min(min(Distance)); % find the minimum value amoung the distances
    [x,y]=find(Distance==minimum); % find the x and y coordinates of where this minimum lies
    x1(i,1)=x(1);    %lat{x}; record x values
    y1(i,1)=y(1);    %lon{y}; record y values
    
end

clearvars -except x1 y1
Coordinates(:,1)=x1;
Coordinates(:,2)=y1;
dlmwrite('LIST_lon.csv',x1,',');
dlmwrite('LIST_Lat.csv',y1,',');
dlmwrite('Coordinates.csv',Coordinates(:,:),',');


