clc
close all
addpath Function;
addpath Data;
addpath 'EDF Data';

%% ===================== part 3 =====================
disp('===================== part 3 =====================')
% Part 3 Finding a criterion for sleep description based on PCA
clc
close all
% This part of code loads all of the subjects' data and takes less than a
% minute but if you need to see the results faster you may comment lines 13-27
% and uncomment the piece of code starting at line 29.
patientNum = 5;
for i = 1:patientNum
    cd 'EDF Data'
    FileDirs = dir;
    edfFileDirs = FileDirs(3:7,1);
    edfFileAddress = edfFileDirs(i).name;
    cd ../Data
    FileDirs = dir;
    hypFileDirs = FileDirs(4:8,1);
    hypFileAddress = hypFileDirs(i).name;
    cd ..
    edfFileAddress;
    hypFileAddress;
    [hdr,record,subject{i}] = FeatureExtraction(sprintf('%s',edfFileAddress),sprintf('%s',hypFileAddress));
end

% If you require to see an specific subject's results, please uncomment the
% following code
%     subNumber = 1;
%     cd 'EDF Data'
%     FileDirs = dir;
%     edfFileDirs = FileDirs(3:7,1);
%     edfFileAddress = edfFileDirs(subNumber).name;
%     cd ../Data
%     FileDirs = dir;
%     hypFileDirs = FileDirs(4:8,1);
%     hypFileAddress = hypFileDirs(subNumber).name;
%     cd ..
%     [hdr,record,subject{subNumber}] = FeatureExtraction(sprintf('%s',edfFileAddress),sprintf('%s',hypFileAddress));

%% =============== Q2 ===============
disp(' =============== Q2 ===============')
%% Applying PCA
% X = (subject{subNumber}.X)./repmat(std(subject{subNumber}.X), [size(subject{subNumber}.X,1) , 1]);
subNumber = 1; % If you have used the second peace of code from previous part,
% then you'll need to change "subNumber" variable properly according to
% value you've chosen at previous section at line 31.
[coeff,score,latent] = pca(subject{subNumber}.X);
%  [coeff,score,latent] = pca(X);

%% Plotting the result
clc
close all
plot(cumsum(latent)/sum(latent))
title(sprintf('Subject #%d',subNumber),'color','r')
xlabel('Number of Eigenvectors','color','b')
ylabel('Information(variance)','color','b')

%% =============== Q3 ===============
disp(' =============== Q3 ===============')
% Plotting the result
clc
close all

handle = axes;
m=1; n=1; p=1; q=1; r=1; s=1;
for i = 1 : length(subject{subNumber}.states)
    
    switch subject{subNumber}.states(i)
        case 0 % Wake
        Data{1}(m,1:3) = score(i,1:3);
        m = m + 1;
        case 1
        Data{2}(n,1:3) = score(i,1:3);
        n = n + 1;
        case 2
        Data{3}(p,1:3) = score(i,1:3);
        p = p + 1;
        case 3
        Data{4}(q,1:3) = score(i,1:3);
        q = q + 1;
        case 4
        Data{5}(r,1:3) = score(i,1:3);
        r = r + 1;
        case 5 % Movement (Don't plot)
%         plot3(stateSpaceX,stateSpaceY,stateSpaceZ,'.','color','k')
        case 6 % REM
        Data{6}(s,1:3) = score(i,1:3);
        s = s + 1;
        
    end
    
end

h = figure(1);

color = [[0 0.5 0.9];[1 0 0];[0.9 0.6 0];[.61 .15 .74];[0.3 1 0];[0 0.15 0.8]];
for i = 1 : 6
    stateSpaceX = Data{i}(:,3);
    stateSpaceY = Data{i}(:,2);
    stateSpaceZ = Data{i}(:,1);

    h = plot3(stateSpaceX,stateSpaceY,stateSpaceZ,'.','color',color(i,1:3));
    hold on
end
set(handle,'DataAspectRatio',[1 1 3])
legend('W','1','2','3','4','R')
grid minor
xlabel('PC3','color','b')
ylabel('PC2','color','b')
zlabel('PC1','color','b')
set(handle,'Ydir','reverse')
set(handle,'Xdir','reverse')
savefig('test1.fig')
close all
h1 = openfig('test1.fig','reuse'); % open figure
ax1 = gca; % get handle to axes of figure
zoomFactor = 2;
h = figure; %create new figure
s1 = subplot(2,2,1); %create and get handle to the subplot axes
fig1 = get(ax1,'children'); %get handle to all the children in the figure
copyobj(fig1,s1); %copy children to new parent axes i.e. the subplot axes
az = 45;
el = 30;
view(az, el);
grid on
zoom(zoomFactor)
xlabel('PC3','color','b')
ylabel('PC2','color','b')
zlabel('PC1','color','b')
s1 = subplot(2,2,2); %create and get handle to the subplot axes
fig1 = get(ax1,'children'); %get handle to all the children in the figure
copyobj(fig1,s1); %copy children to new parent axes i.e. the subplot axes
az = 135;
el = 30;
view(az, el);
grid on
zoom(zoomFactor)
xlabel('PC3','color','b')
ylabel('PC2','color','b')
zlabel('PC1','color','b')
s1 = subplot(2,2,3); %create and get handle to the subplot axes
fig1 = get(ax1,'children'); %get handle to all the children in the figure
copyobj(fig1,s1); %copy children to new parent axes i.e. the subplot axes
az = 225;
el = 30;
view(az, el);
grid on
zoom(zoomFactor)
xlabel('PC3','color','b')
ylabel('PC2','color','b')
zlabel('PC1','color','b')
s1 = subplot(2,2,4); %create and get handle to the subplot axes
fig1 = get(ax1,'children'); %get handle to all the children in the figure
copyobj(fig1,s1); %copy children to new parent axes i.e. the subplot axes
az = -45;
el = 30;
view(az, el);
grid on
zoom(zoomFactor)
xlabel('PC3','color','b')
ylabel('PC2','color','b')
zlabel('PC1','color','b')


%% ===================== part 4 =====================
disp('===================== part 4 =====================')
%% =============== Q1 ===============
disp(' =============== Q1 ===============')

subNumber = 1;
% extracting NREM states
NREMIndeces = (subject{subNumber}.states >= 1) & (subject{subNumber}.states <= 4); % states 1 to 4 correspond to NREM
XNREM = subject{subNumber}.X( NREMIndeces, :);
sNREM = subject{subNumber}.states(NREMIndeces); % states of NREM


% fitting linear model to raw data
linMod = fitlm(XNREM, sNREM);
disp(' ')
disp(' ------- linMod for raw data -------')
disp(linMod)

% fitting linear model to normalized data
nNREM = size(XNREM, 1);
XNREMNorm = (XNREM - repmat(mean(XNREM),[nNREM, 1]))./repmat(std(XNREM), [nNREM, 1]);
linModNorm = fitlm(XNREMNorm, sNREM);
disp(' ')
disp(' ------- linMod for normalized data -------')
disp(linModNorm)

% investigating predictors effect sizes
disp(' ')
disp(' ------- R2 effect size (same for raw or normalized)  -------')
effect_size_in_linMdl(XNREM, sNREM, 'print');

disp(' ')
disp(' ------- coeff effect size for normalized data  -------')
coeffNORM = table2array(linModNorm.Coefficients(2:size(linModNorm.Coefficients, 1), 'Estimate'));
[sorted_eff_size, sorted_indeces] = sort(abs(coeffNORM) , 'descend');
disp('var   eff-size')
for i = 1 : size(XNREM, 2)
    disp(sprintf('%s    %.3f',linModNorm.VariableNames{sorted_indeces(i)}, coeffNORM(sorted_indeces(i)))) 
end

%% =============== Q4 ===============
disp(' =============== Q4 ===============')
figure
hold on
histogram(linMod.Fitted(sNREM == 1), 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(linMod.Fitted(sNREM == 2), 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(linMod.Fitted(sNREM == 3), 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(linMod.Fitted(sNREM == 4), 'Normalization', 'probability', 'BinMethod', 'scott')
legend('1', '2', '3', '4')
axis([0 4 0 0.4])
title(sprintf('fitted states(depth) distribution - subject %d', subNumber))

%% =============== Q5 ===============
disp(' =============== Q5 ===============')

% extracting REM data
REMIndeces = (subject{subNumber}.states == 6); % state 6 corresponds to REM
XREM = subject{subNumber}.X(REMIndeces, :);

% extracting WAKE data
WIndeces = (subject{subNumber}.states == 0); % state 0 corresponds to WAKE
XW = subject{subNumber}.X(WIndeces, :);

% prediction
sREM_predicted = predict(linMod, XREM);
sW_predicted = predict(linMod, XW);

% distribution
figure
hold on
histogram(sREM_predicted, 'Normalization', 'probability', 'BinMethod', 'scott')
histogram(sW_predicted, 'Normalization', 'probability', 'BinMethod', 'scott')
legend('REM', 'WAKE')
title(sprintf('predicted sleep depth for WAKE and REM stages - subject %d', subNumber))

[HREMW, pREMW] = ttest2(sREM_predicted, sW_predicted);
[HREM, pREM] = ttest2(sREM_predicted, linMod.Fitted);
[HW, pW] = ttest2(sW_predicted, linMod.Fitted);

%% =============== Q6 ===============
disp(' =============== Q6 ===============')
disp(' ')
for subNumber = 1 : 5
    disp(sprintf(' ++++++++++++++++ subject %d ++++++++++++++++\n', subNumber))
    part4;
end

%% ===================== part 5 =====================
disp('===================== part 5 =====================')

% repeating part 4 with added feature to test the effect of new features on linear regression
for subNumber = 1 : 5
    subject{subNumber}.X = [subject{subNumber}.X, subject{subNumber}.X_added];
    disp(sprintf(' ++++++++++++++++ subject %d ++++++++++++++++\n', subNumber))
    part4;
end

%% ===================== part 5 ===================== 
%Q5 SVM Classification
clc
close all
% This data will be imported into classification learner
SVMData = [Data{1}(:,:),1*ones(length(Data{1}(:,:)),1)];
SVMData = [SVMData;Data{2}(:,:),2*ones(length(Data{2}(:,:)),1)];
SVMData = [SVMData;Data{3}(:,:),3*ones(length(Data{3}(:,:)),1)];
SVMData = [SVMData;Data{4}(:,:),4*ones(length(Data{4}(:,:)),1)];
SVMData = [SVMData;Data{5}(:,:),5*ones(length(Data{5}(:,:)),1)];
SVMData = [SVMData;Data{6}(:,:),6*ones(length(Data{6}(:,:)),1)];

%% ===================== part 5 =====================
% kmeans clustering stage selection method 1
clc
close all
[coeff,score,latent] = pca(subject{subNumber}.X);
data = score(:,1:3);
[idx,C] = kmeans(data,6,'Distance','sqeuclidean','Replicates',50);

stage = nan(1,6);
for i = 1 : 6
    stageSelect = [];
   
    stageSelect(1) = length(find(subject{subNumber}.states(idx==i)==0))/length(find(subject{subNumber}.states==0));
    stageSelect(2) = length(find(subject{subNumber}.states(idx==i)==1))/length(find(subject{subNumber}.states==1));
    stageSelect(3) = length(find(subject{subNumber}.states(idx==i)==2))/length(find(subject{subNumber}.states==2));
    stageSelect(4) = length(find(subject{subNumber}.states(idx==i)==3))/length(find(subject{subNumber}.states==3));
    stageSelect(5) = length(find(subject{subNumber}.states(idx==i)==4))/length(find(subject{subNumber}.states==4));
%     stageSelect(6) =
%     length(find(subject{subNumber}.states(idx==i)==5))/length(find(subject{subNumber}.states==5));
%     % Movements were removed!
    stageSelect(7) = length(find(subject{subNumber}.states(idx==i)==6))/length(find(subject{subNumber}.states==6));
    [maximum,index] = max(stageSelect');
    stage(i) = index-1;
end
close all

m=1; n=1; p=1; q=1; r=1; s=1;
j = 0;
for i = 1 : length(idx)
    j = j +1;
    switch stage(idx(i))
        case 0 % Wake
        DatA{1}(m,1:3) = score(i,1:3);
        m = m + 1;
        case 1
        DatA{2}(n,1:3) = score(i,1:3);
        n = n + 1;
        case 2
        DatA{3}(p,1:3) = score(i,1:3);
        p = p + 1;
        case 3
        DataA{4}(q,1:3) = score(i,1:3);
        q = q + 1;
        case 4
        DatA{5}(r,1:3) = score(i,1:3);
        r = r + 1;
        case 5 % Movement (Don't plot)
%         plot3(stateSpaceX,stateSpaceY,stateSpaceZ,'.','color','k')
        case 6 % REM
        DatA{6}(s,1:3) = score(i,1:3);
        s = s + 1;
        
    end
    
end

color = [[0 0.5 0.9];[1 0 0];[0.9 0.6 0];[.61 .15 .74];[0.3 1 0];[0 0.15 0.8]];
figure
for i = 1 : 6
    
        stateSpaceX = [];
        stateSpaceY = [];
        stateSpaceZ = [];
    if isempty(DatA{i})==0
        stateSpaceX = DatA{i}(:,3);
        stateSpaceY = DatA{i}(:,2);
        stateSpaceZ = DatA{i}(:,1);
        h = plot3(stateSpaceX,stateSpaceY,stateSpaceZ,'.','color',color(i,1:3));
        hold on
    end
end
xlabel('PC3','color','b')
ylabel('PC2','color','b')
zlabel('PC1','color','b')