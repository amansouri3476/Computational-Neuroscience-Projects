%% =============== Part 2 - Introduction to dataset ================
clc
close all

addpath Data;
addpath Function;
subjects = {};
for i = 1 : 10
   s = load(sprintf('s%d.mat',i));
   subjects{i} = s.(sprintf('s%d',i));
end

%% =============== Part 3 ================
%% ============ Q1 ============
subn = 3; %subject number
Fs = 256; %sampling freq.
bpf = BPF(1001, 0.5, 30, Fs); %passing (0.5 - 30) Hz

filtered_subjects = subjects; %subjects bpfiltered data 
for i = 1 : 10
    filtered_subjects{i}.train(2:9,:) = FilterDFT(subjects{i}.train(2:9,:)', bpf)';
    filtered_subjects{i}.test(2:9,:) = FilterDFT(subjects{i}.test(2:9,:)', bpf)';
    
    trials{i} = TrialExtraction(filtered_subjects{i});
end

%% =============== Part 4 - P300 Speller Implementation ================
%% ============ Q1 ============
clc
close all

% Downsample of target_train
subjectDataDStarget_train = trials{subn}.target_train(:,1:4:end,:);
% Downsample of ntarget_train
subjectDataDSntarget_train = trials{subn}.ntarget_train(:,1:4:end,:);
% Downsample of target_test
subjectDataDStarget_test = trials{subn}.target_test(:,1:4:end,:);
% Downsample of ntarget_test
subjectDataDSntarget_test = trials{subn}.ntarget_test(:,1:4:end,:);
% Downsample of all channels all trials and target/ntarget_train/test. Each
% field of the struct is a downsampled version of the 3-D matrix figure of
% Part 2 in the HW pdf.
subjectsDataDS{subn} = struct('target_train',subjectDataDStarget_train,'ntarget_train',subjectDataDSntarget_train,'target_test',subjectDataDStarget_test,'ntarget_test',subjectDataDSntarget_test);

%% ============ Q2 ============
clc
close all
% target_train
allelectrodestarget_train = subjectsDataDS{subn}.target_train(2:9,:,:);
% rows are different trials and columns are observations
allelectrodesserialtarget_train = reshape(allelectrodestarget_train,[size(allelectrodestarget_train,1)*size(allelectrodestarget_train,2) size(allelectrodestarget_train,3)]);
allelectrodesserialtarget_train = allelectrodesserialtarget_train';

% ntarget_train
allelectrodesntarget_train = subjectsDataDS{subn}.ntarget_train(2:9,:,:);
% rows are different trials and columns are observations
allelectrodesserialntarget_train = reshape(allelectrodesntarget_train,[size(allelectrodesntarget_train,1)*size(allelectrodesntarget_train,2) size(allelectrodesntarget_train,3)]);
allelectrodesserialntarget_train = allelectrodesserialntarget_train';

% target_test
allelectrodestarget_test = subjectsDataDS{subn}.target_test(2:9,:,:);
% rows are different trials and columns are observations
allelectrodesserialtarget_test = reshape(allelectrodestarget_test,[size(allelectrodestarget_test,1)*size(allelectrodestarget_test,2) size(allelectrodestarget_test,3)]);
allelectrodesserialtarget_test = allelectrodesserialtarget_test';

% ntarget_test
allelectrodesntarget_test = subjectsDataDS{subn}.ntarget_test(2:9,:,:);
% rows are different trials and columns are observations
allelectrodesserialntarget_test = reshape(allelectrodesntarget_test,[size(allelectrodesntarget_test,1)*size(allelectrodesntarget_test,2) size(allelectrodesntarget_test,3)]);
allelectrodesserialntarget_test = allelectrodesserialntarget_test';

% Generalizing for all subjects
allsubjectsallelectrodesserial{subn} = struct('target_train',allelectrodesserialtarget_train,'ntarget_train',allelectrodesserialntarget_train,'target_test',allelectrodesserialtarget_test,'ntarget_test',allelectrodesserialntarget_test);


% LDA Predictors using train data
LDApredictors = [allsubjectsallelectrodesserial{subn}.target_train;allsubjectsallelectrodesserial{subn}.ntarget_train];

% Classification Values
trainclassifyvalue = [ones(size(allsubjectsallelectrodesserial{subn}.target_train,1),1);zeros(size(allsubjectsallelectrodesserial{subn}.ntarget_train,1),1)];
obj = fitcdiscr(LDApredictors,trainclassifyvalue);
% Accuracy claculation for train data (subject 1)
predictedtrainlabels = predict(obj,LDApredictors);
trainaccuracy(subn) = (length(predictedtrainlabels)-sum(xor(predictedtrainlabels,trainclassifyvalue)))/length(predictedtrainlabels);
% Accuracy claculation for test data (subject 1)
testdata = [allsubjectsallelectrodesserial{subn}.target_test;allsubjectsallelectrodesserial{subn}.ntarget_test];
testclassifyvalue = [ones(size(allsubjectsallelectrodesserial{subn}.target_test,1),1);zeros(size(allsubjectsallelectrodesserial{subn}.ntarget_test,1),1)];
predictedtestlabels = predict(obj,testdata);
testaccuracy(subn) = (length(predictedtestlabels)-sum(xor(predictedtestlabels,testclassifyvalue)))/length(predictedtestlabels);

%% Cross Validation
clc
close all
% Partitioning train data into 5 folds randomly. When each fold is taken, it's
% been removed from the data to create other folds.

% We have to partition LDApredictors matrix's trials to calculate 5-fold cross
% validation

% permuting trials in predictors randomly 
trialsrandomindeces = randperm(size(LDApredictors,1))';

kfold = 5;
fold = {};
for i = 1:kfold

    fold{i} = LDApredictors(trialsrandomindeces(1+(i-1)*size(LDApredictors,1)/kfold:i*size(LDApredictors,1)/kfold),:);
    foldtrainclassifyvalue{i} = trainclassifyvalue(trialsrandomindeces(1+(i-1)*size(LDApredictors,1)/kfold:i*size(LDApredictors,1)/kfold));
end

for i = 1:kfold
    traindata = [];
    testdata = [];
    testdata = fold{i};
    testclassifyvalue = foldtrainclassifyvalue{i};
    trainclassifyvaluecv{i} = [];
    % collecting train data
    for j = 1:kfold
       if j == i
           continue;
       end
       traindata = [traindata;fold{j}];
       trainclassifyvaluecv{i} = [trainclassifyvaluecv{i};foldtrainclassifyvalue{j}];
    end

  % Classification Values
    objkfold = fitcdiscr(traindata,trainclassifyvaluecv{i});
  % Accuracy claculation for test data
    predictedtestlabels = predict(objkfold,testdata);
    testaccuracykfold(i) = (length(predictedtestlabels)-sum(xor(predictedtestlabels,testclassifyvalue)))/length(predictedtestlabels);

end

kfoldaccuracy = mean(testaccuracykfold);
%% ============ Q3 ============ %%
clc
close all
subn = 3;
% Extraction of subject's filtered test data

for i = 1 : 10
    
    trialstest{i} = TrialExtractionmodified(filtered_subjects{i});
    
end

% Downsample of test data
subjectDataDStest{subn} = trialstest{subn}(:, 1:4:end, :);

% test
allelectrodestest = subjectDataDStest{subn}(2:9,:,:);
% rows are different trials and columns are observations
allelectrodesserialtest = reshape(allelectrodestest,[size(allelectrodestest,1)*size(allelectrodestest,2) size(allelectrodestest,3)]);
allelectrodesserialtest = allelectrodesserialtest';

% Generalizing for all subjects
allsubjectsallelectrodesserialtest{subn} = allelectrodesserialtest;

testlabelpredicted{subn} = predict(obj,allsubjectsallelectrodesserialtest{subn});

trialsindeces = IndExtractionmodified(subjects{subn});
% realtrialsindeces = IndExtraction(subjects{subn});

displayedflashescharacter{subn} = subjects{subn}.test(10,trialsindeces(testlabelpredicted{subn}==1)); 
% realdisplayedflashescharacter{subn} = subjects{subn}.test(10,realtrialsindeces.target_test);

% Estimating the characters.
numofletters = 5;
if subn<=2 % method is SC
            for k = 1:numofletters

                decodedchar = mode(displayedflashescharacter{subn}(1+(k-1)*floor(length(displayedflashescharacter{subn})/numofletters):k*floor(length(displayedflashescharacter{subn})/numofletters)));
                decodedcharacter(subn,k) = lookupSC(decodedchar);

            end
else
        
        for k = 1:numofletters

                partitioned = displayedflashescharacter{subn}(1+(k-1)*floor(length(displayedflashescharacter{subn})/numofletters):k*floor(length(displayedflashescharacter{subn})/numofletters));
                decodedchar = [mode(partitioned(find(partitioned<=6))),mode(partitioned(find(partitioned>6)))];
                decodedcharacter(subn,k) = lookupRC(decodedchar);
        end
end
%% ============ Q4 ============
clc
close all
plot(obj.Coeffs(2).Linear);
hold on
plot(obj.Coeffs(3).Linear);
title('Coeffs of boundary hyperplane','color','r')
xlabel('features','color','b')
ylabel('Importance','color','b')

figure
coeffsreshaped = reshape(obj.Coeffs(3).Linear,[8 length(obj.Coeffs(3).Linear)/8]);
plot(mean(coeffsreshaped,1));
title('Channel Averaged Importance','color','r')
xlabel('time(arbitrary origin)','color','b')
ylabel('Importance','color','b')
figure
plot(mean(coeffsreshaped,2));
title('Time Averaged Importance','color','r')
xlabel('Channels','color','b')
ylabel('Importance','color','b')
%% ============ Q5 ============ %%
%% ============ Q1 repetition for all subjects ============
clc
close all
for i = 1:length(subjects)
        
        subn = i;
        % Downsample of target_train
        subjectDataDStarget_train = trials{subn}.target_train(:,1:4:end,:);
        % Downsample of ntarget_train
        subjectDataDSntarget_train = trials{subn}.ntarget_train(:,1:4:end,:);
        % Downsample of target_test
        subjectDataDStarget_test = trials{subn}.target_test(:,1:4:end,:);
        % Downsample of ntarget_test
        subjectDataDSntarget_test = trials{subn}.ntarget_test(:,1:4:end,:);
        % Downsample of all channels all trials and target/ntarget_train/test. Each
        % field of the struct is a downsampled version of the 3-D matrix figure of
        % Part 2 in the HW pdf.
        subjectsDataDS{subn} = struct('target_train',subjectDataDStarget_train,'ntarget_train',subjectDataDSntarget_train,'target_test',subjectDataDStarget_test,'ntarget_test',subjectDataDSntarget_test);
end

%% ============ Q2 repetition for all subjects ============ %%

clc
close all
for i = 1:length(subjects)
        subn = i;
        % target_train
        allelectrodestarget_train = subjectsDataDS{subn}.target_train(2:9,:,:);
        % rows are different trials and columns are observations
        allelectrodesserialtarget_train = reshape(allelectrodestarget_train,[size(allelectrodestarget_train,1)*size(allelectrodestarget_train,2) size(allelectrodestarget_train,3)]);
        allelectrodesserialtarget_train = allelectrodesserialtarget_train';

        % ntarget_train
        allelectrodesntarget_train = subjectsDataDS{subn}.ntarget_train(2:9,:,:);
        % rows are different trials and columns are observations
        allelectrodesserialntarget_train = reshape(allelectrodesntarget_train,[size(allelectrodesntarget_train,1)*size(allelectrodesntarget_train,2) size(allelectrodesntarget_train,3)]);
        allelectrodesserialntarget_train = allelectrodesserialntarget_train';

        % target_test
        allelectrodestarget_test = subjectsDataDS{subn}.target_test(2:9,:,:);
        % rows are different trials and columns are observations
        allelectrodesserialtarget_test = reshape(allelectrodestarget_test,[size(allelectrodestarget_test,1)*size(allelectrodestarget_test,2) size(allelectrodestarget_test,3)]);
        allelectrodesserialtarget_test = allelectrodesserialtarget_test';

        % ntarget_test
        allelectrodesntarget_test = subjectsDataDS{subn}.ntarget_test(2:9,:,:);
        % rows are different trials and columns are observations
        allelectrodesserialntarget_test = reshape(allelectrodesntarget_test,[size(allelectrodesntarget_test,1)*size(allelectrodesntarget_test,2) size(allelectrodesntarget_test,3)]);
        allelectrodesserialntarget_test = allelectrodesserialntarget_test';

        % Generalizing for all subjects
        allsubjectsallelectrodesserial{subn} = struct('target_train',allelectrodesserialtarget_train,'ntarget_train',allelectrodesserialntarget_train,'target_test',allelectrodesserialtarget_test,'ntarget_test',allelectrodesserialntarget_test);


        % LDA Predictors using train data
        LDApredictors = [allsubjectsallelectrodesserial{subn}.target_train;allsubjectsallelectrodesserial{subn}.ntarget_train];

        % Classification Values
        trainclassifyvalue = [ones(size(allsubjectsallelectrodesserial{subn}.target_train,1),1);zeros(size(allsubjectsallelectrodesserial{subn}.ntarget_train,1),1)];
        obj = fitcdiscr(LDApredictors,trainclassifyvalue);
        % Accuracy claculation for train data (subject 1)
        predictedtrainlabels = predict(obj,LDApredictors);
        trainaccuracy(subn) = (length(predictedtrainlabels)-sum(xor(predictedtrainlabels,trainclassifyvalue)))/length(predictedtrainlabels);
        % Accuracy claculation for test data (subject 1)
        testdata = [allsubjectsallelectrodesserial{subn}.target_test;allsubjectsallelectrodesserial{subn}.ntarget_test];
        testclassifyvalue = [ones(size(allsubjectsallelectrodesserial{subn}.target_test,1),1);zeros(size(allsubjectsallelectrodesserial{subn}.ntarget_test,1),1)];
        predictedtestlabels = predict(obj,testdata);
        testaccuracy(subn) = (length(predictedtestlabels)-sum(xor(predictedtestlabels,testclassifyvalue)))/length(predictedtestlabels);
        if subn<=2
            trialsindeces = IndExtractionmodified(subjects{subn});
            precision(subn) = sum(and(predictedtestlabels,testclassifyvalue))/sum(testclassifyvalue);
            recall(subn) = sum(and(predictedtestlabels,testclassifyvalue))/sum(predictedtestlabels);
              
        end
        
        %============ Cross Validation for all subjects using 5-fold ============%% 
        % Partitioning train data into 5 folds randomly. When each fold is taken, it's
        % been removed from the data to create other folds.

        % We have to partition LDApredictors matrix's trials to calculate 5-fold cross
        % validation

        % permuting trials in predictors randomly 
        trialsrandomindeces = randperm(size(LDApredictors,1))';

        kfold = 5;
        fold = {};
        for k = 1:kfold

            fold{k} = LDApredictors(trialsrandomindeces(1+(k-1)*size(LDApredictors,1)/kfold:k*size(LDApredictors,1)/kfold),:);
            foldtrainclassifyvalue{k} = trainclassifyvalue(trialsrandomindeces(1+(k-1)*size(LDApredictors,1)/kfold:k*size(LDApredictors,1)/kfold));
        end

        for k = 1:kfold
            traindata = [];
            testdata = [];
            testdata = fold{k};
            testclassifyvalue = foldtrainclassifyvalue{k};
            trainclassifyvaluecv{k} = [];
            % collecting train data
            for j = 1:kfold
               if j == k
                   continue;
               end
               traindata = [traindata;fold{j}];
               trainclassifyvaluecv{k} = [trainclassifyvaluecv{k};foldtrainclassifyvalue{k}];
            end

          % Classification Values
            objkfold = fitcdiscr(traindata,trainclassifyvaluecv{k});
          % Accuracy claculation for test data
            predictedtestlabels = predict(objkfold,testdata);
            testaccuracykfold(k) = (length(predictedtestlabels)-sum(xor(predictedtestlabels,testclassifyvalue)))/length(predictedtestlabels);

        end

        kfoldaccuracy(i) = mean(testaccuracykfold);

end


%% ============ Q3 repetition for all subjects ============ %% 
clc
close all

decodedchar = {};
for i = 1 : 10
    subn = i;
    % Extraction of subject's filtered test data
    trialstest{subn} = TrialExtractionmodified(filtered_subjects{subn});
    % Downsample of test data
    subjectDataDStest{subn} = trialstest{subn}(:, 1:4:end, :);
    % test
    allelectrodestest = subjectDataDStest{subn}(2:9,:,:);
    % rows are different trials and columns are observations
    allelectrodesserialtest = reshape(allelectrodestest,[size(allelectrodestest,1)*size(allelectrodestest,2) size(allelectrodestest,3)]);
    allelectrodesserialtest = allelectrodesserialtest';

    % Generalizing for all subjects
    allsubjectsallelectrodesserialtest{subn} = allelectrodesserialtest;
    
    
    % LDA Predictors using train data
    LDApredictors = [allsubjectsallelectrodesserial{subn}.target_train;allsubjectsallelectrodesserial{subn}.ntarget_train];

    % Classification Values
    trainclassifyvalue = [ones(size(allsubjectsallelectrodesserial{subn}.target_train,1),1);zeros(size(allsubjectsallelectrodesserial{subn}.ntarget_train,1),1)];
    obj = fitcdiscr(LDApredictors,trainclassifyvalue);
    testlabelpredicted{subn} = predict(obj,allsubjectsallelectrodesserialtest{subn});

    trialsindeces = IndExtractionmodified(subjects{subn});
    % realtrialsindeces = IndExtraction(subjects{subn});

    displayedflashescharacter{subn} = subjects{subn}.test(10,trialsindeces(testlabelpredicted{subn}==1)); 
    % realdisplayedflashescharacter{subn} = subjects{subn}.test(10,realtrialsindeces.target_test);

    % Estimating the characters.
    numofletters = 5;
    if subn<=2 % method is SC
            for k = 1:numofletters

                decodedchar = mode(displayedflashescharacter{subn}(1+(k-1)*floor(length(displayedflashescharacter{subn})/numofletters):k*floor(length(displayedflashescharacter{subn})/numofletters)));
                decodedcharacter(subn,k) = lookupSC(decodedchar);

            end
    else
        
        for k = 1:numofletters

                partitioned = displayedflashescharacter{subn}(1+(k-1)*floor(length(displayedflashescharacter{subn})/numofletters):k*floor(length(displayedflashescharacter{subn})/numofletters));
                decodedchar = [mode(partitioned(find(partitioned<=6))),mode(partitioned(find(partitioned>6)))];
                decodedcharacter(subn,k) = lookupRC(decodedchar);
        end
    end
    
        % ============ Q4 repetition for all subjects ============ %
        subplot(4,5,i)
        coeffsreshaped = reshape(obj.Coeffs(3).Linear,[8 length(obj.Coeffs(3).Linear)/8]);
        plot(mean(coeffsreshaped,1));
        title(sprintf('subject %d',i),'color','r');
        subplot(4,5,i+10)
        plot(mean(coeffsreshaped,2));
        title(sprintf('subject %d',i),'color','r')
end

%% ============ Q5 a ============ %%
hist(testaccuracy)
xlabel('Accuracy','color','b')
ylabel('Number of Subjects','color','b')
title('Accuracy rate on test data histogram','color','r')