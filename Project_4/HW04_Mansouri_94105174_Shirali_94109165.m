%% ==================== Part 3 ==================== %%

%% ==================== Q1 ==================== %%
clc
addpath('nii tools','dataset','outputs');
% cd outputs;
% % addpath('Run1','Run2','Run3','Run4','Run5','Run6','Run7','Run8','RunAverage');
% cd ..;

 addpath nii' tools';
 addpath dataset;
 addpath outputs_q4;

numOfRuns = 8;
numOfGenres = 5;
for i = 1:numOfRuns
    
    run(i) = importdata(sprintf('run-0%d_events.tsv',i));
    temp = cell2table(run(i).textdata);
    onsetTime(:,i) = cellfun(@str2double , run(i).textdata([2:end],1));
    genre(:,i) = table2array(temp([2:end],3));
%     spm_file_split(sprintf('run-0%d_bold.nii',i));
    
end

ambient = cellfun(@(x) strcmp(x,'ambient'), genre);
country = cellfun(@(x) strcmp(x,'country'), genre);
rocknroll = cellfun(@(x) strcmp(x,'rocknroll'), genre);
symphonic = cellfun(@(x) strcmp(x,'symphonic'), genre);
metal = cellfun(@(x) strcmp(x,'metal'), genre);

% single contrast
contrasts = [1 zeros(1,10);0 0 1 zeros(1,8);zeros(1,4) 1 zeros(1,6);zeros(1,6) 1 zeros(1,4);zeros(1,8) 1 0 0];
% amb contrasts added
contrasts = [contrasts;[1 0 -1 zeros(1,8)];[1 0 0 0 -1 zeros(1,6)];[1 0 0 0 0 0 -1 zeros(1,4)];[1 zeros(1,7) -1 0 0]];
% rr contrasts added
contrasts = [contrasts;-[1 0 -1 zeros(1,8)];[zeros(1,2) 1 0 -1 zeros(1,6)];[zeros(1,2) 1 0 0 0 -1 zeros(1,4)];[zeros(1,2) 1 0 0 0 0 0 -1 zeros(1,2)]];
% ctr contrasts added
contrasts = [contrasts;-[1 0 0 0 -1 zeros(1,6)];-[zeros(1,2) 1 0 -1 zeros(1,6)];[zeros(1,4) 1 0 -1 zeros(1,4)];[zeros(1,4) 1 0 0 0 -1 zeros(1,2)]];
% mtl contrasts added
contrasts = [contrasts; -[1 0 0 0 0 0 -1 zeros(1,4)]; -[zeros(1,2) 1 0 0 0 -1 zeros(1,4)];-[zeros(1,4) 1 0 -1 zeros(1,4)];[zeros(1,6) 1 0 -1 zeros(1,2)]];
% sym contrasts added
contrasts = [contrasts;-[1 zeros(1,7) -1 0 0];-[zeros(1,2) 1 0 0 0 0 0 -1 zeros(1,2)];-[zeros(1,4) 1 0 0 0 -1 zeros(1,2)];-[zeros(1,6) 1 0 -1 zeros(1,2)]];


% Generating Z-values from t-values (Results are not observable for you:) )

% Please do not run the following piece of code which is commented. It was
% written to calculate z-values automatically.

% dof = 135;
% cd outputs/
% for i = 1 : numOfRuns - 1
% 
%     cd(sprintf('Run%d',i));
%     
%     for j = 1 : numOfGenres
%         
%         
%         movefile(sprintf('spmT_000%d.nii',j),'../../')
%         cd ../..
%         temp1 = load_untouch_nii(sprintf('spmT_000%d.nii',j));
%         z = spm_t2z(temp1.img,dof);
%         temp2 = make_nii(z);
%         save_nii(temp2,sprintf('spmZ_000%d',j))
%         movefile(sprintf('spmT_000%d.nii',j),sprintf('outputs/Run%d',i))
%         movefile(sprintf('spmZ_000%d.img',j),sprintf('outputs/Run%d',i))
%         movefile(sprintf('spmZ_000%d.hdr',j),sprintf('outputs/Run%d',i))
%         delete(sprintf('spmZ_000%d.mat',j))
%         cd(sprintf('outputs/Run%d',i));
%         
%     end
%     
%     cd ..
%     
% end
% 
% cd ..
% 
% 
% dof = 135;
% cd outputs/
% for i = 1 : numOfRuns - 1
% 
%     cd(sprintf('Run%d',i));
%     
%     for j = numOfGenres + 1 : 9
%         
%         
%         movefile(sprintf('spmT_000%d.nii',j),'../../')
%         cd ../..
%         temp1 = load_untouch_nii(sprintf('spmT_000%d.nii',j));
%         z = spm_t2z(temp1.img,dof);
%         temp2 = make_nii(z);
%         save_nii(temp2,sprintf('spmZ_000%d',j))
%         movefile(sprintf('spmT_000%d.nii',j),sprintf('outputs/Run%d',i))
%         movefile(sprintf('spmZ_000%d.img',j),sprintf('outputs/Run%d',i))
%         movefile(sprintf('spmZ_000%d.hdr',j),sprintf('outputs/Run%d',i))
%         delete(sprintf('spmZ_000%d.mat',j))
%         cd(sprintf('outputs/Run%d',i));
%         
%     end
%     
%     cd ..
%     
% end
% 
% cd ..
% 
% 
% dof = 135;
% cd outputs/
% for i = 1 : numOfRuns - 1
% 
%     cd(sprintf('Run%d',i));
%     
%     for j = 10 : 25
%         
%         
%         movefile(sprintf('spmT_00%d.nii',j),'../../')
%         cd ../..
%         temp1 = load_untouch_nii(sprintf('spmT_00%d.nii',j));
%         z = spm_t2z(temp1.img,dof);
%         temp2 = make_nii(z);
%         save_nii(temp2,sprintf('spmZ_00%d',j))
%         movefile(sprintf('spmT_00%d.nii',j),sprintf('outputs/Run%d',i))
%         movefile(sprintf('spmZ_00%d.img',j),sprintf('outputs/Run%d',i))
%         movefile(sprintf('spmZ_00%d.hdr',j),sprintf('outputs/Run%d',i))
%         delete(sprintf('spmZ_00%d.mat',j))
%         cd(sprintf('outputs/Run%d',i));
%         
%     end
%     
%     cd ..
%     
% end
% 
% cd ..

%% ==================== Q2 ==================== %%
clc

% Please do not run the following piece of code which is written to
% calculate the average z-value over runs.

% cd outputs/
% Zvals = {0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0};
% for j = 1 : 9
%         for i = 1 : numOfRuns - 1
% 
%             cd(sprintf('Run%d',i));
%             movefile(sprintf('spmZ_000%d.img',j),'../../')
%             movefile(sprintf('spmZ_000%d.hdr',j),'../../')
%             cd ../..
%             temp1 = load_untouch_nii(sprintf('spmZ_000%d.img',j));
%             Zvals{j,1} = Zvals{j,1} + temp1.img;
%             movefile(sprintf('spmZ_000%d.img',j),sprintf('outputs/Run%d',i))
%             movefile(sprintf('spmZ_000%d.hdr',j),sprintf('outputs/Run%d',i))
%             
%             cd outputs
%          end
% 
% 
% end
% cd ..
% 
% 
% cd outputs/
% for j = 10 : 25
%         for i = 1 : numOfRuns - 1
% 
%             cd(sprintf('Run%d',i));
%             movefile(sprintf('spmZ_00%d.img',j),'../../')
%             movefile(sprintf('spmZ_00%d.hdr',j),'../../')
%             cd ../..
%             temp1 = load_untouch_nii(sprintf('spmZ_00%d.img',j));
%             Zvals{j,1} = Zvals{j,1} + temp1.img;
%             movefile(sprintf('spmZ_00%d.img',j),sprintf('outputs/Run%d',i))
%             movefile(sprintf('spmZ_00%d.hdr',j),sprintf('outputs/Run%d',i))
%             
%             cd outputs
%          end
% 
% 
% end
% cd ..
% 
% for i = 1 : 25
%     
%    Zvals{i,1} = Zvals{i,1}/(numOfRuns-1);
% 
% end
%%

% cd outputs/Run1/
% movefile('spmZ_0001.hdr','../../')
% cd ../..
% 
% for i = 1 : 25
%     
%     temp = make_nii(Zvals{i,1});
%     save_nii(temp,sprintf('Zvalue_avg_%d',i))    
%     movefile(sprintf('Zvalue_avg_%d.img',i),'outputs/RunAverage')
%     movefile(sprintf('Zvalue_avg_%d.hdr',i),'outputs/RunAverage')
%     delete(sprintf('Zvalue_avg_%d.mat',i))
% end
% 
% movefile('spmZ_0001.hdr','outputs/Run1/')


%% =============== Part 4 ==================
contrasts = eye(51);

cont_batch = load('contrast_batch.mat');
jobs = cont_batch.matlabbatch;
address = jobs{1}.spm.stats.con.spmmat{1};

for i = 1 : 8
    new_address = strrep(address,'Run1',sprintf('Run%d',i));
    jobs{1}.spm.stats.con.spmmat{1} = new_address;
    spm_jobman('run',jobs);
end

%% ======== Q1 ========
address = 'outputs_q4/Run1/spmT';
imgs = nan(25, 8, 160*160*36);

for run = 1 : 8
    new_address = strrep(address,'Run1',sprintf('Run%d',run));
    for c = 1 : 25 
        new_address = sprintf('%s_%04d.nii', address, c);
        [nii,img] = convertnii2mat(new_address, 'untouch');
        imgs(c, run, :) = img(:);
        close all
    end
end

%% ======== Q2 & 3 & 5 ========
p_vals = nan(size(imgs,3),1);

wb = waitbar(0,'ANOVA1 analysis is in progress');
wb_update_step = floor(size(imgs,3)/500);

for voxel = 1 : size(imgs,3)
    temp = imgs(:,:,voxel);
    M = [temp(ambient) temp(country) temp(metal) temp(rocknroll) temp(symphonic)]; % 8'th column of ambient,country,... is zero
    p_vals(voxel) = anova1(M, 1:5, 'off');
    
    if(mod(voxel,wb_update_step) == 0)
        waitbar(voxel/size(imgs,3), wb)
    end
end
close(wb)
%%
%load('p_vals.mat')
% generating labels
labels = reshape(ambient(:, 1:7), [numel(ambient(:, 1:7)) 1]) + ...
reshape(country(:, 1:7), [numel(country(:, 1:7)) 1]) * 2 + ...
reshape(metal(:, 1:7), [numel(metal(:, 1:7)) 1]) * 3 + ...
reshape(rocknroll(:, 1:7), [numel(rocknroll(:, 1:7)) 1]) * 4 + ...
reshape(symphonic(:, 1:7), [numel(symphonic(:, 1:7)) 1]) * 5;

ths = logspace(-4, -2, 9);
for i = 1 : length(ths)
    th = ths(i);
    % feature reduction by choosing discriminative voxels
    d_voxels_indeces = p_vals < th; % discriminative voxels indeces
    disp(sprintf('number of voxels chose in ANOVA1 analysis and th = %.3f: %d', th, sum(d_voxels_indeces)))

    d_imgs = reshape(imgs(:, 1:7, d_voxels_indeces), [size(imgs,1)*(size(imgs,2) -1) sum(d_voxels_indeces)]);

    % feature reduction by PCA (just to reduce feature's space dimension to
    % number of observations)
    [~, pca_d_imgs, latent] = pca(d_imgs);

    % classification
    predictors = pca_d_imgs(:, 1 : min(24,size(pca_d_imgs,2)) ); 
    discr_obj = fitcdiscr(predictors, labels);

    % testing the error rate on prediction
    predicted_labels = predict(discr_obj, predictors);
    percent_of_correctness(i) = 100*sum(labels == predicted_labels)/length(labels);
    disp(sprintf('percent of correct predictions using fitcdiscr (p_val th = %.3f): %.3f%%', th, percent_of_correctness(i)));
end

figure
loglog(ths, percent_of_correctness)
title('prediction accuracy vs ANOVA1 th')
xlabel('th')
ylabel('%')

%% ======== Q4 & 5 ========
% ANOVA p_vals calculation for each fold
p_vals_cv = nan(size(imgs,3),1);

for r = 1 : 7 % leave-out run
    wb = waitbar(0,'ANOVA1 analysis is in progress');
    wb_update_step = floor(size(imgs,3)/500);
    
    allowed_run = [(1 : r-1) (r+1 : 7)];
    for voxel = 1 : size(imgs,3)
        temp = imgs(:,allowed_run,voxel);
        M = [temp(ambient(:,allowed_run)) temp(country(:,allowed_run)) temp(metal(:,allowed_run)) temp(rocknroll(:,allowed_run)) temp(symphonic(:,allowed_run))]; % 8'th column of ambient,country,... is zero
        p_vals_cv(voxel) = anova1(M, 1:5, 'off');
    
        if(mod(voxel,wb_update_step) == 0)
            waitbar(voxel/size(imgs,3), wb)
        end
    end
    save(sprintf('p_vals_cv_%d.mat', r), 'p_vals_cv')
    close(wb)
end

%%
% cross validation!
nvoxels = floor(logspace(0,4.5,30));
cv_accuracy = nan(size(nvoxels));

for i = 1 : length(nvoxels)
    num_of_voxels = nvoxels(i);
    for r = 1 : 7
        load(sprintf('p_vals_cv_%d.mat', r));
        allowed_run = [(1 : r-1) (r+1 : 7)];
    
        % feature reduction by choosing discriminative voxels
        d_voxels_indeces = zeros(size(p_vals_cv));
        [sorted_p_vals, sorted_indeces] = sort(p_vals_cv, 'ascend');
        d_voxels_indeces(sorted_indeces(1 : num_of_voxels)) = 1;
        d_voxels_indeces = logical(d_voxels_indeces);
    

        d_imgs = reshape(imgs(:, allowed_run, d_voxels_indeces), [size(imgs,1)*length(allowed_run) sum(d_voxels_indeces)]);

        % feature reduction by PCA (just to reduce feature's space dimension to
        % number of observations)
        [coeff, pca_d_imgs, latent] = pca(d_imgs);
    
        % generating labels
        labels = reshape(ambient(:, allowed_run), [numel(ambient(:, allowed_run)) 1]) + ...
        reshape(country(:, allowed_run), [numel(country(:, allowed_run)) 1]) * 2 + ...
        reshape(metal(:, allowed_run), [numel(metal(:, allowed_run)) 1]) * 3 + ...
        reshape(rocknroll(:, allowed_run), [numel(rocknroll(:, allowed_run)) 1]) * 4 + ...
        reshape(symphonic(:, allowed_run), [numel(symphonic(:, allowed_run)) 1]) * 5;

        % classification
        predictors = pca_d_imgs(:, 1 : min(20,size(pca_d_imgs,2)) ); 
        discr_obj = fitcdiscr(predictors, labels);

        % generating predictors and labels for left out run
        r = r;
        test_d_imgs = squeeze(imgs(:, r, d_voxels_indeces));
        % applying pca coeff
        test_pca_d_imgs = (test_d_imgs - repmat(mean(d_imgs), [size(test_d_imgs,1) 1]) ) * coeff ;
        test_predictors = test_pca_d_imgs(:, 1 : size(predictors,2) );
    
        test_labels = ambient(:, r) + country(:, r) * 2 + metal(:, r) * 3 + rocknroll(:, r) * 4 + symphonic(:, r) * 5;
    
        % testing the error rate left out run
        test_predicted_labels = predict(discr_obj, test_predictors);
        test_percent_of_correctness(r) = 100*sum(test_labels == test_predicted_labels)/length(test_labels);
        disp(sprintf('percent of correct predictions using fitcdiscr (num of voxels= %d): %.3f%%', num_of_voxels, test_percent_of_correctness(r)));

    end
    cv_accuracy(i) = mean(test_percent_of_correctness);
end
%%
[max_acc, max_acc_indeces] = max(cv_accuracy);
disp(sprintf('max accuracy: %.2f%% , error bound: %.2f  for num. of voxels: %d', max_acc, std(cv_accuracy)/sqrt(6), nvoxels(max_acc_indeces))) 

figure
hold on
plot(log10(nvoxels), cv_accuracy)
plot(log10(nvoxels), cv_accuracy + std(cv_accuracy)/sqrt(6), 'r--')
plot(log10(nvoxels), cv_accuracy - std(cv_accuracy)/sqrt(6), 'r--')
title('cross val. mean accuracy vs num. of informative voxels')
xlabel('log num. of voxels')
ylabel('%')

%% ======== Q7 ========
addpath 'msvm'
addpath 'logitReg'

%% ======== Q8 ========
% rerunning for best cross val. accuracy
num_of_voxels = nvoxels(max_acc_indeces);
confusion = zeros(5);

for r = 1 : 7
    load(sprintf('p_vals_cv_%d.mat', r));
    allowed_run = [(1 : r-1) (r+1 : 7)];
    
    % feature reduction by choosing discriminative voxels
    d_voxels_indeces = zeros(size(p_vals_cv));
    [sorted_p_vals, sorted_indeces] = sort(p_vals_cv, 'ascend');
    d_voxels_indeces(sorted_indeces(1 : num_of_voxels)) = 1;
    d_voxels_indeces = logical(d_voxels_indeces);
    
    d_imgs = reshape(imgs(:, allowed_run, d_voxels_indeces), [size(imgs,1)*length(allowed_run) sum(d_voxels_indeces)]);
    
    % feature reduction by PCA (just to reduce feature's space dimension to
    % number of observations)
    [coeff, pca_d_imgs, latent] = pca(d_imgs);
    
    % generating labels
    labels = reshape(ambient(:, allowed_run), [numel(ambient(:, allowed_run)) 1]) + ...
    reshape(country(:, allowed_run), [numel(country(:, allowed_run)) 1]) * 2 + ...
    reshape(metal(:, allowed_run), [numel(metal(:, allowed_run)) 1]) * 3 + ...
    reshape(rocknroll(:, allowed_run), [numel(rocknroll(:, allowed_run)) 1]) * 4 + ...
    reshape(symphonic(:, allowed_run), [numel(symphonic(:, allowed_run)) 1]) * 5;

    % classification
    predictors = pca_d_imgs(:, 1 : min(20,size(pca_d_imgs,2)) ); 
    discr_obj = fitcdiscr(predictors, labels);

    % generating predictors and labels for left out run
    test_d_imgs = squeeze(imgs(:, r, d_voxels_indeces));
    % applying pca coeff
    test_pca_d_imgs = (test_d_imgs - repmat(mean(d_imgs), [size(test_d_imgs,1) 1]) ) * coeff ;
    test_predictors = test_pca_d_imgs(:, 1 : size(predictors,2) );
    
    test_labels = ambient(:, r) + country(:, r) * 2 + metal(:, r) * 3 + rocknroll(:, r) * 4 + symphonic(:, r) * 5;
    
    % testing the error rate left out run
    test_predicted_labels = predict(discr_obj, test_predictors);
    
    % filling confusion 
    for j = 1 : length(test_labels)
        confusion(test_labels(j), test_predicted_labels(j)) = confusion(test_labels(j), test_predicted_labels(j)) + 1;
    end
    
end

%% ======== Q9 ========
% rerunning Q2
num_of_voxels = nvoxels(max_acc_indeces);

load('p_vals.mat');
allowed_run = 1 : 7;
    
% feature reduction by choosing discriminative voxels
d_voxels_indeces = zeros(size(p_vals));
[sorted_p_vals, sorted_indeces] = sort(p_vals, 'ascend');
d_voxels_indeces(sorted_indeces(1 : num_of_voxels)) = 1;
d_voxels_indeces = logical(d_voxels_indeces);
    
d_imgs = reshape(imgs(:, allowed_run, d_voxels_indeces), [size(imgs,1)*length(allowed_run) sum(d_voxels_indeces)]);
    
% feature reduction by PCA (just to reduce feature's space dimension to
% number of observations)
[coeff, pca_d_imgs, latent] = pca(d_imgs);
    
% generating labels
labels = reshape(ambient(:, allowed_run), [numel(ambient(:, allowed_run)) 1]) + ...
    reshape(country(:, allowed_run), [numel(country(:, allowed_run)) 1]) * 2 + ...
    reshape(metal(:, allowed_run), [numel(metal(:, allowed_run)) 1]) * 3 + ...
    reshape(rocknroll(:, allowed_run), [numel(rocknroll(:, allowed_run)) 1]) * 4 + ...
    reshape(symphonic(:, allowed_run), [numel(symphonic(:, allowed_run)) 1]) * 5;

% classification
predictors = pca_d_imgs(:, 1 : min(20,size(pca_d_imgs,2)) ); % 20 chose experimentally!
discr_obj = fitcdiscr(predictors, labels);

% generating predictors for test run
test_d_imgs = squeeze(imgs(:, 8, d_voxels_indeces));
% applying pca coeff
test_pca_d_imgs = (test_d_imgs - repmat(mean(d_imgs), [size(test_d_imgs,1) 1]) ) * coeff ;
test_predictors = test_pca_d_imgs(:, 1 : size(predictors,2) );
        
% prediction
test_predicted_labels = predict(discr_obj, test_predictors);
    
 
%% =============== Part 5 ==================
figure
loglog(latent)
title('latents')
%%
idx = kmeans(predictors, 25);
idx_genre = nan(25, 1);
for i = 1 : 25
    idx_genre(i) = mode(labels(idx == i));
end

idx_discr_obj = fitcdiscr(predictors, idx);

idx_predicted_labels = predict(idx_discr_obj, predictors);
idx_percent_of_correctness = 100*sum(idx == idx_predicted_labels)/length(idx);
disp(sprintf('percent of correct predictions using fitcdiscr : %.3f%%', percent_of_correctness));

%%
% cross validation!
nvoxels = floor(logspace(0,3.5,30));
cv_accuracy = nan(size(nvoxels));

for i = 1 : length(nvoxels)
    num_of_voxels = nvoxels(i);
    for r = 1 : 7
        load(sprintf('p_vals_cv_%d.mat', r));
        allowed_run = [(1 : r-1) (r+1 : 7)];
    
        % feature reduction by choosing discriminative voxels
        d_voxels_indeces = zeros(size(p_vals_cv));
        [sorted_p_vals, sorted_indeces] = sort(p_vals_cv, 'ascend');
        d_voxels_indeces(sorted_indeces(1 : num_of_voxels)) = 1;
        d_voxels_indeces = logical(d_voxels_indeces);
    

        d_imgs = reshape(imgs(:, allowed_run, d_voxels_indeces), [size(imgs,1)*length(allowed_run) sum(d_voxels_indeces)]);

        % feature reduction by PCA (just to reduce feature's space dimension to
        % number of observations)
        [coeff, pca_d_imgs, latent] = pca(d_imgs);
    
        % generating labels
        labels = reshape(ambient(:, allowed_run), [numel(ambient(:, allowed_run)) 1]) + ...
        reshape(country(:, allowed_run), [numel(country(:, allowed_run)) 1]) * 2 + ...
        reshape(metal(:, allowed_run), [numel(metal(:, allowed_run)) 1]) * 3 + ...
        reshape(rocknroll(:, allowed_run), [numel(rocknroll(:, allowed_run)) 1]) * 4 + ...
        reshape(symphonic(:, allowed_run), [numel(symphonic(:, allowed_run)) 1]) * 5;
    
        % classification
        predictors = pca_d_imgs(:, 1 : min(20,size(pca_d_imgs,2)) ); 
        idx = kmeans(predictors, 25);
        
        discr_obj = fitcdiscr(predictors, idx);

        % generating predictors and labels for left out run
        test_d_imgs = squeeze(imgs(:, r, d_voxels_indeces));
        % applying pca coeff
        test_pca_d_imgs = (test_d_imgs - repmat(mean(d_imgs), [size(test_d_imgs,1) 1]) ) * coeff ;
        test_predictors = test_pca_d_imgs(:, 1 : size(predictors,2) );
    
        test_labels = ambient(:, r) + country(:, r) * 2 + metal(:, r) * 3 + rocknroll(:, r) * 4 + symphonic(:, r) * 5;
    
        % testing the error rate left out run
        test_idx_predicted_labels = predict(discr_obj, test_predictors);
        for j = 1 : length(test_idx_predicted_labels)
            test_predicted_labels(j) = idx_genre(test_idx_predicted_labels(j));
        end
        
        test_percent_of_correctness(r) = 100*sum(test_labels == test_predicted_labels)/length(test_labels);
        disp(sprintf('percent of correct predictions using fitcdiscr (num of voxels= %d): %.3f%%', num_of_voxels, test_percent_of_correctness(r)));

    end
    cv_accuracy(i) = mean(test_percent_of_correctness);
end

[max_acc, max_acc_indeces] = max(cv_accuracy);
disp(sprintf('max accuracy: %.2f%% , error bound: %.2f%%  for num. of voxels: %d', max_acc, std(cv_accuracy)/sqrt(6), nvoxels(max_acc_indeces))) 

