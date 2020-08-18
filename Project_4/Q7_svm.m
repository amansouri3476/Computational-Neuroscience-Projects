%% ======== Q2 & 3 & 5 ========
%load('p_vals.mat')
% generating labels
labels = reshape(ambient(:, 1:7), [numel(ambient(:, 1:7)) 1]) + ...
reshape(country(:, 1:7), [numel(country(:, 1:7)) 1]) * 2 + ...
reshape(metal(:, 1:7), [numel(metal(:, 1:7)) 1]) * 3 + ...
reshape(rocknroll(:, 1:7), [numel(rocknroll(:, 1:7)) 1]) * 4 + ...
reshape(symphonic(:, 1:7), [numel(symphonic(:, 1:7)) 1]) * 5;

ths = logspace(-4, -2, 50);
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
    svm_obj = svm.train(predictors, labels);

    % testing the error rate on prediction
    predicted_labels = svm.predict(svm_obj, predictors);
    percent_of_correctness(i) = 100*sum(labels == predicted_labels)/length(labels);
    disp(sprintf('percent of correct predictions using fitcdiscr (p_val th = %.3f): %.3f%%', th, percent_of_correctness(i)));
end

figure
loglog(ths, percent_of_correctness)
title('prediction accuracy vs ANOVA1 th')
xlabel('th')
ylabel('%')

%% ======== Q4 & 5 ========
% cross validation!
nvoxels = floor(logspace(0,4,10));
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
        svm_obj = svm.train(predictors, labels);

        % generating predictors and labels for left out run
        test_d_imgs = squeeze(imgs(:, r, d_voxels_indeces));
        % applying pca coeff
        test_pca_d_imgs = (test_d_imgs - repmat(mean(d_imgs), [size(test_d_imgs,1) 1]) ) * coeff ;
        test_predictors = test_pca_d_imgs(:, 1 : size(predictors,2) );
    
        test_labels = ambient(:, r) + country(:, r) * 2 + metal(:, r) * 3 + rocknroll(:, r) * 4 + symphonic(:, r) * 5;
    
        % testing the error rate left out run
        test_predicted_labels = svm.predict(svm_obj, test_predictors);
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
