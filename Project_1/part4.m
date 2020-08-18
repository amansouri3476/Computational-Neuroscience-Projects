%% ================= Part 4 ================= %%
%% ============= Q1 ============= %%
fig = figure;
fig.OuterPosition = [100 100 800 240];
hold on

neuron_exp_extracted_stimuli = Spk_trg_neuron; % neuron_exp_extracted_stimuli{neuron_no , exp_no} = extracted_stimuli (16 * 16 * N)

spk_trg_stimuli = nan(16 * 16 , 1); % will be 256 * n matrix where n is the total number of stimuli extracted from all neurons and experiments.

for exp_no = 1 : length(neurons{1}{neuron_no})
    extracted_stimuli = neuron_exp_extracted_stimuli{neuron_no , exp_no};
    spk_trg_stimuli = [spk_trg_stimuli , reshape(extracted_stimuli , [16*16 , size(extracted_stimuli , 3)])];
end

spk_trg_stimuli = spk_trg_stimuli(: , 2:end);

% calculation of corr matrix and its eigenvec-val.s
spk_trg_corr = corr(spk_trg_stimuli');
[V,D] = eig(spk_trg_corr); % A*V = V*D

% sorting eigenvalues to determine significant eigenvectors
[sorted_D , sorting_I] = sort(diag(D) , 'descend');

highest_D(neuron_no,:) = sorted_D(sorting_I(1:2));

% displaying significant eigenvectors
min_range = min(min(V)); % ranges for images to be displayed
max_range = max(max(V));
for i = 1 : 3
    subplot(1,4,i)
    imshow(reshape(V(:,sorting_I(i)) , [16 , 16]) , [min_range , max_range] , 'InitialMagnification' , 'fit')
    title(sprintf('v%d' , sorting_I(i)))   
end
subplot(1,4,1)
xlabel('x -->')
ylabel('<-- t')

%% ============= Q2 ============= %%
num_of_ctrl_mat = 5; % times we repeat making control matrices
ctrl_sorted_D = nan(256 , num_of_ctrl_mat); % each col corresponds to sorted eigenvalues of a control correlation matrix

num_of_control_stimuli = size(spk_trg_stimuli , 2); % same to num of spike tiggered stimuli
control_stimuli = nan(256 , num_of_control_stimuli) ; % same to size of spk_trg_stimuli

for i = 1 : num_of_ctrl_mat
    stimuli_indeces = floor((length(stimuli) - 15) * rand(num_of_control_stimuli,1)) + 16;
    counter = 1;
    for stimulus_index = stimuli_indeces'
        control_stimuli(:,counter) = reshape(stimuli(stimulus_index-15 : stimulus_index,:) , [256 , 1]);
        counter = counter + 1;
    end
    
    control_corr = corr(control_stimuli');
    [ctrl_V,ctrl_D] = eig(control_corr); 
    [ctrl_sorted_D(:,i) , ctrl_sorting_I] = sort(diag(ctrl_D) , 'descend');
end

plotting_n = 10; % num of eigenvals to be plotted
subplot(1,4,4)
hold on
stem(sorted_D(1:plotting_n))
plot(1:plotting_n , sorted_D(1:plotting_n) + 10.4 * std(ctrl_sorted_D(1:plotting_n , :) , 0 , 2) , 'r--');
plot(1:plotting_n , sorted_D(1:plotting_n) - 10.4 * std(ctrl_sorted_D(1:plotting_n , :) , 0 , 2) , 'r--');
title('sorted eigenvalues(D)')

fig.PaperPositionMode = 'auto';
% saveas(fig,sprintf('%s_vecs.bmp' , neurons{1}{neuron_no}(1).hdr.FileInfo.Fname(1:12)) )
%% ============= Q4 ============= %%
fig = figure;
fig.OuterPosition = [100 100 800 300];
hold on

v = V(: , sorting_I(1:2)); % 2 eigenvectors with highest eigenvalues 

% projection of spk_trg_stimuli
spk_trg_stimuli_projected = spk_trg_stimuli' * v;

% projection of control_stimuli (comes from Q2 , last run of loop)
control_stimuli_projected = control_stimuli' * v;

%plotting
for i = 1 : size(v , 2)
    subplot(1,3,i)
    hold on
    histogram(spk_trg_stimuli_projected(: , i) , linspace(-10 , 10 , 35));
    histogram(control_stimuli_projected(: , i) , linspace(-10 , 10 , 35));
    axis([-10 10 0 6000])
    legend('spk-trg' , 'control' , 'Location' , 'Northwest')
    title(sprintf('projected on v%d' , i))
end


%%
subplot(1,3,3)
hold on
h1 = hist3([spk_trg_stimuli_projected(: , 1) , spk_trg_stimuli_projected(: , 2)] , [15 15] , 'FaceColor', 'red' );
h2 = hist3([control_stimuli_projected(: , 1) , control_stimuli_projected(: , 2)] , [15 15] , 'FaceColor', 'blue');

%%
imshow(cat(3 , h1 , h1*0 , h2)/max(max(h2)) , 'InitialMagnification' , 'fit')
title('joint distribution (red:spk-trg , blue:control)')
xlabel('v2')
ylabel('v1')

fig.PaperPositionMode = 'auto';
% saveas(fig,sprintf('%s_dist.bmp' , neurons{1}{neuron_no}(1).hdr.FileInfo.Fname(1:12)))
%% ============= Q5 ============= %%
% assume v (defined in previous section) contains meaningful eigenvectors!

% assume distribution of projected stimuli are jointly normal, we fit a
% normal dist. to each demension

test_train_ratio = 0.1;
test_stimuli_indeces = floor(rand(floor(size(spk_trg_stimuli_projected , 1) * test_train_ratio) , 1) * size(spk_trg_stimuli_projected , 1)) + 1;
train_stimuli_indeces = (1:size(spk_trg_stimuli_projected , 1))';
train_stimuli_indeces(test_stimuli_indeces) = [];
for i = 1 : size(v , 2)
    spk_trg_pd(i) = fitdist(spk_trg_stimuli_projected(train_stimuli_indeces , i) , 'normal');
    control_pd(i) = fitdist(control_stimuli_projected(train_stimuli_indeces , i) , 'normal');
end

dist(neuron_no) = sqrt((spk_trg_pd(1).mu - control_pd(1).mu)^2 + (spk_trg_pd(2).mu - control_pd(2).mu)^2);
disp(sprintf('%d- spk_trg( %.3f , %.3f ) , ctrl ( %.3f , %.3f ) >>>>> dist = %.2f\n' , neuron_no , spk_trg_pd(1).mu , spk_trg_pd(2).mu , control_pd(1).mu , control_pd(2).mu , dist(neuron_no)));

% testing how well random stimuli be categorized
%p_spk_trg = size(spk_trg_stimuli , 2)/length(neurons{1}{neuron_no})/size(stimuli , 1);
p_spk_trg = 0.5;
p_control = 1 - p_spk_trg; % approximately

num_of_tests = 2000;
num_of_failure = 0;
for i = 1 : num_of_tests
    random_index = test_stimuli_indeces(floor(length(test_stimuli_indeces) * rand(1,1)) + 1); % chose from test_indeces
    
    if (rand(1,1) < p_spk_trg)
        predicted_dist = which_dist_is_more_probable(control_pd , spk_trg_pd , p_control , p_spk_trg , spk_trg_stimuli_projected(random_index , :));
        num_of_failure = num_of_failure + (2 - predicted_dist);
    else
        predicted_dist = which_dist_is_more_probable(control_pd , spk_trg_pd , p_control , p_spk_trg , control_stimuli_projected(random_index , :));
        num_of_failure = num_of_failure + (predicted_dist - 1);
    end
    
    %disp(sprintf('%.1f%% of tests done, failures: %.1f%%' , i/num_of_tests*100 , num_of_failure/i*100))
end
disp(sprintf('neuron: %s , %.1f%% failed' , neurons{1}{neuron_no}(1).hdr.FileInfo.Fname(1:10) , num_of_failure/num_of_tests*100))

%% ============= exrta ============= %%
r_v_1 = reshape(v(:,1) , [16 16]);
r_v_2 = reshape(v(:,2) , [16 16]);

spatial_corr_coeff(neuron_no,1) = sum(sum(r_v_1(:,2:end) .* r_v_1(:,1:end-1)))/(v(:,1)'*v(:,1));
temporal_corr_coeff(neuron_no,1) = sum(sum(r_v_1(2:end,:) .* r_v_1(1:end-1,:)))/(v(:,1)'*v(:,1));
spatial_corr_coeff(neuron_no,2) = sum(sum(r_v_2(:,2:end) .* r_v_2(:,1:end-1)))/(v(:,2)'*v(:,2));
temporal_corr_coeff(neuron_no,2) = sum(sum(r_v_2(2:end,:) .* r_v_2(1:end-1,:)))/(v(:,2)'*v(:,2));

ctrl_v = ctrl_V(:,sorting_I(1));
r_ctrl_v = reshape(ctrl_v , [16 16]);
ctrl_spatial_corr_coeff(neuron_no) = sum(sum(r_ctrl_v(:,2:end) .* r_ctrl_v(:,1:end-1)))/(ctrl_v'*ctrl_v);
ctrl_temporal_corr_coeff(neuron_no) = sum(sum(r_ctrl_v(2:end,:) .* r_ctrl_v(1:end-1,:)))/(ctrl_v'*ctrl_v);
