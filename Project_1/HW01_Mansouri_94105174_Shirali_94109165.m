%% ======================= HW1 ======================= %%
% attention:
% - Some parts of this code were written on a MAC system and it has been
% checked on both MAC and windows (in case something unusual occures!)
% - all files related to data must be in the same directory as the this
% mfile. Folders 'Data' and 'MatlabFunctions' need to be in the same
% directory as mfile as well as the following mfiles:
% fload_log, fretrieve_log, fsearch_log, Func_ReadData,
% Func_StimuliExtraction, part4, str2cell, tori, tview,
% which_dist_is_more_probable

% - some abbreviations used in parameters naming:
%   spk: spike
%   trg: triggered 
%   ctrl: control
% - run the sections successively and avoid running them multiple times!
% - some parts need long time to be executed, be patient please!

%% ================= Part 2 ================= %%
%% ============= Q3 ============= %%
clc
close all

neurons_dir = dir('Data/Spike_and_Log_Files'); % You need to open this mfile where the
% folder 'Data' exists

num_of_neurons = 61;

neurons = {};

spike_count_rate = nan(61,1);



for i = length(neurons_dir) : -1 : length(neurons_dir) - num_of_neurons + 1 % avoiding redundant folders not related to the neurons
    events_hdrs = Func_ReadData(neurons_dir(i).name);
    
    neurons{i - length(neurons_dir) + num_of_neurons} = events_hdrs;
    
    spk_c_r = nan(size(events_hdrs));
    for j = 1 : length(spk_c_r)
        spk_c_r(j) = length(events_hdrs(j).events) / events_hdrs(j).events(end) * 10000; % time rescale to second
    end
    
    spike_count_rate(i) = mean(spk_c_r);
end

%% Extracting Frame Rates

for i = length(neurons_dir) : -1 : length(neurons_dir) - num_of_neurons + 1 % avoiding redundant folders not related to the neurons

    address = neurons_dir(i).name;
    cd (sprintf('Data/Spike_and_Log_Files/%s',address));

    file_dirs = dir;
    flag = 0;
    for j = 1:length(file_dirs)
        file_name = file_dirs(j).name; 
        if (~isempty(strfind(lower(file_name) , 'msq1d.log')) && flag ~= 1)
            file_log = importdata(sprintf('%s',file_dirs(j).name,''));
            frame_rate(i - length(neurons_dir) + num_of_neurons) = str2double(file_log.textdata{14}(10:end));
            flag = 1;
        end
    end
    cd ..//..//..//;
end


%% ============= Q3 ============= %%
hist(spike_count_rate , 10)
xlabel('Firing Rate','color','b')
ylabel('Number of Neurons','color','b')
title('Neurons'' Firing Rate Histogram','color','r')

disp(sprintf('# low firing rate neurons (r < 2) : %d \n' , sum(spike_count_rate < 2)));

for i = find(spike_count_rate < 2)'
    disp(neurons{i}(1).hdr.FileInfo.Fname(1:10))
end

% If you run the following code more than once, you'll receive the error
% :'Index exceeds matrix dimensions.' which is not important since there
% are no more low firing rate neurons and the index which 'find' returns,
% is no longer existing in the neurons cell and frame_rate vector. Hence
% index exceeds matrix dimension.
neurons(find(spike_count_rate < 2)) = []; % Removing low spike rates data from the cell 'neurons'
frame_rate(find(spike_count_rate < 2))=[];
neurons = {neurons frame_rate}; % Constructing new cell including frame rates
%% ============= Q4 ============= %%

stimuli = load('Data/Stimulus_Files/msq1D.mat');
stimuli = stimuli.msq1D;

% We just load the stimuli here and the desired function is attached.
%% ============= Q5 ============= %%
% Attention : Please copy the following .m files from MatlabFunctions/tview
% to the main search path. There was a problem for us using addpath and it
% didn't work properly

% Please copy : str2cell.m, tori.m, tview.m, fload_log.m, fretrieve_log.m,
% fsearch_log.m

clc
close all
addpath MatlabFunctions/tview % adding this path to search to use its functions
% First Neuron
cd Data/Spike_and_Log_Files/000503.a03

flog1 = tview('000503.a03atune.log');

cd ..//..//..//

% Second Neuron
cd Data/Spike_and_Log_Files/000601.c05

flog2 = tview('000601.c05atune.log') ;

cd ..//..//..//

% Third Neuron
cd Data/Spike_and_Log_Files/011025.A.d07

flog3 = tview('011025.A.d07atune.log') ;

cd ..//..//..//

%% ================= Part 3 ================= %%
%% ============= Q1 ============= %%
close all
clc

% We find STA for each neuron and each experiment on each neuron

Spk_trg_neuron = {};
STA_Exp = {};
STA = {};
for i = 1 : length(neurons{1})
    
   for j = 1 : length(neurons{1}{i})
        Spk_trg_neuron{i,j} = Func_StimuliExtraction(neurons{1}{i}(j).events,stimuli,i,frame_rate);
        STA_Exp{i,j} = mean(Spk_trg_neuron{i,j},3); % STA for each experiment of each of neurons above firing rate threshold
   end
   
   STA{i,1} = STA_Exp{i,1};
   
   for j = 2 : length(neurons{1}{i})
        STA{i,1} = STA{i,1} + STA_Exp{i,j};
   end
   
   STA{i,1} = STA{i,1}/length(neurons{1}{i});
end

%% ============= Q1(continued) ============= %%
clc
close all

% We have Chosen neuron #20 (After removing subthreshold firing rate
% neurons) whose name is '000802.c06'

chosen_neuron = 20;

fprintf('Number of Experiments on This Neuron : %d\n',length(neurons{1}{chosen_neuron}));
figure
imshow(STA{chosen_neuron},[min(min(STA{chosen_neuron})), max(max(STA{chosen_neuron}))],'InitialMagnification','fit')
xlabel('Spatial','color','b')
ylabel('Temporal','color','b')
title('STA - 000802.c06','color','r')

%% ============= Q2 ============= %%
clc
close all
chosen_neuron_STA_elements = {};
numOfNeuron = chosen_neuron;
for i = 1 : 16

    for j = 1 : 16
           element_experiment_var = [];
                for experimentNo = 1 : length(neurons{1}{numOfNeuron})
                       experimentVect = zeros(length(Spk_trg_neuron{numOfNeuron,experimentNo}(i,j,:)),1);
                       experimentVect(:,1) = Spk_trg_neuron{numOfNeuron,experimentNo}(i,j,:);
                       element_experiment_var = vertcat(element_experiment_var,experimentVect);
                end
          chosen_neuron_STA_elements{i,j} = element_experiment_var;


     end

end

% Applying ttest to each element of our 16*16 'chosen_neuron_STA_elements'
% cell
for i = 1 : 16
    
    for j = 1 : 16
        
        [H(i,j),P(i,j)] = ttest(chosen_neuron_STA_elements{i,j});
        
    end
    
end
figure
imshow(P,[min(min(P)),max(max(P))],'InitialMagnification','fit')
xlabel('Spatial','color','b')
ylabel('Temporal','color','b')
title('P Value - 000802.c06','color','r')

%% ============= Q3 ============= %%

% Constructing control stimuli

clc
close all
numOfControlStimuli = length(Spk_trg_neuron{numOfNeuron,1});
control_stimuli = {};
stimuli_index = floor((length(stimuli) - 15) * rand(numOfControlStimuli,1)) + 16;
for i = 1 : length(stimuli_index)
        control_stimuli{i,1} = stimuli(stimuli_index(i,1)-15:stimuli_index(i,1),:);
end


% Stimuli Projection on STA distribution for both spike-triggered stimuli
% and control stimuli for the chosen neuron(neuron #20)


% We will use first experiments spike-triggered-stimuli and the above
% control stimuli for projection on STA

% Spike-triggered stimuli projection

% Reshaping STA for dot product in 256-Dimensional Space
STA_reshaped = reshape(STA{numOfNeuron},[256 1]);
for i = 1 : length(Spk_trg_neuron{numOfNeuron,1})
    
        % Reshaping for inner product
        Spk_trg_neuron_reshaped = reshape(Spk_trg_neuron{numOfNeuron,1}(:,:,i),[256 1]);
        
        spk_trg_Projection_on_STA(i,1) = STA_reshaped'*(Spk_trg_neuron_reshaped);
end
        
% Control stimuli projection

for i = 1 : numOfControlStimuli
    
        % Reshaping for inner product
        control_stimuli_reshaped = reshape(control_stimuli{i,1},[256 1]);
        
        control_stimuli_Projection_on_STA(i,1) = STA_reshaped'*(control_stimuli_reshaped); 
end

% Plotting Histograms


H1 = histogram(control_stimuli_Projection_on_STA);
hold on
H2 = histogram(spk_trg_Projection_on_STA);
legend('Control','Spike','Location','Northeast')
H1.Normalization = 'probability';
H1.BinWidth = 0.1;
H2.Normalization = 'probability';
H2.BinWidth = 0.1;

%% ============= Q4 ============= %%
clc
close all
[H,P] = ttest2(sort(control_stimuli_Projection_on_STA),sort(spk_trg_Projection_on_STA));
%% ============= Q5 ============= %%
clc
close all
pd_control = fitdist(control_stimuli_Projection_on_STA,'normal');
pd_spk_trg_projection = fitdist(spk_trg_Projection_on_STA,'normal');
x_values = -1:0.01:1;

plot(x_values,pdf(pd_control,x_values))
hold on
plot(x_values,pdf(pd_spk_trg_projection,x_values))
legend('Control','Spike-Triggered')
xlabel('Projection on STA','color','b')
ylabel('f(x)','color','b')
title('Distributions of two stimuli types','color','r')

% Finding Threshold

Threshold = x_values(min(find(pdf(pd_spk_trg_projection,x_values)>pdf(pd_control,x_values))));
fprintf('Projection on STA Threshold to spike is: %d\n',Threshold);

sample_stimuli_projection = vertcat(control_stimuli_Projection_on_STA,spk_trg_Projection_on_STA);

counter = 0;
for i = 1 : length(sample_stimuli_projection)
    
   if sample_stimuli_projection(i)>Threshold && length(control_stimuli_Projection_on_STA)+1<i<length(sample_stimuli_projection)
       counter = counter+1;
   end
   if sample_stimuli_projection(i)<Threshold && i<length(control_stimuli_Projection_on_STA)
       counter = counter+1;
   end
    
end
correct_prediction_prob = 100*counter/length(sample_stimuli_projection);
fprintf('Correct Prediction Probability is: %d\n',correct_prediction_prob);
%% ============= Q6 ============= %%
% Included in report
%% Generalizing Q1,Q2,Q3
clc
close all

% We will use first experiments spike-triggered-stimuli and the above
% control stimuli for projection on STA


control_stimuli_Projection_on_STA = {};
spk_trg_Projection_on_STA = {};

for i = 1 : length(neurons{1})
    disp(i) % Time neeed for this section is long! So we use this counter to see how close we are to the end!
    figure('units','normalized','outerposition',[0 0 1 1])% Showing in full Screen
    
    % Constructing control stimuli proper for each neuron
    
    numOfControlStimuli = length(Spk_trg_neuron{i,1});
    control_stimuli = {};
    stimuli_index = floor((length(stimuli) - 15) * rand(numOfControlStimuli,1)) + 16;
    for r = 1 : length(stimuli_index)
            control_stimuli{r,1} = stimuli(stimuli_index(r,1)-15:stimuli_index(r,1),:);
    end
    k = i;
    % Plotting STA for each neuron
    subplot(1,3,1)
    imshow(STA{i},[min(min(STA{i})), max(max(STA{i}))],'InitialMagnification','fit')
    xlabel('Spatial','color','b','FontSize',15)
    ylabel('Temporal','color','b','FontSize',15)
    if k<=27
            title(sprintf('STA Neuron : %s',neurons{1}{i}(1).hdr.FileInfo.Fname(1:10)),'color','r','FontSize',15)
    else
            title(sprintf('STA Neuron : %s',neurons{1}{i}(1).hdr.FileInfo.Fname(1:12)),'color','r','FontSize',15)
    end
            

    chosen_neuron_STA_elements = {};
                    for m = 1 : 16

                        for n = 1 : 16
                               element_experiment_var = [];
                                    for experimentNo = 1 : length(neurons{1}{k})
                                           experimentVect = zeros(length(Spk_trg_neuron{k,experimentNo}(m,n,:)),1);
                                           experimentVect(:,1) = Spk_trg_neuron{k,experimentNo}(m,n,:);
                                           element_experiment_var = vertcat(element_experiment_var,experimentVect);
                                    end
                              chosen_neuron_STA_elements{m,n,k} = element_experiment_var;


                         end

                    end

            % Applying ttest to each element of our 16*16 'chosen_neuron_STA_elements'
            % cell

            H = [];
            P = [];
                for m = 1 : 16

                        for n = 1 : 16

                                [H(m,n,k),P(m,n,k)] = ttest(chosen_neuron_STA_elements{m,n,k});

                        end

                end
            

            % Showing P-value images
            subplot(1,3,2)
            imshow(P(:,:,i))
            xlabel('Spatial','color','b','FontSize',15)
            ylabel('Temporal','color','b','FontSize',15)
            if k<=27
                    title(sprintf('P Value Neuron : %s',neurons{1}{i}(1).hdr.FileInfo.Fname(1:10)),'color','r','FontSize',15)
            else
                    title(sprintf('P Value Neuron : %s',neurons{1}{i}(1).hdr.FileInfo.Fname(1:12)),'color','r','FontSize',15)
            end
            
            % Spike-triggered stimuli projection

            % Reshaping STA for dot product in 256-Dimensional Space (for all neurons)
            STA_reshaped = {};
                        
            STA_reshaped{k,1} = reshape(STA{k},[256 1]);


            spk_trg_Projection_on_STA{k,1} = [];
            for q = 1 : length(Spk_trg_neuron{k,1}(:,:,:)) % using first experiment
                                
                % Reshaping for inner product
                                
                Spk_trg_neuron_reshaped = reshape(Spk_trg_neuron{k,1}(:,:,q),[256 1]);

                spk_trg_Projection_on_STA{k,1} = [spk_trg_Projection_on_STA{k,1};(STA_reshaped{k,1})'*(Spk_trg_neuron_reshaped)];
            end
            
            % Control stimuli projection
            
            control_stimuli_Projection_on_STA{k,1} = [];        
            for q = 1 : numOfControlStimuli

                                
                % Reshaping for inner product
                                
                control_stimuli_reshaped = reshape(control_stimuli{q,1},[256 1]);

                                
                control_stimuli_Projection_on_STA{k,1} = [control_stimuli_Projection_on_STA{k,1};(STA_reshaped{k,1})'*(control_stimuli_reshaped)]; 
            end
           
            
            % Plotting Histograms

                        
            subplot(1,3,3)
                        
            H1 = histogram(control_stimuli_Projection_on_STA{k,1});
            hold on
                        
            H2 = histogram(spk_trg_Projection_on_STA{k,1});
            legend('Control','Spike')
                        
            H1.Normalization = 'probability';
                        
            H1.BinWidth = 0.1;
                        
            H2.Normalization = 'probability';
                        
            H2.BinWidth = 0.1;

end
%% Generalizing Q1

% You can proceed to next section. The following code is not used in
% other section.

% Viewing all receptive fields together


clc
close all

figure('units','normalized','outerposition',[0 0 1 1])% Showing in full Screen
for i = 1 : length(neurons{1})
        subplot(9,6,i)
        imshow(STA{i},[min(min(STA{i})), max(max(STA{i}))],'InitialMagnification','fit')
        title(sprintf('#%d',i),'color','r','FontSize',5)
end
%% Generalizing Q2

% You can proceed to next section. The following code is not used in
% other section.

% Viewing all P values together

clc
close all
chosen_neuron_STA_elements = {};
for k = 1 : length(neurons{1})
        for i = 1 : 16

            for j = 1 : 16
                   element_experiment_var = [];
                        for experimentNo = 1 : length(neurons{1}{k})
                               experimentVect = zeros(length(Spk_trg_neuron{k,experimentNo}(i,j,:)),1);
                               experimentVect(:,1) = Spk_trg_neuron{k,experimentNo}(i,j,:);
                               element_experiment_var = vertcat(element_experiment_var,experimentVect);
                        end
                  chosen_neuron_STA_elements{i,j,k} = element_experiment_var;


             end

        end
end

% Applying ttest to each element of our 16*16 'chosen_neuron_STA_elements'
% cell

H = [];
P = [];
for k = 1 : length(neurons{1})
        for i = 1 : 16

            for j = 1 : 16

                [H(i,j,k),P(i,j,k)] = ttest(chosen_neuron_STA_elements{i,j,k});

            end

        end
end

% Showing P-value images
figure('units','normalized','outerposition',[0 0 1 1])% Showing in full Screen
for i = 1 : length(neurons{1})
    subplot(9,6,i)
    imshow(P(:,:,i))
    title(sprintf('#%d',i),'color','r','FontSize',5)
end

%% Generalizing Q4

clc
close all
for i = 1 : length(neurons{1})
            [H_Gen(1,i),P_Gen(1,i)] = ttest2(control_stimuli_Projection_on_STA{i,1},spk_trg_Projection_on_STA{i,1});
end

Names = {};

for i = 1 : length(neurons{1})
    if i<=27
             Names{i,1} = neurons{1}{i}(1).hdr.FileInfo.Fname(1:10);
    else
             Names{i,1} = neurons{1}{i}(1).hdr.FileInfo.Fname(1:12);
    end
    
end


Table = table(H_Gen',P_Gen','RowNames',Names)

%% Generalizing Q5
clc
close all
for k = 1 : length(neurons{1})
            pd_control = fitdist(control_stimuli_Projection_on_STA{k,1},'normal');
            pd_spk_trg_projection = fitdist(spk_trg_Projection_on_STA{k,1},'normal');
            x_values = -1:0.01:1;

            % Finding Threshold

            Threshold(k,1) = x_values(min(find(pdf(pd_spk_trg_projection,x_values)>pdf(pd_control,x_values))));

            sample_stimuli_projection = vertcat(control_stimuli_Projection_on_STA{k,1},spk_trg_Projection_on_STA{k,1});

            counter = 0;
            for i = 1 : length(sample_stimuli_projection)

               if sample_stimuli_projection(i)>Threshold(k,1) && length(control_stimuli_Projection_on_STA{k,1})+1<i<length(sample_stimuli_projection)
                   counter = counter+1;
               end
               if sample_stimuli_projection(i)<Threshold(k,1) && i<length(control_stimuli_Projection_on_STA{k,1})
                   counter = counter+1;
               end

            end

            correct_prediction_prob(k,1) = 100*counter/length(sample_stimuli_projection);
            
end

Table = table(Threshold,correct_prediction_prob,'RowNames',Names)

%% ============= Q7 ============= %%
% included in report
%% ================= Part 4 ================= %%
%% ============= Q1 ============= %%
figure

neuron_exp_extracted_stimuli = Spk_trg_neuron; % neuron_exp_extracted_stimuli{neuron_no , exp_no} = extracted_stimuli (16 * 16 * N)

spk_trg_stimuli = nan(16 * 16 , 1); % will be 256 * n matrix where n is the total number of stimuli extracted from all experiments.

%neuron_no = chosen_neuron;
neuron_no = 18;

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

% displaying significant eigenvectors
min_range = min(min(V)); % ranges for images to be displayed
max_range = max(max(V));
for i = 1 : 3
    subplot(1,3,i)
    imshow(reshape(V(:,sorting_I(i)) , [16 , 16]) , [min_range , max_range] , 'InitialMagnification' , 'fit')
    title(sprintf('v%d' , sorting_I(i)))   
end
subplot(1,3,1)
xlabel('x -->')
ylabel('<-- t')

%% ============= Q2 ============= %%
figure
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

plotting_n = length(sorted_D); % num of eigenvals to be plotted
hold on
stem(sorted_D(1:plotting_n))
plot(1:plotting_n , sorted_D + 20 * std(ctrl_sorted_D(1:plotting_n , :) , 0 , 2) , 'r--');
plot(1:plotting_n , sorted_D - 20 * std(ctrl_sorted_D(1:plotting_n , :) , 0 , 2) , 'r--');
title('sorted eigenvalues(D)')
%% ============= Q4 ============= %%
figure

v = V(: , sorting_I(1:2)); % 2 eigenvectors with highest eigenvalues 

% projection of spk_trg_stimuli
spk_trg_stimuli_projected = spk_trg_stimuli' * v;

% projection of control_stimuli (comes from Q2 , last run of loop)
control_stimuli_projected = control_stimuli' * v;

%plotting
for i = 1 : size(v , 2)
    subplot(1,2,i)
    hold on
    histogram(spk_trg_stimuli_projected(: , i) , linspace(-10 , 10 , 35));
    histogram(control_stimuli_projected(: , i) , linspace(-10 , 10 , 35));
    axis([-10 10 0 6000])
    legend('spk-trg' , 'control' , 'Location' , 'Northwest')
    title(sprintf('projected on v%d' , i))
end


%%
figure
hold on
hist3([spk_trg_stimuli_projected(: , 1) , spk_trg_stimuli_projected(: , 2)] , [15 15] , 'FaceColor', 'red' )
hist3([control_stimuli_projected(: , 1) , control_stimuli_projected(: , 2)] , [15 15] , 'FaceColor', 'blue')
title('joint distribution (red:spk-trg , blue:control)')
xlabel('v2')
ylabel('v1')

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


%% ============= Q6 ============= %%

for neuron_no = 1 : length(neurons{1})
    part4;
    close all;
end

%% ============= Q7 ============= %%
% temporal and spatial correlations have been calculated in part4.m for
% each neuron

% lets do it for stas!
for i = 1 : size(STA,1)
    s = STA{i};
    sta_temporal_corr_coeff(i) = sum(sum(s(2:end , :) .* s(1:end-1 , :))) / sum(sum(s .* s));
    sta_spatial_corr_coeff(i) = sum(sum(s(: , 2:end) .* s(: , 1:end-1))) / sum(sum(s .* s));
end
%%
% comparing v1 , v2 , ctrl
figure
hold on
h1 = histogram((temporal_corr_coeff(:,1)) , 10);
h2 = histogram((temporal_corr_coeff(:,2)) , 10);
h3 = histogram((ctrl_temporal_corr_coeff) , 10);
h1.Normalization = 'probability';
h2.Normalization = 'probability';
h3.Normalization = 'probability';

legend('v1' , 'v2' , 'ctrl')
title('temporal correlation')

figure
hold on
h1 = histogram((spatial_corr_coeff(:,1)) , 15);
h2 = histogram((spatial_corr_coeff(:,2)) , 15);
h3 = histogram((ctrl_spatial_corr_coeff) , 15);
h1.Normalization = 'probability';
h2.Normalization = 'probability';
h3.Normalization = 'probability';
legend('v1' , 'v2' , 'ctrl')
title('spatial correlation')

%%
% comparing sta , eigenvectors , ctrl
figure
hold on
h1 = histogram((temporal_corr_coeff(:)) , 10);
h2 = histogram((sta_temporal_corr_coeff) , 10);
h3 = histogram((ctrl_temporal_corr_coeff) , 10);
h1.Normalization = 'probability';
h2.Normalization = 'probability';
h3.Normalization = 'probability';

legend('v1 , v2' , 'STA' , 'ctrl')
title('temporal correlation')

figure
hold on
h1 = histogram((spatial_corr_coeff(:)) , 15);
h2 = histogram((sta_spatial_corr_coeff) , 15);
h3 = histogram((ctrl_spatial_corr_coeff) , 15);
h1.Normalization = 'probability';
h2.Normalization = 'probability';
h3.Normalization = 'probability';

legend('v1 , v2' , 'STA' , 'ctrl')
title('spatial correlation')
%%
% spatial-temporal correlation
figure
hold on
plot(spatial_corr_coeff(:,1) , temporal_corr_coeff(:,1) , 'b*')
plot(spatial_corr_coeff(:,2) , temporal_corr_coeff(:,2) , 'r*')
plot(sta_spatial_corr_coeff , sta_temporal_corr_coeff , 'kx')
plot(ctrl_spatial_corr_coeff , ctrl_temporal_corr_coeff , 'g^')
xlabel('spatial')
ylabel('temporal')
legend('v1','v2','sta','ctrl')

%%
% v1 , v2 correlation
figure
hold on
plot(spatial_corr_coeff(:,1),spatial_corr_coeff(:,2) , 'r*')
plot(temporal_corr_coeff(:,1),temporal_corr_coeff(:,2) , 'bx')
legend('spatial','temporal','Location','Northwest')
xlabel('v1')
ylabel('v2')

%%
% v1 , v2 correlation with eigenvalues
figure
hold on
plot(spatial_corr_coeff(:,1) , sum(highest_D , 2) , 'b*')
plot(temporal_corr_coeff(:,1) , sum(highest_D , 2) , 'r*')
plot(spatial_corr_coeff(:,2) , sum(highest_D , 2) , 'k*')
plot(temporal_corr_coeff(:,2) , sum(highest_D , 2) , 'g*')
%%
% clustering
figure
hold on
plot(sort(spatial_corr_coeff(:,1)), '*')
plot(sort(ctrl_spatial_corr_coeff), '+')
title('sorted spatial corr')
legend('v1' , 'ctrl')

g1 = spatial_corr_coeff(:,1) < (0.3);