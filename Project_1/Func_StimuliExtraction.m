function spk_gen_stimuli = Func_StimuliExtraction(events,stimuli,neuron_num,frame_rate)

% spk_gen_stimuli = nan(16,16,length(events)); % frame*bar*spike_number

frm_rt = frame_rate(neuron_num);

count = 0;

for i = 1 : length(events)
    if (floor(events(i)*frm_rt/1e4)-16 >= 1 && floor(events(i)*frm_rt/1e4)-1<=length(stimuli)) % Avoiding non-positive index for matrices
        spk_gen_stimuli(:,:,i-count) = stimuli(floor(events(i)*frm_rt/1e4)-16:floor(events(i)*frm_rt/1e4)-1,:);
    else
        count = count + 1;
    end
    
end


end