function spk_hdrs = Func_ReadData(address)
    cd(sprintf('Data/Spike_and_Log_Files/%s',address));
    
    addpath '..//..//..//MatlabFunctions/fileload';
    file_dirs = dir;
    
    spk_hdrs = [];
    
    for i = 1:length(file_dirs)
        file_name = file_dirs(i).name; 
        if (~isempty(strfind(lower(file_name) , 'msq1d.sa0')) && isempty(strfind(lower(file_name) , '.sub')) && isempty(strfind(lower(file_name) , '.vecs')))
            events = fget_spk(file_name);
            hdr = fget_hdr(file_name);
            spk_hdrs = [spk_hdrs struct('events',events,'hdr',hdr)];
        end
        
    end
    
    
    cd '..//..//..';
    
end