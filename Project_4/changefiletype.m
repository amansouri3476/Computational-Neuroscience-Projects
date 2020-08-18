function changefiletype(filename)

    V=spm_vol(sprintf('%s.img',filename));
    ima=spm_read_vols(V);
    V.fname=sprintf('%s.nii',filename);
    spm_write_vol(V,ima);
    
    
end