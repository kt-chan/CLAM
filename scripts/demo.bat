 ## MHIST
 python create_patches_fp.py --source .\data\inputs\MHIST\images --save_dir .\data\outputs\CLAM_RESULTS --patch_size 256 --seg --patch
 
 python extract_features_fp.py --data_h5_dir  .\data\outputs\CLAM_RESULTS\patches --data_slide_dir .\data\inputs\MHIST\images --csv_path .
 \data\outputs\CLAM_RESULTS\process_list_autogen.csv --feat_dir .\data\outputs\CLAM_RESULTS\MHIST_FEATURES --batch_size 512 


 ## TCGA

python create_patches_fp.py --source .\data\inputs\TCGA --save_dir .\data\outputs\TCGA --patch_size 256 --seg --patch 

python extract_features_fp.py --data_h5_dir  .\data\outputs\CLAM_RESULTS\patches --data_slide_dir .\data\inputs\MHIST\images --csv_path .
 \data\outputs\CLAM_RESULTS\process_list_autogen.csv --feat_dir .\data\outputs\CLAM_RESULTS\MHIST_FEATURES --batch_size 512 
