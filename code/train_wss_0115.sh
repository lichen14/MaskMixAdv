python -u train_weakly_supervised_segmentation_pCE_mae_masked_post_process_aux_dropout.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_mae_post_process_adv_feature_mask-60000 --max_iterations 60000 --batch_size 12 --model unet_mae_cct_0110 &
python -u train_weakly_supervised_segmentation_pCE_mae_masked_post_process_aux_dropout.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_mae_post_process_adv_feature_mask-60000 --max_iterations 60000 --batch_size 12 --model unet_mae_cct_0110 
python -u train_weakly_supervised_segmentation_pCE_mae_masked_post_process_aux_dropout.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_mae_post_process_adv_feature_mask-60000 --max_iterations 60000 --batch_size 12 --model unet_mae_cct_0110 &
python -u train_weakly_supervised_segmentation_pCE_mae_masked_post_process_aux_dropout.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_mae_post_process_adv_feature_mask-60000 --max_iterations 60000 --batch_size 12  --model unet_mae_cct_0110 
python -u train_weakly_supervised_segmentation_pCE_mae_masked_post_process_aux_dropout.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_mae_post_process_adv_feature_mask-60000 --max_iterations 60000 --batch_size 12 --model unet_mae_cct_0110
python test_2D_fully_1201.py  --sup_type scribble  --exp ACDC/WeaklySeg_pCE_mae_post_process_adv_feature_mask-60000 --model unet_mae_cct_0110