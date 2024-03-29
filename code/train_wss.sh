python -u train_fully_supervised_2D.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/FullySupervisedSeg --max_iterations 60000 --batch_size 12 &
python -u train_fully_supervised_2D.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/FullySupervisedSeg --max_iterations 60000 --batch_size 12 &
python -u train_fully_supervised_2D.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/FullySupervisedSeg --max_iterations 60000 --batch_size 12 &
python -u train_fully_supervised_2D.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/FullySupervisedSeg --max_iterations 60000 --batch_size 12 &
python -u train_fully_supervised_2D.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/FullySupervisedSeg --max_iterations 60000 --batch_size 12

python -u train_weakly_supervised_pCE_2D.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_2D.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_2D.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_2D.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_2D.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE --max_iterations 60000 --batch_size 12

python -u train_weakly_supervised_pCE_GatedCRFLoss_2D.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_GatedCRFLoss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_GatedCRFLoss_2D.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_GatedCRFLoss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_GatedCRFLoss_2D.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_GatedCRFLoss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_GatedCRFLoss_2D.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_GatedCRFLoss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_GatedCRFLoss_2D.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_GatedCRFLoss --max_iterations 60000 --batch_size 12

python -u train_weakly_supervised_pCE_Entropy_Mini_2D.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_EntMini --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_Entropy_Mini_2D.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_EntMini --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_Entropy_Mini_2D.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_EntMini --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_Entropy_Mini_2D.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_EntMini --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_Entropy_Mini_2D.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_EntMini --max_iterations 60000 --batch_size 12

python -u train_weakly_supervised_pCE_MumfordShah_Loss_2D.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MumfordShah_Loss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_MumfordShah_Loss_2D.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MumfordShah_Loss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_MumfordShah_Loss_2D.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MumfordShah_Loss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_MumfordShah_Loss_2D.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MumfordShah_Loss --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_MumfordShah_Loss_2D.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MumfordShah_Loss --max_iterations 60000 --batch_size 12

python -u train_weakly_supervised_ustm_2D.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_USTM_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_ustm_2D.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_USTM_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_ustm_2D.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_USTM_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_ustm_2D.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_USTM_pCE --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_ustm_2D.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_USTM_pCE --max_iterations 60000 --batch_size 12

python -u train_weakly_supervised_pCE_random_walker_2D.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_Random_Walker --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_random_walker_2D.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_Random_Walker --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_random_walker_2D.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_Random_Walker --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_random_walker_2D.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_Random_Walker --max_iterations 60000 --batch_size 12 &
python -u train_weakly_supervised_pCE_random_walker_2D.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_Random_Walker --max_iterations 60000 --batch_size 12

python -u  train_weakly_supervised_pCE_WSL4MIS.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_WSL4MIS --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_WSL4MIS.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_WSL4MIS --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_WSL4MIS.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_WSL4MIS --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_WSL4MIS.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_WSL4MIS --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_WSL4MIS.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_WSL4MIS --max_iterations 60000 --batch_size 12 

python -u  train_weakly_supervised_pCE_MaskMixAdv.py --fold fold1 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MaskMixAdv --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_MaskMixAdv.py --fold fold2 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MaskMixAdv --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_MaskMixAdv.py --fold fold3 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MaskMixAdv --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_MaskMixAdv.py --fold fold4 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MaskMixAdv --max_iterations 60000 --batch_size 12 &
python -u  train_weakly_supervised_pCE_MaskMixAdv.py --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC/WeaklySeg_pCE_MaskMixAdv --max_iterations 60000 --batch_size 12 