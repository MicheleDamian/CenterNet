cd src
# train
python main.py ctdet --exp_id coco_player_resdcn18 --dataset coco_player --arch resdcn_18 --batch_size 16 --lr 5e-4 --num_workers 2
cd ..
