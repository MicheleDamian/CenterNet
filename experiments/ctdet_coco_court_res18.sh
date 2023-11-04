cd src
# test
python test.py ctdet --exp_id coco_player_res18 --dataset coco_court --data_dir "{Config.root_path}" --arch res_18 --keep_res --resume
cd ..