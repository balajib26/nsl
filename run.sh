conda activate clip_prefix_caption
python parse_coco.py --clip_model_type ViT-B/32 --num_examples 10
python train.py --only_prefix --data ./data/coco/oscar_split_ViT-B_32_train.pkl --out_dir ./coco_train/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40
python predict_output.py


