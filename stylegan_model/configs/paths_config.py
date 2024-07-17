dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': '',
	'celeba_test': '',

	#  Cars Dataset (In the paper: Stanford cars)
	'cars_train': '',
	'cars_test': '',

	#  Horse Dataset (In the paper: LSUN Horse)
	'horse_train': '',
	'horse_test': '',

	#  Church Dataset (In the paper: LSUN Church)
	'church_train': '',
	'church_test': '',

	#  Cats Dataset (In the paper: LSUN Cat)
	'cats_train': '',
	'cats_test': '',

	'leaves_train':'/home/yotamnitzan/datasets/plant_village/flat_image_dir_splits/train',
	'leaves_test':'/home/yotamnitzan/datasets/plant_village/flat_image_dir_splits/val'
}

model_paths = {
	'stylegan_ffhq': '/disk2/amirh/attacking-stg-encoders/pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '/disk2/amirh/attacking-stg-encoders/pretrained_models/model_ir_se50.pth',
	'shape_predictor': '/disk2/amirh/attacking-stg-encoders/pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': '/disk2/amirh/attacking-stg-encoders/pretrained_models/moco_v2_800ep_pretrain.pth',
	'e4e': '../stylegan_model/weights/e4e_ffhq_encode.pt',
	'psp': '../stylegan_model/weights/psp_ffhq_encode.pt',
	'restyle': '../stylegan_model/weights/restyle_e4e_ffhq_encode.pt'
}
