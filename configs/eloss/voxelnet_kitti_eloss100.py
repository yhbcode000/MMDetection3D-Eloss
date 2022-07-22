# model settings
_base_ = 'hv_second_secfpn_fp16_6x8_80e_kitti-3d-car-eloss.py'

# exp02-eloss50
# runner = dict(max_epochs = 45)

# exp02-eloss100
runner = dict(max_epochs = 50)