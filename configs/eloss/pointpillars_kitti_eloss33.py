# model settings
_base_ = 'hv_pointpillars_secfpn_fp16_6x8_160e_kitti-3d-car-eloss.py'

# exp02-eloss33
runner = dict(max_epochs = 85)

# exp02-eloss66
# runner = dict(max_epochs = 90)

# exp02-eloss99
# runner = dict(max_epochs = 95)