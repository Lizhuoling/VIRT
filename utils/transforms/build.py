from . import transforms as T

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(
        mean=cfg['DATA']['IMG_NORM_MEAN'], std=cfg['DATA']['IMG_NORM_STD'], to_bgr=False,
    )

    transform = T.Compose(
        [
            normalize_transform,
        ]
    )
    return transform
