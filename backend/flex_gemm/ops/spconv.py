def set_algorithm(_algorithm):
    return None


def set_hashmap_ratio(_ratio):
    return None


def sparse_submanifold_conv3d(*_args, **_kwargs):
    raise RuntimeError(
        "flex_gemm native sparse convolution is not available on this Windows install. "
        "Use the spconv backend instead."
    )

