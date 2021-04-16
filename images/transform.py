from nflows import utils, transforms
from nflows.nn.nets import ConvResidualNet


def create_transform_step(num_channels,
                          hidden_channels,
                          num_res_blocks,
                          resnet_batchnorm,
                          dropout_prob,
                          actnorm,
                          spline_type,
                          num_bins):
    def create_convnet(in_channels, out_channels):
        net = ConvResidualNet(in_channels=in_channels,
                              out_channels=out_channels,
                              hidden_channels=hidden_channels,
                              num_blocks=num_res_blocks,
                              use_batch_norm=resnet_batchnorm,
                              dropout_probability=dropout_prob)
        return net

    step_transforms = []

    mask = utils.create_mid_split_binary_mask(num_channels)

    if spline_type == 'rational_quadratic':
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            num_bins=num_bins,
            tails='linear'
        )
    elif spline_type == 'quadratic':
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            num_bins=num_bins,
            tails='linear'
        )
    else:
        raise RuntimeError('Unkown spline_type')

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return transforms.CompositeTransform(step_transforms)


def create_transform(c, h, w, num_bits,
                     levels,
                     steps_per_level,
                     step_config):
    mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
    for level in range(levels):
        squeeze_transform = transforms.SqueezeTransform()
        c, h, w = squeeze_transform.get_output_shape(c, h, w)

        transform_level = transforms.CompositeTransform(
            [squeeze_transform]
            + [create_transform_step(c, **step_config) for _ in range(steps_per_level)]
            + [transforms.OneByOneConvolution(c)]  # End each level with a linear transformation.
        )

        new_shape = mct.add_transform(transform_level, (c, h, w))
        if new_shape:  # If not last layer
            c, h, w = new_shape

    # Map to [-0.5,0.5]
    preprocess_transform = transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits),
                                                            shift=-0.5)

    transform = transforms.CompositeTransform([
        preprocess_transform,
        mct
    ])

    return transform