from nflows import distributions, transforms, flows


def create_flow(flow_type):
    distribution = distributions.StandardNormal((3,))

    if flow_type == 'lu_flow':
        transform = transforms.CompositeTransform([
            transforms.RandomPermutation(3),
            transforms.LULinear(3, identity_init=False)
        ])
    elif flow_type == 'qr_flow':
        transform = transforms.QRLinear(3, num_householder=3)
    else:
        raise RuntimeError('Unknown type')

    return flows.Flow(transform, distribution)
