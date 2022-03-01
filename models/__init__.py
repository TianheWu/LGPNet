def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'lgpnet':
        from .lgpnet import LGPNet
        return LGPNet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
