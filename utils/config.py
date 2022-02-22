import yaml

def parse(path):
    """Parse a config file for running a model.

    Arguments
    ---------
    path : str
        Path to the YAML-formatted config file to parse.

    Returns
    -------
    config : dict
        A `dict` containing the information from the config file at `path`.

    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    if not config['train']:
        raise ValueError('"train must be true.')
    # if config['train'] and config['training_data_csv'] is None:
    #     raise ValueError('"training_data_csv" must be provided if training.')
    # if config['infer'] and config['inference_data_csv'] is None:
    #     raise ValueError('"inference_data_csv" must be provided if "infer".')
    if config['training'].get('lr', None) is not None:
        config['training']['lr'] = float(config['training']['lr'])

    if config['training'].get('lr_sar', None) is not None:
        config['training']['lr_sar'] = float(config['training']['lr_sar'])
    if config['training'].get('lr_rgb', None) is not None:
        config['training']['lr_rgb'] = float(config['training']['lr_rgb'])
    if config['training'].get('lr_D', None) is not None:
        config['training']['lr_D'] = float(config['training']['lr_D'])

    # if config['validation_augmentation'] is not None \
    #         and config['inference_augmentation'] is None:
    #     config['inference_augmentation'] = config['validation_augmentation']
    
    return config

