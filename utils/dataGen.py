
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.io import imread, _check_channel_order
from utils.core import _check_df_load
from utils.transform import _check_augs

def make_data_generator(config, df, stage='train'):
    """Create an appropriate data generator based on the framework used.

    A wrapper for the high-end ``solaris`` API to create data generators.
    Using the ``config`` dictionary, this function creates an instance of
    :class:`TorchDataset`. If using Torch, this
    instance is then wrapped in a :class:`torch.utils.data.DataLoader` and
    returned; 

    Arguments
    ---------
    config : dict
        The config dictionary for the entire pipeline.
    df : :class:`pandas.DataFrame` or :class:`str`
        A :class:`pandas.DataFrame` containing two columns: ``'image'``, with
        the path to images for training, and ``'label'``, with the path to the
        label file corresponding to each image.
    stage : str, optional
        Either ``'train'`` or ``'validate'``, indicates whether the object
        created is being used for training or validation. This determines which
        augmentations from the config file are applied within the returned
        object.

    Returns
    -------
    data_gen : :class:`torch.utils.data.DataLoader`
        An object to pass data into the :class:`solaris.nets.train.Trainer`
        instance during model training.
    """
    # make sure the df is loaded
    df = _check_df_load(df)

    if stage == 'train':
        augs = config['training_augmentation']
        shuffle = config['training_augmentation']['shuffle']
    elif stage == 'validate':
        augs = config['validation_augmentation']
        shuffle = False

    try:
        num_classes = config['model_specs']['classes']
    except KeyError:
        num_classes = 1
    
    dataset = TorchDataset(
            df,
            augs=augs,
            batch_size=config['batch_size'],
            label_type=config['data_specs']['label_type'],
            is_categorical=config['data_specs']['is_categorical'],
            num_classes=num_classes,
            dtype=config['data_specs']['dtype'])

    # set up workers for DataLoader for pytorch
    data_workers = config['data_specs'].get('data_workers')
    if data_workers == 1 or data_workers is None:
        data_workers = 0  # for DataLoader to run in main process
    
    data_gen = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config['training_augmentation']['shuffle'],
            num_workers=data_workers)

    return data_gen

def make_SAR_RGB_data_generator(config, sar_df, rgb_df, stage='train'):

    # make sure the df is loaded
    sar_df = _check_df_load(sar_df)
    rgb_df = _check_df_load(rgb_df)

    if stage == 'train':
        augs = config['training_augmentation']
        shuffle = config['training_augmentation']['shuffle']
    elif stage == 'validate':
        augs = config['validation_augmentation']
        shuffle = False

    try:
        num_classes = config['model_specs']['classes']
    except KeyError:
        num_classes = 1
    
    dataset = SAR_RGB_TorchDataset(sar_df,
            rgb_df,
            augs=augs,
            batch_size=config['batch_size'],
            label_type=config['data_specs']['label_type'],
            is_categorical=config['data_specs']['is_categorical'],
            num_classes=num_classes,
            dtype=config['data_specs']['dtype'])

    # set up workers for DataLoader for pytorch
    data_workers = config['data_specs'].get('data_workers')
    if data_workers == 1 or data_workers is None:
        data_workers = 0  # for DataLoader to run in main process

    data_gen = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config['training_augmentation']['shuffle'],
            num_workers=data_workers,
            drop_last=True)

    return data_gen

class TorchDataset(Dataset):
    """A PyTorch dataset object for solaris.

    Note that this object is wrapped in a :class:`torch.utils.data.DataLoader`
    before being passed to the :class:solaris.nets.train.Trainer` instance.

    Attributes
    ----------
    df : :class:`pandas.DataFrame`
        The :class:`pandas.DataFrame` specifying where inputs are stored.
    aug : :class:`albumentations.core.composition.Compose`
        An albumentations Compose object to pass imagery through before
        passing it into the neural net. If an augmentation config subdict
        was provided during initialization, this is created by parsing the
        dict with :func:`solaris.nets.transform.process_aug_dict`.
    batch_size : int
        The batch size generated.
    n_batches : int
        The number of batches per epoch. Inferred based on the number of
        input files in `df` and `batch_size`.
    dtype : :class:`numpy.dtype`
        The numpy dtype that image inputs should be when passed to the model.
    is_categorical : bool
        Indicates whether masks output are boolean or categorical labels.
    num_classes: int
        Indicates the number of classes in the dataset
    dtype : class:`numpy.dtype`
        The data type images should be converted to before being passed to
        neural nets.
    """

    def __init__(self, df, augs, batch_size, label_type='mask',
                 is_categorical=False, num_classes=1, dtype=None):
        """
        Create an instance of TorchDataset for use in model training.

        Arguments
        ---------
        df : :class:`pandas.DataFrame`
            A pandas DataFrame specifying images and label files to read into
            the model. See `the reference file creation tutorial`_ for more.
        augs : :class:`dict` or :class:`albumentations.core.composition.Compose`
            Either the config subdict specifying augmentations to apply, or
            a pre-created :class:`albumentations.core.composition.Compose`
            object containing all of the augmentations to apply.
        batch_size : int
            The number of samples in a training batch.
        label_type : str, optional
            The type of labels to be used. At present, only ``"mask"`` is
            supported.
        is_categorical : bool, optional
            Is the data categorical or boolean (default)?
        num_classes: int
            Indicates the number of classes in the dataset
        dtype : str, optional
            The dtype that image arrays should be converted to before being
            passed to the neural net. If not provided, defaults to
            ``"float32"``. Must be one of the `numpy dtype options`_.

        .. _numpy dtype options: https://docs.scipy.org/doc/numpy/user/basics.types.html
        """
        super().__init__()

        self.df = df
        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        self.aug = _check_augs(augs)
        self.is_categorical = is_categorical
        self.num_classes = num_classes

        if dtype is None:
            self.dtype = np.float32  # default
        # if it's a string, get the appropriate object
        elif isinstance(dtype, str):
            try:
                self.dtype = getattr(np, dtype)
            except AttributeError:
                raise ValueError(
                    'The data type {} is not supported'.format(dtype))
        # lastly, check if it's already defined in the right format for use
        elif issubclass(dtype, np.number) or isinstance(dtype, np.dtype):
            self.dtype = dtype
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get one image, mask pair"""
        # Generate indexes of the batch
        image = imread(self.df['image'].iloc[idx])
        mask = imread(self.df['label'].iloc[idx])
        if not self.is_categorical:
            mask[mask != 0] = 1
        # if len(mask.shape) == 2:
        #     mask = mask[:, :, np.newaxis]
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        sample = {'image': image, 'mask': mask}

        if self.aug:
            sample = self.aug(**sample)

        sample['image'] = _check_channel_order(sample['image'],
                                               'torch').astype(self.dtype)
        
        return sample


class SAR_RGB_TorchDataset(Dataset):

    def __init__(self, df_sar, df_rgb, augs, batch_size, label_type='mask',
                 is_categorical=False, num_classes=1, dtype=None):

        super().__init__()

        self.df_sar = df_sar
        self.df_rgb = df_rgb
        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.df_sar)/self.batch_size))
        self.aug = _check_augs(augs, additional_targets={'imageRGB':'image'})
        self.is_categorical = is_categorical
        self.num_classes = num_classes

        if dtype is None:
            self.dtype = np.float32  # default
        # if it's a string, get the appropriate object
        elif isinstance(dtype, str):
            try:
                self.dtype = getattr(np, dtype)
            except AttributeError:
                raise ValueError(
                    'The data type {} is not supported'.format(dtype))
        # lastly, check if it's already defined in the right format for use
        elif issubclass(dtype, np.number) or isinstance(dtype, np.dtype):
            self.dtype = dtype

    def __len__(self):
        return len(self.df_sar)    

    def __getitem__(self, idx):
        """Get one SAR image, RGB image, mask triplet"""
        # Generate indexes of the batch
        SAR_image = imread(self.df_sar['image'].iloc[idx])
        RGB_image = imread(self.df_rgb['image'].iloc[idx])
        mask = imread(self.df_sar['label'].iloc[idx])    

        if not self.is_categorical:
            mask[mask != 0] = 1
        # if len(mask.shape) == 2:
        #     mask = mask[:, :, np.newaxis]
        if len(SAR_image.shape) == 2:
            SAR_image = SAR_image[:, :, np.newaxis]
            RGB_image = RGB_image[:, :, np.newaxis]

        sample = {'image': SAR_image, 'imageRGB': RGB_image, 'mask': mask}

        if self.aug:
            sample = self.aug(**sample)
        
        sample['image'] = _check_channel_order(sample['image'],
                                               'torch').astype(self.dtype)
        sample['imageRGB'] = _check_channel_order(sample['imageRGB'],
                                               'torch').astype(self.dtype)
        
        return sample