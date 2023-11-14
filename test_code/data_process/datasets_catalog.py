from data_process.UC import UC_names as UC_names_list

# from datasets.AID import class_names as AID_names_list


_PREFIX = 'image_directory'
_SOURCE_INDEX = 'image_and_label_list_file'
_MEAN = 'rgb_mean'
_STD = 'rgb_std'
_CLASSES_LIST = 'class_names_list'
_NUM_CLASSES = 'identities number'

# Available datasets
_DATASETS = {
    'UC': {
        _PREFIX: 'data_process/UC/',
        _SOURCE_INDEX: {
            'train': 'data_process/UC/UC_train_{}.txt',
            'test': 'data_process/UC/UC_test_{}.txt'
        },
        _MEAN:
            [0.4842271, 0.4900518, 0.45050353],
        _STD:
            [0.1734842, 0.1635247, 0.15547599],
        _CLASSES_LIST: UC_names_list,
        _NUM_CLASSES: 21
    },
    'none': {
        _PREFIX:
            [''],
        _SOURCE_INDEX:
            [''],
        _NUM_CLASSES: 1000
    },
}


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()

def contains(name):
    return name in _DATASETS.keys()

def get_prefix(name):
    return _DATASETS[name][_PREFIX]

def get_source_index(name):
    return _DATASETS[name][_SOURCE_INDEX]

def get_num_classes(name):
    return _DATASETS[name][_NUM_CLASSES]

def get_mean(name):
    return _DATASETS[name][_MEAN]

def get_std(name):
    return _DATASETS[name][_STD]

def get_names_list(name):
    return _DATASETS[name][_CLASSES_LIST]
