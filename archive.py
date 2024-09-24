


def print_dict_structure(d, indent=0):
    """
    Recursively prints the structure of a nested dictionary without printing the values.

    :param d: The dictionary to explore.
    :param indent: The current level of indentation (used for nested dictionaries).
    """
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + str(key) + ':')
            print_dict_structure(value, indent + 1)
    elif isinstance(d, list):
        print('  ' * indent + '[List]')
    else:
        print('  ' * indent + '[Value]')