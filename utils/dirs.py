import os


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def get_files(path, suffix=".json", is_debug = False):
    '''
    obtain the json file list in the path recursively
    :param path:
    :param suffix:
    :return: json file list
    '''
    file_list = []
    try:
        if os.path.exists(path):
            # browse each folder
            for home, dirs, files in os.walk(path):
                for file in files:
                    if suffix in file:
                        file_path = os.path.join(home, file)
                        if is_debug:
                            print("{}".format(file_path))
                        file_list.append(file_path)
    except Exception as e:
        print(e)
        exit(-1)
    return file_list