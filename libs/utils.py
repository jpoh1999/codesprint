import yaml

def load_config_file(config_path : str) :
    """
        Helper function to load configuration yaml file 

        -----
        Args :
            config_path (str) : the path of our config file

        ------
        Returns :
            (dict) : the dictionary of our config key-value mappings
    """

    with open(config_path, "r") as file_obj :
        return yaml.load(file_obj, Loader=yaml.SafeLoader)
