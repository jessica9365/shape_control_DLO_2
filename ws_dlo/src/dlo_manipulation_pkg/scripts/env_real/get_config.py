import json
import argparse

def load_args_from_file(filepath):
    with open(filepath, 'r') as file:
        args_dict = json.load(file)
    # Convert dictionary back to Namespace
    args = argparse.Namespace(**args_dict)
    return args

args = load_args_from_file(filepath=r"C:/Users/91990/Documents/GitHub/FYP_Object_Detection_Model/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config.json")
args_ur = load_args_from_file(filepath=r"C:/Users/91990/Documents/GitHub/FYP_Object_Detection_Model/shape_control_DLO_2/ws_dlo/src/dlo_system_pkg/config/config_tf.json")


if __name__ == '__main__':
    print(args)
    print(args_ur)