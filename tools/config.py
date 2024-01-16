import yaml


def load_yml(yml_file):
    with open(yml_file, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


if __name__ == '__main__':
    yml_file = '../config.yml'
    args = load_yml(yml_file)
    print(args)
