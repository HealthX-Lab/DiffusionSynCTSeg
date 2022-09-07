import gin
from src.image_model import Image
from src.datasetscollection_model import DatasetsCollection


@gin.configurable("main")
def main(operation):
    if operation == 'EDA':
        datasets = DatasetsCollection()
        # image = Image(datasets.get_path_to_iamges())


if __name__ == "__main__":
    gin.parse_config_file("./configs/1_config.gin")
    main()