import gin
from src.image_model import Image
from src.datasetscollection_model import DatasetsCollection


@gin.configurable("main")
def main(operation):
    if operation == 'EDA':
        datasets = DatasetsCollection()
        image_obj = Image(datasets.get_paths())
        images = image_obj.get_images()
        image_obj.show_images()
        a = 2


if __name__ == "__main__":
    gin.parse_config_file("./configs/1_config.gin")
    main()