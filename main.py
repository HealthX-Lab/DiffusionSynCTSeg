import gin
from src.image_model import Image
from src.dataset_model import Dataset

@gin.configurable("main")
def main(operation):
    if operation == 'EDA':
        dic_dir_to_images = Dataset()
        Image(dic_dir_to_images)


if __name__ == "__main__":
    gin.parse_config_file("./configs/1_config.gin")
    main()