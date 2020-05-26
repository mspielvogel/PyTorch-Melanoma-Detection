import os
import random
import shutil
import concurrent.futures


class DatasetCreator():
    def __init__(self, config):
        """
        Args:
            config (dict): dictionary of parameters
        """
        self._input_folder = config["input_folder"]
        self._output_folder = config["output_folder"]
        self._dataset_name = config.get("dataset_name", "HAM10000")
        self._datasplit = config.get("datasplit", [0.7, 0.2, 0.1])  # train, validation, test

    def start(self, balance=False):
        """
        Args:
            config (bool): indicates if balancing is applied on train data
        """
        self.__read_dataset()
        self.__split_dataset()
        if balance:
            self.balance_dataset()
        self.__create_dataset()

    def __read_dataset(self):
        if self._dataset_name == "HAM10000":
            self._image_paths = []
            paths = [os.path.join(self._input_folder, x) for x in os.listdir(self._input_folder)]
            for path in paths:
                if os.path.isdir(path):
                    self._image_paths.extend([os.path.join(path, x) for x in os.listdir(path)])
                elif os.path.isfile(path) and path.endswith("metadata.csv"):
                    self._metadata_path = path
        else:
            raise (NotImplementedError, "Currently, there is only the dataset HAM10000 available.")

    def __balance_dataset(self):
        # Only balancing for train set
        pass

    def __split_dataset(self):
        # Set seed to get always the same split for the same dataset
        random.seed(42)
        random.shuffle(self._image_paths)

        n = len(self._image_paths)
        train, val, _ = self._datasplit
        train_split = int(n * train) + 1
        val_split = int(n * (train + val)) + 1
        self._split_paths = {}
        self._split_paths["train"] = self._image_paths[:train_split]
        self._split_paths["validation"] = self._image_paths[train_split:val_split]
        self._split_paths["test"] = self._image_paths[val_split:]

    def __create_dataset(self):
        if os.path.exists(self._output_folder):
            shutil.rmtree(self._output_folder)

        with concurrent.futures.ThreadPoolExecutor(max_workers=512) as executor:
            # Copy image paths to corresponding split folder in output folder
            for split, image_paths in self._split_paths.items():
                split_folder = os.path.join(self._output_folder, split)
                if not os.path.exists(split_folder):
                    os.makedirs(split_folder)

                for image_path in image_paths:
                    image_name = os.path.basename(image_path)
                    output_path = os.path.join(split_folder, image_name)
                    executor.submit(shutil.copy(image_path, output_path))
            # Copy metadata to output folder
            executor.submit(shutil.copy(self._metadata_path, self._output_folder))


if __name__ == "__main__":
    config = {"input_folder": "data/data_sets/HAM10000",
              "output_folder": "data/training_sets/HAM10000"}

    dataset_creator = DatasetCreator(config)
    dataset_creator.start()
