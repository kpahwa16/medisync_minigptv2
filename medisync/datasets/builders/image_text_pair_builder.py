import os
import logging
import warnings

from medisync.common.registry import registry
from medisync.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from medisync.datasets.datasets.openi_caption import OPENICapDataset
from medisync.datasets.datasets.mimic_caption import MIMICCapDataset
from medisync.datasets.datasets.roco_rad_caption import ROCORADCapDataset
from medisync.datasets.datasets.roco_nonrad_caption import ROCONONRADCapDataset
from medisync.datasets.datasets.vqarad_dataset import VQARADDataset
from medisync.datasets.datasets.pmc_vqa import PMCVQADataset
from medisync.datasets.datasets.pmc_caption import PMCCapDataset
from medisync.datasets.datasets.slake import SLAKEGroundedDetailDataset
from medisync.datasets.datasets.slake_vqa import SLAKEVQADataset

@registry.register_builder("pmc_caption")
class PMCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = PMCCapDataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/khushbu/medisyncMed/medisync/configs/datasets/pmc/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
        vis_processor=self.vis_processors["train"],
        text_processor=self.text_processors["train"],
        ann_path=os.path.join(storage_path, 'pmcvqa_captions.json'),  # Correct keyword argument
        vis_root=os.path.join(storage_path, 'figures'),
        )

        return datasets
    
@registry.register_builder("mimic_caption")
class MIMICBuilder(BaseDatasetBuilder):
    train_dataset_cls = MIMICCapDataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/mimic/defaults.yaml"}

    def _download_ann(self):
        # Implement downloading logic if necessary
        pass

    def _download_vis(self):
        # Implement downloading logic if necessary
        pass

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn(f"Storage path {storage_path} does not exist.")
        
        # Here we pass the path as a string directly, not as a list
        ann_path = os.path.join(storage_path, 'filter_cap.json')
        vis_root = os.path.join(storage_path, 'train')

        # Create dataset instance
        datasets['train'] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=vis_root
        )

        return datasets

@registry.register_builder("openi_cap")
class OpenIBuilder(BaseDatasetBuilder):
    train_dataset_cls = OPENICapDataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/openi/defaults.yaml"}

    def _download_ann(self):
        # Implement downloading logic if necessary
        pass

    def _download_vis(self):
        # Implement downloading logic if necessary
        pass

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn(f"Storage path {storage_path} does not exist.")
        
        # Here we pass the path as a string directly, not as a list
        ann_path = os.path.join(storage_path, 'filter_cap.json')
        vis_root = os.path.join(storage_path, 'train')

        # Create dataset instance
        datasets['train'] = self.train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=vis_root
        )

        return datasets

@registry.register_builder("rocorad_caption")
class ROCORADCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ROCORADCapDataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/rocorad_caption/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        
        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
        vis_processor=self.vis_processors["train"],
        text_processor=self.text_processors["train"],
        ann_path=os.path.join(storage_path, "roco_rad_captions.json"),  # Correct keyword argument
        vis_root=os.path.join(storage_path, 'images'),
        )

        return datasets





@registry.register_builder("rocononrad_caption")
class ROCONONRADCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ROCONONRADCapDataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/rocononrad_caption/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
        vis_processor=self.vis_processors["train"],
        text_processor=self.text_processors["train"],
        ann_path=os.path.join(storage_path, 'roco_nonrad_captions.json'),  # Correct keyword argument
        vis_root=os.path.join(storage_path, 'images'),
        )

        return datasets
    


@registry.register_builder("pmc_vqa")
class PMCVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PMCVQADataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/pmc/defaults_vqa.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'pmc_vqa_dataset.json')],
            vis_root=os.path.join(storage_path, 'figures'),
        )

        return datasets


@registry.register_builder("slake_vqa")
class SLAKEVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = SLAKEVQADataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/slake/defaults_vqa.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        print("inside slake vqa: ", storage_path)
        ann_paths=[os.path.join(storage_path, 'full_SLAKE_VQA_train.json')]
        print("ann_pahts", ann_paths)

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
       
            ann_paths=[os.path.join(storage_path, 'full_SLAKE_VQA_train.json')],
            vis_root=os.path.join(storage_path, 'imgs'),
        )

        return datasets


@registry.register_builder("vqarad")
class VQARADBuilder(BaseDatasetBuilder):
    train_dataset_cls = VQARADDataset

    DATASET_CONFIG_DICT = {"default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/vqa_rad/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=os.path.join(storage_path, 'trainset.json'),
            vis_root=os.path.join(storage_path, 'images'),
        )

        return datasets


@registry.register_builder("slake_grounded_caption")
class SLAKEGroundedCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = SLAKEGroundedDetailDataset
    DATASET_CONFIG_DICT = {
        "default": "/home/kp66/medisync_minigptv2/medisync/configs/datasets/slake/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets
