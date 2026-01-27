from dataset_loader.loader_cora import loader_cora
from dataset_loader.loader_ogb_arxiv_year import load_ogb_arxiv_year
from dataset_loader.loader_squirrel import loader_squirrel
from dataset_loader.loader_chameleon import loader_chameleon
from dataset_loader.loader_snap_patents_year import loader_snap_patents_year
from dataset_loader.loader_reddit2 import load_reddit2
from dataset_loader.loader_coauthor import load_coauthor_cs
from dataset_loader.loader_amazon_ratings import load_amazon_ratings
from dataset_loader.loader_roman_empire import load_roman_empire


DATASET_STORAGE_PATH = "./dataset/"


def dataset_loader(dataset_name: str, config):
    if dataset_name == 'cora':
        return loader_cora(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'arxiv': # TODO fix the seed for the splits
        return load_ogb_arxiv_year(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'squirrel':
        return loader_squirrel(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'chameleon':
        return loader_chameleon(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'patents':
        return loader_snap_patents_year(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'reddit2':
        return load_reddit2(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'coauthor':
        return load_coauthor_cs(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'amazon_ratings':
        return load_amazon_ratings(DATASET_STORAGE_PATH, config)
    elif dataset_name == 'roman_empire':
        return load_roman_empire(DATASET_STORAGE_PATH, config)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
