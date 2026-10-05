"""Importing this package registers every dataset.

To add a dataset: create ``<name>.py`` here with a raw loader and a
``register_dataset(DatasetSpec(...))`` call, then import it below
(see the ``add-dataset`` skill).
"""

from cgnn.data.sources import (  # noqa: F401
    coauthor,
    csbm,
    heterophilous,
    planetoid,
    reddit2,
    synthetic,
    wikipedia,
    year_graphs,
)
