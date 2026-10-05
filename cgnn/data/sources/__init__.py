"""Importing this package registers every dataset.

To add a dataset: create ``<name>.py`` here with a raw loader and a
``register_dataset(DatasetSpec(...))`` call, then import it below.
"""

from cgnn.data.sources import (  # noqa: F401
    coauthor,
    heterophilous,
    reddit2,
    synthetic,
    wikipedia,
    year_graphs,
)
