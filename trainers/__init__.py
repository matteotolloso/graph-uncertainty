import os
import importlib

from .vanilla_trainer import vanilla_train
from .odin_tester import odin_test
from .credal_trainer import credal_train
from .ensamble_tester import ensemble_tester
from .credal_LJ_trainer import credal_LJ_train
from .mahalanobis_tester import mahalanobis_test
from .knn_tester import knn_test
from .energy_tester import energy_test
from .knn_LJ_tester import knn_LJ_test
from .gnn_safe_tester import gnnsafe_test
# from .gebm_tester import gebm_test
from .credal_frozen_trainer import credal_frozen_joint_train
from .cagcn_trainer import cagcn_train



# -- Do it automatically --

# # trainers/__init__.py
# import os
# import importlib

# # Automatically import all modules
# package_dir = os.path.dirname(__file__)
# for filename in os.listdir(package_dir):
#     if filename.endswith(".py") and filename != "__init__.py":
#         module_name = filename[:-3]
#         module = importlib.import_module(f"{__name__}.{module_name}")
#         # Pull out all functions that end in `_train` and expose them
#         for attr_name in dir(module):
#             if attr_name.endswith("_train"):
#                 globals()[attr_name] = getattr(module, attr_name)
