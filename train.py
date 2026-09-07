import argparse
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch

from utils import *
from train_test import SGADetector


# =========================================================
# Set random seed
# =========================================================
'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# =========================================================
# Override model_config using command-line arguments
# =========================================================
def override_model_config(model_config, args):

    override_keys = [
        "lr",
        "weight_decay",
        "h_feats",
        "n_layers",
        "alpha",
        "dropout",
        "hidden_dim",
        "num_queries",
        "high_homophily_threshold",
        "low_homophily_threshold",
    ]

    for key in override_keys:

        value = getattr(
            args,
            key,
            None
        )

        # Only override when explicitly specified
        # from the command line
        if value is not None:
            model_config[key] = value

    return model_config


# =========================================================
# Save experiment results
# =========================================================
def save_results(
    result_file,
    model,
    model_config,
    train_config,
    datasets_train,
    datasets_test,
    auc_dict,
    pre_dict,
    auc_mean_dict,
    auc_std_dict,
    pre_mean_dict,
    pre_std_dict,
    args,
):

    result_dir = os.path.dirname(
        result_file
    )

    if result_dir:
        os.makedirs(
            result_dir,
            exist_ok=True
        )

    with open(
        result_file,
        "a",
        encoding="utf-8",
    ) as f:

        # =================================================
        # Experiment header
        # =================================================
        f.write("\n")
        f.write("=" * 120 + "\n")

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        f.write(
            "Time: {}\n".format(
                current_time
            )
        )

        f.write(
            "Model: {}\n".format(
                model
            )
        )

        f.write(
            "Train datasets: {}\n".format(
                datasets_train
            )
        )

        f.write(
            "Test datasets: {}\n".format(
                datasets_test
            )
        )

        f.write(
            "Trials: {}\n".format(
                args.trials
            )
        )

        f.write(
            "K: {}\n".format(
                args.K
            )
        )

        # =================================================
        # Training configuration
        # =================================================
        f.write("\n")
        f.write(
            "[Train Config]\n"
        )

        for key, value in train_config.items():

            f.write(
                "{}: {}\n".format(
                    key,
                    value
                )
            )

        # =================================================
        # Model configuration
        # =================================================
        f.write("\n")
        f.write(
            "[Model Config]\n"
        )

        for key, value in model_config.items():

            f.write(
                "{}: {}\n".format(
                    key,
                    value
                )
            )

        f.write("\n")
        f.write(
            "-" * 120 + "\n"
        )

        f.write(
            "[Test Results]\n"
        )

        # =================================================
        # Results for each test dataset
        # =================================================
        for test_data_name in auc_mean_dict:

            str_result = (
                "AUROC:{:.4f}+-{:.4f}, "
                "AUPRC:{:.4f}+-{:.4f}"
            ).format(
                auc_mean_dict[
                    test_data_name
                ],
                auc_std_dict[
                    test_data_name
                ],
                pre_mean_dict[
                    test_data_name
                ],
                pre_std_dict[
                    test_data_name
                ],
            )

            f.write(
                "{}: {}\n".format(
                    test_data_name,
                    str_result
                )
            )

            # Save every trial
            f.write(
                "  AUROC trials: {}\n".format(
                    auc_dict[
                        test_data_name
                    ]
                )
            )

            f.write(
                "  AUPRC trials: {}\n".format(
                    pre_dict[
                        test_data_name
                    ]
                )
            )

        f.write(
            "=" * 120 + "\n"
        )


# =========================================================
# Main
# =========================================================
warnings.filterwarnings(
    "ignore"
)


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser()


# ---------------------------------------------------------
# Basic experiment settings
# ---------------------------------------------------------
parser.add_argument(
    "--trials",
    type=int,
    default=4
)

parser.add_argument(
    "--model",
    type=str,
    default="SGAD"
)

parser.add_argument(
    "--shot",
    type=int,
    default=0
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=2000
)

parser.add_argument(
    "--K",
    type=int,
    default=2
)

parser.add_argument(
    "--json_dir",
    type=str,
    default="./params"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=60
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:1"
)


# =========================================================
# Model hyperparameters
#
# IMPORTANT:
# default=None means:
#
# if shell does not specify the parameter,
# keep the value from default/saved model_config
# =========================================================
parser.add_argument(
    "--lr",
    type=float,
    default=None
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=None
)

parser.add_argument(
    "--h_feats",
    type=int,
    default=None
)

parser.add_argument(
    "--n_layers",
    type=int,
    default=None
)

parser.add_argument(
    "--alpha",
    type=float,
    default=None
)
parser.add_argument(
    "--beta",
    type=float,
    default=None
)

parser.add_argument(
    "--dropout",
    type=float,
    default=None
)

parser.add_argument(
    "--hidden_dim",
    type=int,
    default=None
)

parser.add_argument(
    "--num_queries",
    type=int,
    default=None
)

parser.add_argument(
    "--high_homophily_threshold",
    type=float,
    default=None
)
parser.add_argument(
    "--low_homophily_threshold",
    type=float,
    default=None
)


# =========================================================
# Result file
# =========================================================
parser.add_argument(
    "--result_file",
    type=str,
    default="./results/SGAD_abla_all(opt+jac)_inject-pub+citeseer-seed4-tiao.txt"
)


args = parser.parse_known_args()[0]


# =========================================================
# Dataset configuration
# =========================================================
datasets_test = [
   #"Dgraphfin",
   "Facebook",  
    "Weibo",
     "Reddit",
     'Questions', 
     "Elliptic",
     "Tfinance",
     'Amazon',
     "Yelpchi",   
    'Amazon-all',
    "Yelpchi-all",
      "Disney",
      'Book',
    "Tolokers",  
    #'photo'
   # "Enron",    
]

datasets_train = [  
  #'pubmed_0p6', 'pubmed_0p8',
    'ASN_pubmed_0p2_feature', 'ASN_citeseer_0p2_feature',
     'Equ_pubmed_0p2_feature','Equ_citeseer_0p2_feature', 
     'ALN_pubmed_0p2_feature', 'ALN_citeseer_0p2_feature',

    # 'citeseer_0p2',  'citeseer_0p4', 
    
    
     
     
     #'cora_0p3',  'cora_0p7', 





#'pubmed_0p4', 'pubmed_0p6','Equ_pubmed_0p4_feature','Equ_pubmed_0p6_feature',
#'ASN_pubmed_0p4_feature','ASN_pubmed_0p6_feature',
#'ALN_pubmed_0p4_feature', 'ALN_pubmed_0p6_feature',
    #'physics_0p4', 'physics_0p2'
    #'citeseer_0p1',
    #'citeseer_0p2','citeseer_0p3','citeseer_0p4','citeseer_0p5',
    #'citeseer_0p6','citeseer_0p7','citeseer_0p8','citeseer_0p9',
]


model = args.model

model_result = {
    "name": model
}


print(
    "Training on {} datasets:".format(
        len(datasets_train)
    ),
    datasets_train
)

print(
    "Test on {} datasets:".format(
        len(datasets_test)
    ),
    datasets_test
)


# =========================================================
# Train configuration
# =========================================================
train_config = {

    "device": args.device,

    "epochs": args.epochs,

    "K": args.K,

    "batch_size": args.batch_size,

    "testdsets": datasets_test,
}


# =========================================================
# Load datasets
# =========================================================
data_train = [
    Dataset(name)
    for name in datasets_train
]

data_test = [
    Dataset(name)
    for name in datasets_test
]


# =========================================================
# Read saved model configuration
# =========================================================
model_config = read_json(
    model,
    args.K,
    args.shot,
    args.json_dir,
)


# =========================================================
# Default model configuration
# =========================================================
if model_config is None:

    model_config = {

        "model": "SGAD",

        "lr": 1e-4,

        "weight_decay": 5e-5,

        "h_feats": 2048,

        "n_layers": 4,

        "alpha": 0.5,

        'beta': 0.7,

        "dropout": 0.0,

        "hidden_dim":64,

        "num_queries": 32,

        "high_homophily_threshold": 0.7,

        "low_homophily_threshold": 0.3,
    }

    print(
        "use default model config"
    )

else:

    print(
        "use saved best model config"
    )

    print(
        model_config
    )


# =========================================================
# Command-line arguments override model_config
# =========================================================
model_config = override_model_config(
    model_config,
    args,
)


# =========================================================
# Other model parameters
# =========================================================
model_config[
    "model"
] = model

model_config[
    "input_dim"
] = 2*(args.K + 1)


# =========================================================
# Print final model configuration
# =========================================================
print("\n")
print(
    "=" * 100
)

print(
    "Final Model Configuration"
)

print(
    "=" * 100
)

for key, value in model_config.items():

    print(
        "{}: {}".format(
            key,
            value
        )
    )

print(
    "=" * 100
)


# =========================================================
# Preprocess graph bases
# =========================================================
for tr_data in data_train:

    tr_data.load_base(
        train_config["K"]
    )
    #tr_data.split_vail()
    model_config[
    "num_node"] = tr_data.graph.x.shape[0]


for te_data in data_test:

    te_data.load_base(
        train_config["K"]
    )
    #tr_data.split_vail()
    model_config[
    "num_node"] = te_data.graph.x.shape[0]


# =========================================================
# Store scores across trials
# =========================================================
auc_dict = {}

pre_dict = {}


# =========================================================
# Multiple trials
# =========================================================
for t in range(
    args.trials
):

    seed = t+args.trials

    set_seed(
        seed
    )

    print("\n")
    print(
        "=" * 100
    )

    print(
        "Model {}, Trial {}".format(
            model,
            t
        )
    )

    print(
        "=" * 100
    )

    train_config[
        "seed"
    ] = seed


    data = {

        "train": data_train,

        "test": data_test,
    }


    # =====================================================
    # Initialize detector
    # =====================================================
    detector = SGADetector(
        train_config,
        model_config,
        data
    )


    # =====================================================
    # Train + zero-shot test
    # =====================================================
    test_score_list = (
        detector.train()
    )


    # =====================================================
    # Aggregate scores
    # =====================================================
    for (
        test_data_name,
        test_score
    ) in test_score_list.items():

        if (
            test_data_name
            not in auc_dict
        ):

            auc_dict[
                test_data_name
            ] = []

            pre_dict[
                test_data_name
            ] = []


        auc_dict[
            test_data_name
        ].append(
            test_score[
                "AUROC"
            ]
        )


        pre_dict[
            test_data_name
        ].append(
            test_score[
                "AUPRC"
            ]
        )


        print(
            "Test on {}, "
            "AUC is {}".format(
                test_data_name,
                auc_dict[
                    test_data_name
                ]
            )
        )


# =========================================================
# Mean and standard deviation
# =========================================================
auc_mean_dict = {}

auc_std_dict = {}

pre_mean_dict = {}

pre_std_dict = {}


for test_data_name in auc_dict:

    auc_mean_dict[
        test_data_name
    ] = np.mean(
        auc_dict[
            test_data_name
        ]
    )


    auc_std_dict[
        test_data_name
    ] = np.std(
        auc_dict[
            test_data_name
        ]
    )


    pre_mean_dict[
        test_data_name
    ] = np.mean(
        pre_dict[
            test_data_name
        ]
    )


    pre_std_dict[
        test_data_name
    ] = np.std(
        pre_dict[
            test_data_name
        ]
    )


# =========================================================
# Print final results
# =========================================================
print("\n")
print(
    "#" * 120
)

print(
    "Final Results"
)

print(
    "#" * 120
)


for test_data_name in auc_mean_dict:

    str_result = (
        "AUROC:{:.4f}+-{:.4f}, "
        "AUPRC:{:.4f}+-{:.4f}"
    ).format(

        auc_mean_dict[
            test_data_name
        ],

        auc_std_dict[
            test_data_name
        ],

        pre_mean_dict[
            test_data_name
        ],

        pre_std_dict[
            test_data_name
        ],
    )


    print(
        "-" * 50
        + test_data_name
        + "-"
        * 50
    )

    print(
        "str_result",
        str_result
    )


# =========================================================
# Save configuration + results to txt
# =========================================================
save_results(

    result_file=args.result_file,

    model=model,

    model_config=model_config,

    train_config=train_config,

    datasets_train=datasets_train,

    datasets_test=datasets_test,

    auc_dict=auc_dict,

    pre_dict=pre_dict,

    auc_mean_dict=auc_mean_dict,

    auc_std_dict=auc_std_dict,

    pre_mean_dict=pre_mean_dict,

    pre_std_dict=pre_std_dict,

    args=args,
)


print("\n")
print(
    "Results saved to: {}".format(
        args.result_file
    )
)
