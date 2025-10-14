"""
Robot distance field (RDF) with Neural Network
==============================================

This example demonstrates how to use the RDF_NN class to train a Neural Network model for the robot distance field
from URDF/MJCF files and visualize the reconstructed whole body.

Updated to use new multi-chain and backend support.
"""

import argparse
import os

import numpy as np
import torch

from robocore.modeling.robot_model import RobotModel
from robocore.utils.backend import set_backend
from robocore.wdf.rdf import RDF


def rdf_from_robot_model(args):
    assert args.modelType in ["NN", "BP"], "Invalid model type. Choose either 'NN' or 'BP'."

    # Set global backend
    backend_type = 'torch' if 'cuda' in args.device else 'numpy'
    set_backend(backend_type, device=args.device)

    asset_path = os.path.join(args.assetRoot, args.assetFile)

    # Create robot model with load_mesh_flag=True
    robot = RobotModel(
        asset_path,
        base_link=args.baseLink,
        end_link="Link6",  # Single chain mode
        load_mesh_flag=True
    )

    # Create RDF instance (no groups = single chain mode)
    rdf_instant = RDF(args, robot, model_type=args.modelType)
    rdf_dir = os.path.join(os.path.dirname(asset_path), "rdf")

    if args.modelType == "NN":
        model_name = f'NN_h{args.hiddenDim}_e{args.trainEpochs}'
        rdf_model_path = os.path.join(rdf_dir, 'NN', f'{model_name}.pt')
    elif args.modelType == "BP":
        model_name = f'BP_{args.numFuncs}'
        rdf_model_path = os.path.join(rdf_dir, 'BP', f'{model_name}.pt')

    if not os.path.exists(rdf_model_path) or args.forceTrain:  # train the model
        rdf_instant.train()

    if args.device == 'cpu':
        rdf_model = torch.load(rdf_model_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        rdf_model = torch.load(rdf_model_path, weights_only=False)

    rdf_instant.create_surface_mesh(rdf_model, nbData=128, vis=False, save_mesh_name=model_name)

    # Use num_chain_dof for the chain to end_link
    num_joint = rdf_instant.robot.num_chain_dof
    joint_value = np.zeros(num_joint)
    base_trans = np.eye(4)

    trans_dict = rdf_instant.robot.get_trans_dict(joint_value, base_trans)

    # Visualize (optional)
    # rdf_instant.visualize_reconstructed_whole_body(rdf_model, trans_dict, tag=model_name)

    # Run RDF_NN inference example
    import robolab
    robolab.wdf.plot_3D_sdf_with_gradient(joint_value, rdf_instant, model=rdf_model, device=args.device)


if __name__ == '__main__':
    from robocore.utils.path import get_robocore_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)

    # Training args
    parser.add_argument('--trainEpochs', default=300, type=int, help="Epochs for NN training")  # Keep for NN
    parser.add_argument('--forceTrain', action='store_true', help="Force training even if model exists")
    parser.add_argument('--saveMeshDict', action='store_false', help="Save trained NN models")  # Keep general name
    parser.add_argument('--samplePoints', action='store_false', help="Force resampling points (if not existing)")
    parser.add_argument('--parallel', action='store_true', help="Use multiprocessing for point sampling")
    parser.add_argument('--domainMax', default=1.0, type=float)
    parser.add_argument('--domainMin', default=-1.0, type=float)
    parser.add_argument('--modelType', default="NN", type=str)  # BP or NN
    # NN specific args
    parser.add_argument('--hiddenDim', default=256, type=int, help="Hidden dimension size for SDF NN")
    parser.add_argument('--learningRate', default=1e-4, type=float, help="Learning rate for NN training")
    parser.add_argument('--nnBatchSize', default=819200, type=int, help="Batch size for NN training")
    # BP specific args
    parser.add_argument('--numFuncs', default=16, type=int)

    # Asset args
    parser.add_argument('--assetName', default="Bruce", type=str, help="Name of the asset (e.g., for finding files)")
    parser.add_argument('--assetRoot', default=get_robocore_path("assets"),
                        type=str, help="Root directory for assets")
    parser.add_argument('--assetFile', default="robot/mjcf/Alicia-D_v5_5/alicia_duo_with_gripper.xml", type=str, help="Path to asset file (URDF/MJCF)")
    parser.add_argument('--baseLink', default="base_link", type=str, help="Base link of the robot")

    args = parser.parse_args()

    rdf_from_robot_model(args)
