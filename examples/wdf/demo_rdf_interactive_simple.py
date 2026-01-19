"""
Simple Interactive RDF Visualization using Matplotlib
=====================================================

A simpler version using matplotlib sliders (no PyQt required).
The visualization opens in a separate trimesh window.

Dependencies:
- matplotlib
- trimesh
- pyglet
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from robocore.wdf.rdf import RDF
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path
from robocore.wdf.vis import plot_3D_sdf_envelope


class SimpleRDFVisualizer:
    """
    Simple interactive visualizer using matplotlib sliders.
    """
    
    def __init__(self, args):
        self.args = args
        
        # Initialize RDF
        print("Loading robot model...")
        asset_path = os.path.join(args.assetRoot, args.assetFile)
        self.robot = RobotModel(asset_path, base_link=args.baseLink, load_mesh_flag=True)
        
        print("Loading RDF model...")
        self.rdf_instant = RDF(args, self.robot, model_type=args.modelType)
        
        rdf_dir = os.path.join(os.path.dirname(asset_path), "rdf")
        if args.modelType == "NN":
            model_name = f'NN_h{args.hiddenDim}_e{args.trainEpochs}'
            rdf_model_path = os.path.join(rdf_dir, 'NN', f'{model_name}.pt')
        elif args.modelType == "BP":
            model_name = f'BP_{args.numFuncs}'
            rdf_model_path = os.path.join(rdf_dir, 'BP', f'{model_name}.pt')
        
        if not os.path.exists(rdf_model_path):
            print(f"Model not found at {rdf_model_path}")
            print("Please train the model first using demo_rdf.py")
            return
        
        if args.device == 'cpu':
            self.rdf_model = torch.load(rdf_model_path, map_location=torch.device('cpu'),
                                       weights_only=False)
        else:
            self.rdf_model = torch.load(rdf_model_path, weights_only=False)
        
        print("Model loaded successfully!")
        
        # Joint angles
        self.num_joints = self.robot.num_joints
        self.joint_angles = np.zeros(self.num_joints)
        
        # Use default joint limits (most robots use -pi to pi)
        self.joint_limits = [(-3.14, 3.14)] * self.num_joints
        
        # Visualization parameters
        self.nbData = args.resolution
        self.distance_levels = args.distanceLevels
        self.show_robot = args.showRobot
        
        # Track viewer window
        self.viewer_window = None
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        """Create matplotlib UI with sliders."""
        # Create figure with appropriate size for sliders
        fig_height = min(3 + self.num_joints * 0.5, 12)
        self.fig, self.ax = plt.subplots(figsize=(10, fig_height))
        self.fig.canvas.manager.set_window_title('RDF Interactive Control')
        
        # Hide the main axes (we only use it for layout)
        self.ax.axis('off')
        
        # Add title
        self.ax.text(0.5, 0.95, 'Robot Distance Field Interactive Visualizer',
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    transform=self.ax.transAxes)
        
        # Add instructions
        instructions = (
            'Adjust joint angles using sliders below.\n'
            'Click "Update Visualization" to see the distance field.\n'
            'Use "Reset" to return to zero configuration.'
        )
        self.ax.text(0.5, 0.88, instructions,
                    ha='center', va='top', fontsize=9,
                    transform=self.ax.transAxes, style='italic')
        
        # Create sliders
        self.sliders = []
        slider_height = 0.03
        slider_spacing = 0.05
        start_y = 0.75
        
        for i in range(self.num_joints):
            # Slider position
            slider_y = start_y - i * slider_spacing
            slider_ax = self.fig.add_axes([0.15, slider_y, 0.7, slider_height])
            
            # Get limits for this joint
            min_angle, max_angle = self.joint_limits[i]
            
            # Create slider
            slider = Slider(
                slider_ax,
                f'Joint {i+1}',
                min_angle,
                max_angle,
                valinit=0.0,
                valstep=0.01
            )
            
            # Connect update function
            slider.on_changed(lambda val, idx=i: self.update_joint(idx, val))
            
            self.sliders.append(slider)
        
        # Add buttons
        button_y = start_y - (self.num_joints + 0.5) * slider_spacing
        
        # Update button
        update_ax = self.fig.add_axes([0.15, button_y, 0.25, 0.04])
        self.update_button = Button(update_ax, 'Update Visualization', color='lightblue')
        self.update_button.on_clicked(self.visualize)
        
        # Reset button
        reset_ax = self.fig.add_axes([0.45, button_y, 0.15, 0.04])
        self.reset_button = Button(reset_ax, 'Reset', color='lightcoral')
        self.reset_button.on_clicked(self.reset)
        
        # Random button
        random_ax = self.fig.add_axes([0.65, button_y, 0.2, 0.04])
        self.random_button = Button(random_ax, 'Random Config', color='lightgreen')
        self.random_button.on_clicked(self.random_config)
        
        # Status text
        status_y = button_y - 0.08
        self.status_text = self.fig.text(
            0.5, status_y,
            'Ready. Click "Update Visualization" to start.',
            ha='center', fontsize=10, color='green'
        )
        
        plt.subplots_adjust(bottom=0.05, top=0.95)
    
    def update_joint(self, joint_idx, value):
        """Update joint angle from slider."""
        self.joint_angles[joint_idx] = value
    
    def reset(self, event=None):
        """Reset all joints to zero."""
        for slider in self.sliders:
            slider.set_val(0.0)
        self.joint_angles = np.zeros(self.num_joints)
        self.status_text.set_text('Joints reset to zero configuration.')
        self.status_text.set_color('blue')
        self.fig.canvas.draw_idle()
    
    def random_config(self, event=None):
        """Set random joint configuration."""
        for i, (slider, (min_angle, max_angle)) in enumerate(zip(self.sliders, self.joint_limits)):
            random_angle = np.random.uniform(min_angle, max_angle)
            slider.set_val(random_angle)
        self.status_text.set_text('Random configuration set.')
        self.status_text.set_color('blue')
        self.fig.canvas.draw_idle()
    
    def visualize(self, event=None):
        """Generate and show distance field visualization."""
        self.status_text.set_text('Generating visualization... (this may take a moment)')
        self.status_text.set_color('orange')
        self.fig.canvas.draw_idle()
        plt.pause(0.01)  # Force update
        
        try:
            print(f"\n{'='*60}")
            print(f"Generating RDF for joint angles:")
            print(f"{self.joint_angles}")
            print(f"Resolution: {self.nbData}^3, Levels: {self.distance_levels}")
            print(f"{'='*60}")
            
            print("Generating distance field envelope...")
            
            # Close previous window if exists
            if self.viewer_window is not None:
                try:
                    self.viewer_window.close()
                    print("Closed previous visualization window")
                except:
                    pass
                self.viewer_window = None
            
            # Generate scene without showing it
            scene = plot_3D_sdf_envelope(
                self.joint_angles,
                self.rdf_instant,
                model=self.rdf_model,
                device=self.args.device,
                nbData=self.nbData,
                distance_levels=self.distance_levels,
                show_robot=self.show_robot,
                return_scene=True  # Return scene instead of showing
            )
            
            # Show the scene in a new window
            self.viewer_window = scene.show()
            self.status_text.set_text('New visualization window opened!')
            self.status_text.set_color('green')
            print("New visualization window opened!")
            
        except Exception as e:
            error_msg = f'Error: {str(e)}'
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            self.status_text.set_text(error_msg)
            self.status_text.set_color('red')
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the matplotlib window."""
        # Show initial visualization
        print("\nShowing initial visualization...")
        self.visualize()
        
        # Show control panel
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Simple interactive RDF visualization'
    )
    
    # Device
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       type=str, help='Device (cuda:0 or cpu)')
    
    # Training args (from demo_rdf.py)
    parser.add_argument('--trainEpochs', default=300, type=int, help="Epochs for NN training")
    parser.add_argument('--forceTrain', action='store_true', help="Force training even if model exists")
    parser.add_argument('--saveMeshDict', action='store_false', help="Save trained NN models")
    parser.add_argument('--samplePoints', action='store_false', help="Force resampling points (if not existing)")
    parser.add_argument('--parallel', action='store_true', help="Use multiprocessing for point sampling")
    parser.add_argument('--domainMax', default=1.0, type=float)
    parser.add_argument('--domainMin', default=-1.0, type=float)
    parser.add_argument('--modelType', default="NN", type=str, help="Model type: BP or NN")
    
    # NN specific args
    parser.add_argument('--hiddenDim', default=256, type=int, help="Hidden dimension size for SDF NN")
    parser.add_argument('--learningRate', default=1e-4, type=float, help="Learning rate for NN training")
    parser.add_argument('--nnBatchSize', default=819200, type=int, help="Batch size for NN training")
    
    # BP specific args
    parser.add_argument('--numFuncs', default=16, type=int)
    
    # Asset args
    parser.add_argument('--assetName', default="Bruce", type=str, help="Name of the asset")
    parser.add_argument('--assetRoot', default=get_robocore_path("assets"),
                       type=str, help="Root directory for assets")
    parser.add_argument('--assetFile',
                       default="robot/mjcf/Bessica-D_v1_0/Bessica-D_Covered.xml",
                       type=str, help="Path to asset file (URDF/MJCF)")
    parser.add_argument('--baseLink', default="base_link", type=str, help="Base link of the robot")
    
    # Visualization args
    parser.add_argument('--resolution', default=48, type=int,
                       help="Grid resolution (lower=faster)")
    parser.add_argument('--distanceLevels', default=[0.05, 0.1, 0.15],
                       type=float, nargs='+')
    parser.add_argument('--showRobot', default=True, type=bool)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Simple RDF Interactive Visualizer")
    print("="*60)
    print(f"Robot: {args.assetFile}")
    print(f"Model: {args.modelType}")
    print(f"Resolution: {args.resolution}^3")
    print(f"Distance levels: {args.distanceLevels}")
    print("="*60)
    
    visualizer = SimpleRDFVisualizer(args)
    visualizer.show()


if __name__ == '__main__':
    main()

