"""
Interactive RDF Envelope Visualization with Joint Control
=========================================================

This demo provides an interactive GUI to control robot joint angles
and visualize the distance field envelope in real-time.

Dependencies:
- PyQt5 or PySide2 (for GUI)
- trimesh (for 3D visualization)
- pyglet (for 3D rendering)
"""

import sys
import argparse
import os
import numpy as np
import torch
import trimesh

from robocore.wdf.rdf import RDF
from robocore.modeling.robot_model import RobotModel
from robocore.utils.path import get_robocore_path

# Try PyQt5 first, fall back to PySide2
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox)
    from PyQt5.QtCore import Qt, QTimer
    USING_PYQT5 = True
except ImportError:
    try:
        from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                       QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox)
        from PySide2.QtCore import Qt, QTimer
        USING_PYQT5 = False
    except ImportError:
        print("Error: PyQt5 or PySide2 is required. Install with:")
        print("  pip install PyQt5")
        print("  or")
        print("  pip install PySide2")
        sys.exit(1)


class RDFInteractiveVisualizer(QMainWindow):
    """
    Interactive visualizer for RDF with joint angle control.
    """
    
    def __init__(self, args):
        super().__init__()
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
            sys.exit(1)
        
        if args.device == 'cpu':
            self.rdf_model = torch.load(rdf_model_path, map_location=torch.device('cpu'), 
                                       weights_only=False)
        else:
            self.rdf_model = torch.load(rdf_model_path, weights_only=False)
        
        print("Model loaded successfully!")
        
        # Joint angles (initialize to zero)
        self.num_joints = self.robot.num_joints
        self.joint_angles = np.zeros(self.num_joints)
        
        # Visualization parameters
        self.nbData = args.resolution
        self.distance_levels = args.distanceLevels
        self.show_robot = args.showRobot
        
        # Scene for visualization (will be updated)
        self.scene = None
        self.viewer_process = None
        
        # Setup UI
        self.initUI()
        
        # Timer for updating visualization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.auto_update = False  # Disabled by default to avoid multiple windows
        
        # Track viewer window
        self.viewer_scene = None
        self.viewer_window = None
        
    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle('RDF Interactive Visualizer')
        self.setGeometry(100, 100, 400, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel('Robot Distance Field Visualizer')
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # Joint control group
        joint_group = QGroupBox("Joint Angles (radians)")
        joint_layout = QVBoxLayout()
        
        self.joint_sliders = []
        self.joint_labels = []
        
        # Use default joint limits (most robots use -pi to pi)
        # If specific limits are needed, they can be set manually
        
        for i in range(self.num_joints):
            joint_layout.addWidget(QLabel(f'Joint {i+1}:'))
            
            # Create horizontal layout for slider and value label
            h_layout = QHBoxLayout()
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            
            # Set range (use standard -pi to pi range)
            min_angle, max_angle = -3.14, 3.14
            
            slider.setMinimum(int(min_angle * 1000))  # Scale by 1000 for precision
            slider.setMaximum(int(max_angle * 1000))
            slider.setValue(0)
            slider.valueChanged.connect(lambda val, idx=i: self.on_joint_changed(idx, val))
            
            # Value label
            value_label = QLabel('0.000')
            value_label.setMinimumWidth(60)
            
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            
            joint_layout.addLayout(h_layout)
            
            self.joint_sliders.append(slider)
            self.joint_labels.append(value_label)
        
        joint_group.setLayout(joint_layout)
        main_layout.addWidget(joint_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        reset_button = QPushButton('Reset to Zero')
        reset_button.clicked.connect(self.reset_joints)
        button_layout.addWidget(reset_button)
        
        random_button = QPushButton('Random Config')
        random_button.clicked.connect(self.random_joints)
        button_layout.addWidget(random_button)
        
        main_layout.addLayout(button_layout)
        
        # Visualization controls
        vis_group = QGroupBox("Visualization Controls")
        vis_layout = QVBoxLayout()
        
        update_button = QPushButton('Update Visualization')
        update_button.clicked.connect(self.manual_update)
        vis_layout.addWidget(update_button)
        
        auto_update_button = QPushButton('Toggle Auto-Update')
        auto_update_button.clicked.connect(self.toggle_auto_update)
        vis_layout.addWidget(auto_update_button)
        
        self.status_label = QLabel('Auto-update: OFF (Single window mode)')
        self.status_label.setStyleSheet("color: green;")
        vis_layout.addWidget(self.status_label)
        
        vis_group.setLayout(vis_layout)
        main_layout.addWidget(vis_group)
        
        # Info
        info_text = QLabel(
            'Adjust joint angles using sliders.\n'
            'Click "Update Visualization" to see changes.\n'
            '✅ Single window mode: Old window closes automatically.\n'
            'Only one 3D window will be open at a time.'
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: gray; font-size: 10px;")
        main_layout.addWidget(info_text)
        
        main_layout.addStretch()
    
    def on_joint_changed(self, joint_idx, value):
        """Handle joint slider change."""
        angle = value / 1000.0  # Convert back from scaled value
        self.joint_angles[joint_idx] = angle
        self.joint_labels[joint_idx].setText(f'{angle:.3f}')
        
        if self.auto_update:
            # Debounce: restart timer
            self.update_timer.stop()
            self.update_timer.start(500)  # Update after 500ms of no changes
    
    def reset_joints(self):
        """Reset all joints to zero."""
        for i, slider in enumerate(self.joint_sliders):
            slider.setValue(0)
    
    def random_joints(self):
        """Set random joint configuration."""
        for slider in self.joint_sliders:
            min_val = slider.minimum()
            max_val = slider.maximum()
            random_val = np.random.randint(min_val, max_val)
            slider.setValue(random_val)
    
    def toggle_auto_update(self):
        """Toggle auto-update mode."""
        self.auto_update = not self.auto_update
        if self.auto_update:
            self.status_label.setText('Auto-update: ON (Single window mode)')
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.status_label.setText('Auto-update: OFF (Single window mode)')
            self.status_label.setStyleSheet("color: green;")
            self.update_timer.stop()
    
    def manual_update(self):
        """Manually trigger visualization update."""
        self.update_visualization()
    
    def update_visualization(self):
        """Update the 3D visualization with current joint angles."""
        print(f"\n{'='*60}")
        print(f"Updating visualization for joint angles:")
        print(f"{self.joint_angles}")
        print(f"{'='*60}")
        
        try:
            # Generate envelope visualization
            from robocore.wdf.vis import plot_3D_sdf_envelope
            
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
            print("New visualization window opened!")
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive RDF visualization with joint control'
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
    parser.add_argument('--resolution', default=64, type=int,
                       help="Grid resolution (32/64/128)")
    parser.add_argument('--distanceLevels', default=[0.02, 0.05, 0.1, 0.15],
                       type=float, nargs='+',
                       help="Distance levels for envelope")
    parser.add_argument('--showRobot', default=True, type=bool)
    
    args = parser.parse_args()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = RDFInteractiveVisualizer(args)
    window.show()
    
    # Show initial visualization
    window.update_visualization()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

