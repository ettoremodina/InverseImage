"""
Neural Cellular Automata (NCA) - Learning to Grow

A modular implementation of Neural Cellular Automata for learning to grow
target images from a single seed cell.
"""

from .config import Config, ResolutionStage
from .model import CAModel
from .data import load_image, prepare_image, create_seed
from .training import Trainer, ProgressiveTrainer
from .visualization import display_animation, save_animation, save_training_gif
