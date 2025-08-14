# -------------------------------
# FILE: main.py
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt
from robot_utils import init_robot_chain, solve_ik, compute_manipulability, compute_joint_limit, compute_singularity_metric
from sampling import create_grid, sample_cone_orientations
from evaluation import evaluate_position
from path_plan import choose_nearest_solution
from draw import draw_cuboid


# Parameters to configure
URDF_PATH = "iiwa14.urdf"
TARGET = (0.6, 0.0, 0.5) 
RADIUS_X = 0.2 # plus long en X
RADIUS_Y = 0.1
RADIUS_Z = 0.05
phantom_x = 0.06
phantom_y = 0.03
phantom_z = 0.03
STEP = 0.007 
AXIS7 = np.array([1, 0, 0])
HALF_ANGLE = np.deg2rad(30)
N_ORIENT = 1

# Entrées pour reshape
NX = int((2 * RADIUS_X) / STEP) + 1
NY = int((2 * RADIUS_Y) / STEP) + 1
NZ = int((2 * RADIUS_Z) / STEP) + 1

if __name__ == "__main__":
    # 1. Initialisation du robot
    chain = init_robot_chain(URDF_PATH)
    # 2. Création de la grille spatiale
    grid = create_grid(TARGET, (RADIUS_X, RADIUS_Y, RADIUS_Z), STEP)
    # 3. Échantillonnage des orientations du cône
    orientations = sample_cone_orientations(AXIS7, HALF_ANGLE, N_ORIENT)
    # 4. Évaluation de chaque point de la grille
    scores = []
    for pos in grid:
        score = evaluate_position(chain, pos, orientations)
        scores.append(score)

    # Convertir la liste en tableau numpy
    scores = np.array(scores)

    # Normalisation entre 0 et 1
    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score - min_score != 0:
        scores_normalized = (scores - min_score) / (max_score - min_score)
    else:
        scores_normalized = np.zeros_like(scores)

    # Reshape en 3D
    scores = scores_normalized.reshape((NX, NY, NZ))





    # 5. Visualisation

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=scores.ravel(), cmap='turbo', s=20)
    fig.colorbar(sc, ax=ax, label='Score')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Scatter Plot of Full Score Cuboid')

    

    # Afficher le robot dans cette configuration
    ik_solution = chain.inverse_kinematics(TARGET)
    #chain.plot(ik_solution, ax, target=TARGET)
    desired_last_joint = np.deg2rad(45)  # e.g., 45 degrees
    ik_solution[-1] = desired_last_joint

    # Plot
    chain.plot(ik_solution, ax, target=TARGET)

    # Afficher le phantom
    draw_cuboid(ax, TARGET, size=(2*phantom_x, 2*phantom_y, 2*phantom_z), alpha=0.15)


    # Définir les limites des axes
    ax.set_xlim(-0.9 , 0.9)
    ax.set_ylim(-0.60, 0.60)
    ax.set_zlim(0, 2)

    # Ajouter un repère XYZ plus visible
    frame_length = 0.05
    origin = np.array(TARGET)
    ax.quiver(*origin, frame_length, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.2)
    ax.quiver(*origin, 0, frame_length, 0, color='g', linewidth=2, arrow_length_ratio=0.2)
    ax.quiver(*origin, 0, 0, frame_length, color='b', linewidth=2, arrow_length_ratio=0.2)
    ax.text(*(origin + [frame_length, 0, 0]), 'X', color='r')
    ax.text(*(origin + [0, frame_length, 0]), 'Y', color='g')
    ax.text(*(origin + [0, 0, frame_length]), 'Z', color='b')

    plt.show()


    

