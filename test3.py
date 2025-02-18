import math
# Define Helper Functions
# Import Required Libraries
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def rotation_matrix_x(theta):
    """
    Create a rotation matrix for rotating around the x-axis by theta radians.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    """
    Create a rotation matrix for rotating around the y-axis by theta radians.
    """
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    """
    Create a rotation matrix for rotating around the z-axis by theta radians.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
def connect_points_in_order(n):
    """
    Connect points in the given order and return the lines connecting them.
    """
    return [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
def connect_all_points(n):
    """
    Connect every point to every other point and return the lines connecting them.
    """
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def line_intersection(A, B, C, D):
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (A[0] - B[0], C[0] - D[0])
    ydiff = (A[1] - B[1], C[1] - D[1])

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('Lines do not intersect')

    d = (det(A, B), det(C, D))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
def first(x,y,z, line):
    return (x[line[0]], y[line[0]], z[line[0]])
def second(x,y,z, line):
    return (x[line[1]], y[line[1]], z[line[1]])
def first_2d(x,y, line):
    return (x[line[0]], y[line[0]])
def second_2d(x,y, line):
    return (x[line[1]], y[line[1]])
@profile
def show_2d(x, y, z, lines, ax_2d):
    ax_2d.cla()
    ax_2d.scatter(x, y, c='r', marker='o')

    # Annotate each point with its index
    for idx, (x_coord, y_coord) in enumerate(zip(x, y)):
        ax_2d.annotate(idx, (x_coord, y_coord), textcoords="offset points", xytext=(0, 10), ha='center')


    def calculate_z_coordinate(intersection_point,A,B):
        z_A = A[2]
        z_B = B[2]
        if A[0] != B[0]:
            return z_A + (z_B - z_A) * ((intersection_point[0] - A[0]) / (B[0] - A[0]))
        else:
            return z_A + (z_B - z_A) * ((intersection_point[1] - A[1]) / (B[1] - A[1]))

    overlapping_lines = [[] for _ in range(len(lines))]
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            if(lines[i][0] in lines[j] or lines[i][1] in lines[j]):
                continue
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) >= (B[1] - A[1]) * (C[0] - A[0])
            A = first(x,y,z, lines[i])
            B = second(x,y,z, lines[i])
            C = first(x,y,z, lines[j])
            D = second(x,y,z, lines[j])
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                inter = line_intersection(A, B, C, D)
                z1 = calculate_z_coordinate(inter, A,B)
                z2 = calculate_z_coordinate(inter, C,D)
                if z1 > z2:
                    overlapping_lines[j].append(inter)
                else:
                    overlapping_lines[i].append(inter)

    for i, line in enumerate(overlapping_lines):
        if first_2d(x,y,lines[i])[0] > first_2d(x,y,lines[i])[1]:
            overlapping_lines[i] = sorted(line, key=lambda point: point[0], reverse=True)
        else:
            overlapping_lines[i] = sorted(line, key=lambda point: point[0])

    for i, l in enumerate(overlapping_lines):
        start_point = first_2d(x,y, lines[i])
        end_point = second_2d(x,y, lines[i])
        if len(l) == 0:
            ax_2d.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='b', linewidth=1)
        else:
            def calculate_inner_point(A, B, l):
                size = math.sqrt((A[0]-B[0]) ** 2 + (A[1]-B[1]) ** 2)
                if(size == 0):
                    return A
                l = 0.15
                return (A[0] + l * (B[0] - A[0]) / size, A[1] + l * (B[1] - A[1]) / size)

            t01 = calculate_inner_point(l[0],start_point, 0.2)
            t02 = calculate_inner_point(l[len(l) - 1], end_point, 0.2)
            ax_2d.plot((start_point[0], t01[0]), (start_point[1], t01[1]), color='b', linewidth=1)
            for index in range(len(l) - 1):
                t1 = calculate_inner_point(l[index], l[index + 1], 0.2)
                t2 = calculate_inner_point(l[index + 1],l[index], 0.2)
                ax_2d.plot((t1[0], t2[0]), (t1[1], t2[1]), color='b', linewidth=1)
                start_point = l[index]

            ax_2d.plot((t02[0], end_point[0]), (t02[1], end_point[1]), color='b', linewidth=1)
    ax_2d.set_title('2D Projection of 3D Points')
    ax_2d.set_axis_off()
    ax_2d.set_xlim(-3, 3)
    ax_2d.set_ylim(-3, 3)
    return ax_2d
@profile
def show_3d(x,y,z,lines,ax1):
    ax1.cla()
    ax1.scatter(x, y, z, c='r', marker='o')
    ax1.set_title('3D Projection of Points')

    # Annotate each point with its index
    for idx, (x_coord, y_coord, z_coord) in enumerate(zip(x, y, z)):
        ax1.text(x_coord, y_coord, z_coord, '%d' % idx, size=10, zorder=1, color='k')

    ax1.set_axis_off()
    # Connect the points with lines
    ax1.plot(x, y, z, color='b', linewidth=1)
    for (i,j) in lines:
        ax1.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='b', linewidth=1)



from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

# Generate random points in 3D
points = [(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(15)]
three_points = [
    (1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)
]
# Define the vertices of a cube
four_points = [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, 1, -1), (1, 1, 1), (1, -1, -1), (1, -1, 1), 
]


phi = (1 + np.sqrt(5)) / 2  # Golden ratio
five_points = [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    (0, -1/phi, -phi), (0, -1/phi, phi), (0, 1/phi, -phi), (0, 1/phi, phi),
    (-1/phi, -phi, 0), (-1/phi, phi, 0), (1/phi, -phi, 0), (1/phi, phi, 0),
    (-phi, 0, -1/phi), (-phi, 0, 1/phi), (phi, 0, -1/phi), (phi, 0, 1/phi)
]
three_points = [np.dot(rotation_matrix_x(np.pi/4 + 2), np.dot(rotation_matrix_y(np.pi/4), np.dot(rotation_matrix_z(np.pi/4),point))) for point in three_points]
four_points = [np.dot(rotation_matrix_x(np.pi/4 + 2), np.dot(rotation_matrix_y(np.pi/4), np.dot(rotation_matrix_z(np.pi/4),point))) for point in four_points]
five_points = [np.dot(rotation_matrix_x(np.pi/4 + 2), np.dot(rotation_matrix_y(np.pi/4), np.dot(rotation_matrix_z(np.pi/4),point))) for point in five_points]
# Sort points to follow the surfaces
sorted_points = sorted(points, key=lambda p: (p[0], p[1], p[2]))
points = sorted_points
x, y, z = zip(*points)

# Scatter plot of the points
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
ax1 = fig.add_subplot(231, projection='3d')
ax2 = fig.add_subplot(234)
ax3 = fig.add_subplot(232, projection='3d')
ax4 = fig.add_subplot(235)
ax5 = fig.add_subplot(233, projection='3d')
ax6 = fig.add_subplot(236)
# Function to update the plot
@profile
def update(num, x, y, z, ax1, ax2,ax3,ax4,ax5,ax6):
    theta = num/ 700 * 2 * np.pi
    R_x = rotation_matrix_x(1.3*theta)
    R_y = rotation_matrix_y(2.1 * theta)
    R_z = rotation_matrix_z(4.4 * theta)
    rotated_three_points = [np.dot(R_z, np.dot(R_y, np.dot(R_x, point))) for point in three_points]

    show_3d(*zip(*rotated_three_points),connect_points_in_order(len(three_points)),ax1)
    show_2d(*zip(*rotated_three_points), connect_points_in_order(len(three_points)), ax2)
    rotated_four_points = [np.dot(R_z, np.dot(R_y, np.dot(R_x, point))) for point in four_points]

    show_3d(*zip(*rotated_four_points),connect_points_in_order(len(four_points)),ax3)
    show_2d(*zip(*rotated_four_points), connect_points_in_order(len(four_points)), ax4)


    rotated_five_points = [np.dot(R_z, np.dot(R_y, np.dot(R_x, point))) for point in five_points]
    show_3d(*zip(*rotated_five_points),connect_points_in_order(len(five_points)),ax5)
    show_2d(*zip(*rotated_five_points), connect_points_in_order(len(five_points)), ax6)

    
    return ax1, ax2, ax3, ax4, ax5, ax6

# Create an animation
ani = FuncAnimation(fig, update, frames=50, fargs=(x, y, z, ax1, ax2,ax3,ax4,ax5,ax6), interval=100)

#plt.show()
# Save the animation as a video file
# Save the animation as a video file with GPU acceleration
writer = FFMpegWriter(fps=30)
ani.save('knot_theory_visualization.mp4', writer=writer)