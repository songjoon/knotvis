import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_orthogonal_projection(points, lines):
    fig, ax = plt.subplots()

    # Draw lines
    for line in lines:
        (x1, y1), (x2, y2) = points[line[0]], points[line[1]]
        ax.plot([x1, x2], [y1, y2], 'k-')

    # Draw points
    for point in points:
        ax.plot(point[0], point[1], 'ro')

    # Add patches to simulate hidden lines
    for line in lines:
        (x1, y1), (x2, y2) = points[line[0]], points[line[1]]
        if is_hidden(line, lines):
            ax.add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), color='white', linewidth=2))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def is_hidden(line, lines):
    # Implement logic to determine if the line is hidden by another line
    # This is a placeholder function and needs to be implemented based on specific requirements
    return False

# Example usage
points = [(0, 0), (1, 1), (2, 0), (1, -1), (1.3, 1.5)]
lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4,3)]
draw_orthogonal_projection(points, lines)