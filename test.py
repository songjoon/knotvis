import moderngl
import numpy as np
import pygame
from pygame.locals import *

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((800, 600), OPENGL | DOUBLEBUF)
ctx = moderngl.create_context()

# 쉐이더 프로그램
prog = ctx.program(
    vertex_shader="""
        #version 330
        in vec2 in_vert;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
    """
)

# 삼각형 정점 데이터
vertices = np.array([
    -0.6, -0.6,
     0.6, -0.6,
     0.0,  0.6,
], dtype='f4')

# 버퍼 객체 및 VAO 생성
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False
    
    ctx.clear(0.2, 0.3, 0.3)
    vao.render(moderngl.TRIANGLES)
    pygame.display.flip()

pygame.quit()
