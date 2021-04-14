import pandas as pd
import sys
import csv
import numpy as np

import Box2D
from Box2D.b2 import (contactListener, pi, world, vec2, fixtureDef, polygonShape, circleShape, staticBody, dynamicBody)
import pygame
from pygame.locals import (QUIT)


def data_to_vecs(data):
    output = []
    for _, row in data.iterrows():
        output.append((row[0], row[1]))
    return output


# constant
SNOWBALL_RADIUS = 0.001
PPM = 400.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
H_OFFSET = 320
V_OFFSET = 240

# setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Animation')
clock = pygame.time.Clock()

start_frame = int(sys.argv[1])
end_frame = int(sys.argv[2])

# load metadata
metadata = pd.read_csv("metadata.csv", delimiter = ",", header = None)
metadata = data_to_vecs(metadata)

# load initial stroke data
data = pd.read_csv(str(start_frame) + ".csv", delimiter = ",", header = None)
data = data_to_vecs(data)

stop = []
for ind in range(len(data)):
    stop.append(-2)

class myContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)
    def BeginContact(self, contact):
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        to_stop = False

        if fixture_a.body.type == 2:
            body = fixture_a.body
            if fixture_b.body.type == 0:
                to_stop = True
        elif fixture_b.body.type == 2:
            body = fixture_b.body
            to_stop = True

        if to_stop:
            ind = int(body.userData)
            stop[ind] = -1
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass

# construct world
world = world(gravity=(0,0), doSleep=True, contactListener=myContactListener())

# add static body
env_shape = polygonShape(vertices=metadata)
# env_body = world.CreateStaticBody(position=(0,0), shapes=env_shape)

# add borders (for debugging)
border_shape_h = polygonShape(vertices = [(-0.75, 0), (-0.75, 0.01), (0.75, 0.01), (0.75, 0)])
border_shape_v = polygonShape(vertices = [(0, -0.6), (0.01, -0.6), (0.01, 0.6), (0, 0.6)])

# border_body1 = world.CreateStaticBody(position=(0,0.39), shapes=border_shape_h)

# border_body2 = world.CreateStaticBody(position=(0,-0.4), shapes=border_shape_h)

# border_body3 = world.CreateStaticBody(position=(0.59,0), shapes=border_shape_v)

# border_body4 = world.CreateStaticBody(position=(-0.6,0), shapes=border_shape_v)

# env = [env_body, border_body1, border_body2, border_body3, border_body4]
# env = [env_body]
env = []

# create dynamic bodies
snowflakes = []

for ind in range(len(data)):
    row = data[ind]
    circle = circleShape(pos=(0,0), radius=SNOWBALL_RADIUS)
    snowflake_body = world.CreateDynamicBody(position=row, userData=ind)
    fixture = snowflake_body.CreateFixture(fixtureDef(shape=circle, density=1, friction=0.3))
    snowflakes.append(snowflake_body)

force_magnitude = 1
force_decrement = force_magnitude / (end_frame - start_frame - 1)
# force_decrement = 0
force_vec = vec2(0.0001, 0)

# main game loop
running = True
frame = start_frame
while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT:
            # The user closed the window
            running = False

    screen.fill((0, 0, 0, 0))

    # Draw the world
    for body in env:
        for fixture in body.fixtures:
            shape = fixture.shape
            draw_vertices = [(body.transform * v) * PPM for v in shape.vertices]
            # print("*", body.transform * shape.vertices[0])
            draw_vertices = [(v[0] + H_OFFSET, SCREEN_HEIGHT - (v[1] + V_OFFSET)) for v in draw_vertices]
            pygame.draw.polygon(screen, (255, 255, 255, 255), draw_vertices)

    for index in range(len(snowflakes)):
        snowflake_body = snowflakes[index]
        for fixture in snowflake_body.fixtures:
            shape = fixture.shape
            # print(snowflake_body.transform * shape.pos)
            draw_pos = (snowflake_body.transform * shape.pos) * PPM
            draw_pos = (draw_pos[0] + H_OFFSET, SCREEN_HEIGHT - (draw_pos[1] + V_OFFSET))
            pygame.draw.circle(screen, (255, 255, 255, 255), (int(draw_pos[0]), int(draw_pos[1])), int(shape.radius * PPM))

    # Make Box2D simulate the physics of our world for one step.
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    # See the manual (Section "Simulating the World") for further discussion
    # on these parameters and their implications.
    world.Step(TIME_STEP, 10, 10)

    world.ClearForces()

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)
    if frame == end_frame:
        break

    print(snowflakes[0].linearVelocity)

    for body in world.bodies:
        body.ApplyForce(force_vec * force_magnitude, body.position, True)

    print(snowflakes[0].linearVelocity)
    force_magnitude = max(0, force_magnitude - force_decrement)

    next_data = pd.read_csv(str(frame + 1) + ".csv", delimiter = ",", header = None)
    next_data = data_to_vecs(next_data)

    # calculate and set linear velocities
    for index in range(len(data)):
        vel = tuple(map(lambda i, j: (1 - force_magnitude) * TARGET_FPS * (i - j), next_data[index], snowflakes[index].transform * snowflakes[index].fixtures[0].shape.pos))
        if stop[index] == -2:
            snowflakes[index].linearVelocity = (vel)
            # print(next_data[index])
            # print(snowflakes[index].transform * snowflakes[index].fixtures[0].shape.pos)
            # print(vel)
            # print("*")
        else:
            snowflakes[index].linearVelocity = (0,0)
            if stop[index] == -1:
                stop[index] = str(frame)
            data[index] = stop[index]
    
    # physics_data.append(data)

    data = next_data
    frame = frame + 1

# output intermediate data
# print(physics_data)
# for ind in range(len(physics_data)):
#     with open("/Users/ilenee/Documents/2020-2021/Thesis/2_13/" + str(ind) + ".csv", "w") as f:
#         writer = csv.writer(f)
#         writer.writerows(physics_data[ind])

# for ind in range(len(stop)):
#     if stop[ind] == False:
#         stop[ind] = -1

print(stop)

stop_data = []
for item in stop:
    stop_data.append([item])

with open("/Users/ilenee/Documents/2020-2021/Thesis/2_13/output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(stop_data)

pygame.quit()
print('Done!')