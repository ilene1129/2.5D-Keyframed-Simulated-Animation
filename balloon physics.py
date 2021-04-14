import pandas as pd
import sys
import csv
import numpy as np
import math

import Box2D
from Box2D.b2 import (chainShape, contactListener, pi, world, vec2, fixtureDef, polygonShape, circleShape, staticBody, dynamicBody)
import pygame
from pygame.locals import (QUIT)


def update_body(body, force_magnitude, frame_num, stroke_num, poly):
    if poly:
        next_data = pd.read_csv("frame"+str(frame_num + 1)+"stroke" +str(stroke_num)+ ".csv", delimiter = ",", header = None)
        next_vecs = data_to_vecs_poly(next_data)
        shape = body.fixtures[0].shape
        set_lin_vel(next_vecs, body, shape, force_magnitude, poly=True)


def update_chain(chain, force_magnitude, frame_num, stroke_num):
    next_data = pd.read_csv("frame"+str(frame_num + 1)+"stroke" +str(stroke_num)+ ".csv", delimiter = ",", header = None)
    next_vecs = data_to_vecs(next_data)

    for ind in range(len(chain)):
        body = chain[ind]
        shape = body.fixtures[0].shape
        vel = tuple(map(lambda i, j: (1 - force_magnitude) * TARGET_FPS * (i - j), next_vecs[ind], body.transform * shape.pos))
        body.linearVelocity = (vel)
    

def output_chain(chain, frame_num, stroke_num):
    output = []

    for body in chain:
        shape = body.fixtures[0].shape
        screen_pos = body.transform * shape.pos
        output.append(screen_pos)

    with open("/Users/ilenee/Documents/2020-2021/Thesis/3:2/output_frame"+str(frame_num)+"stroke"+str(stroke_num)+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(output)


def output_body(body, frame_num, stroke_num):
    data = pd.read_csv("frame"+str(frame_num)+"stroke" +str(stroke_num)+ ".csv", delimiter = ",", header = None)
    vecs = data_to_vecs_poly(data)
    # multiplier = math.floor(len(next_data) / len(next_vecs))
    # shape = body.fixtures[0].shape

    output = []

    for ind in range(len(vecs)):
        vert = vecs[ind]
        screen_coords = body.transform * vert
        # if ind % multiplier == 0:
        #     output.append()
        #     vert = shape.vertices[int(ind / multiplier)]
        #     # translate
        #     screen_coords = body.transform * vert
        # else:
        #     prev_ind = math.floor(ind / multiplier)
        #     next_ind = prev_ind + 1
        #     # check for overflow
        #     if next_ind * multiplier > len(next_data) - 1:
        #         vert = shape.next_data[ind]
        #         screen_coords = body.transform * vert
        #         output.append()
        # for mult_ind in range(1, multiplier):
        #     if ind == len(shape.vertices) - 1:
        #         output.append(screen_coords)
        #     else:
        #         next_screen_coords = body.transform * shape.vertices[ind + 1]
        #         output.append(mult_ind / multiplier * next_screen_coords + (1 - mult_ind / multiplier) * screen_coords)
        output.append(screen_coords)

    with open("/Users/ilenee/Documents/2020-2021/Thesis/3:2/output_frame"+str(frame_num)+"stroke"+str(stroke_num)+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(output)


def data_to_vecs(data):
    output = []
    for _, row in data.iterrows():
        output.append((row[0], row[1]))
    return output


def data_to_vecs_poly(data):
    output = []
    for ind, row in data.iterrows():
        if ind % math.ceil(len(data) / 16) == 0:
            output.append((row[0], row[1]))
    return output


def set_lin_vel(next_data, body, shape, force_magnitude, poly):
    next_avg = np.average(next_data, axis=0)
    curr_avg = [0,0]

    for vertex in shape.vertices:
        curr_avg = np.add(curr_avg, body.transform * vertex)

    curr_avg = curr_avg / len(shape.vertices)

    vel = tuple(map(lambda i, j: (1 - force_magnitude) * TARGET_FPS * (i - j), next_avg, curr_avg))
    body.linearVelocity = (vel)

# constant
SNOWBALL_RADIUS = 0.001
PPM = 50.0  # pixels per meter
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

# load balloons
balloon1 = pd.read_csv("frame0stroke0.csv", delimiter = ",", header = None)
balloon1 = data_to_vecs_poly(balloon1)
# print(balloon1)
balloon1_knot = pd.read_csv("frame0stroke1.csv", delimiter = ",", header = None)
balloon1_knot = data_to_vecs_poly(balloon1_knot)

balloon2 = pd.read_csv("frame0stroke3.csv", delimiter = ",", header = None)
balloon2 = data_to_vecs_poly(balloon2)

balloon2_knot = pd.read_csv("frame0stroke4.csv", delimiter = ",", header = None)
balloon2_knot = data_to_vecs_poly(balloon2_knot)

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
            # stop[ind] = -1
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass

# construct world
world = world(gravity=(0,0), doSleep=True, contactListener=myContactListener())

# add balloons
balloon1_shape = polygonShape(vertices=balloon1)
balloon1_knot_shape = polygonShape(vertices=balloon1_knot)
balloon2_shape = polygonShape(vertices=balloon2)
balloon2_knot_shape = polygonShape(vertices=balloon2_knot)

balloon1_body = world.CreateDynamicBody(position=(0,0))
fixture = balloon1_body.CreateFixture(fixtureDef(shape=balloon1_shape, density=1, friction=0.3))
balloon1_knot_body = world.CreateDynamicBody(position=(0,0))
fixture = balloon1_knot_body.CreateFixture(fixtureDef(shape=balloon1_knot_shape, density=1, friction=0.3))

balloon2_body = world.CreateDynamicBody(position=(0,0))
fixture = balloon2_body.CreateFixture(fixtureDef(shape=balloon2_shape, density=1, friction=0.3))
balloon2_knot_body = world.CreateDynamicBody(position=(0,0))
fixture = balloon2_knot_body.CreateFixture(fixtureDef(shape=balloon2_knot_shape, density=1, friction=0.3))

# weld joints
world.CreateWeldJoint(bodyA=balloon1_body, bodyB=balloon1_knot_body, anchor=balloon1_knot[0])
world.CreateWeldJoint(bodyA=balloon2_body, bodyB=balloon2_knot_body, anchor=balloon2_knot[0])

# add borders (for debugging)
border_shape_h = polygonShape(vertices = [(-0.75, 0), (-0.75, 0.01), (0.75, 0.01), (0.75, 0)])
border_shape_v = polygonShape(vertices = [(0, -0.6), (0.01, -0.6), (0.01, 0.6), (0, 0.6)])

# border_body1 = world.CreateStaticBody(position=(0,0.39), shapes=border_shape_h)

# border_body2 = world.CreateStaticBody(position=(0,-0.4), shapes=border_shape_h)

# border_body3 = world.CreateStaticBody(position=(0.59,0), shapes=border_shape_v)

# border_body4 = world.CreateStaticBody(position=(-0.6,0), shapes=border_shape_v)

# env = [env_body, border_body1, border_body2, border_body3, border_body4]
# env = [env_body]
# env = []

# create strings
string1 = data_to_vecs(pd.read_csv("frame0stroke2.csv", delimiter = ",", header = None))
chain1 = []
for ind in range(len(string1)):
    row = string1[ind]
    circle = circleShape(pos=(0,0), radius=0.001)
    new_body = world.CreateDynamicBody(position=row, userData=ind)
    fixture = new_body.CreateFixture(fixtureDef(shape=circle, density=1, friction=0.3))
    chain1.append(new_body)

    if ind > 0:
        world.CreateRevoluteJoint(bodyA=chain1[ind-1], bodyB=new_body, anchor=chain1[ind-1].worldCenter)

world.CreateRevoluteJoint(bodyA=chain1[0], bodyB=balloon1_knot_body, anchor=chain1[0].worldCenter)

string2 = data_to_vecs(pd.read_csv("frame0stroke5.csv", delimiter = ",", header = None))
chain2 = []
for ind in range(len(string2)):
    row = string2[ind]
    circle = circleShape(pos=(0,0), radius=0.001)
    new_body = world.CreateDynamicBody(position=row, userData=ind)
    fixture = new_body.CreateFixture(fixtureDef(shape=circle, density=1, friction=0.3))
    chain2.append(new_body)

    if ind > 0:
        world.CreateRevoluteJoint(bodyA=chain2[ind-1], bodyB=new_body, anchor=chain2[ind-1].worldCenter)

world.CreateRevoluteJoint(bodyA=chain2[0], bodyB=balloon2_knot_body, anchor=chain2[0].worldCenter)

force_magnitude = 1
force_decrement = force_magnitude / (end_frame - start_frame - 1)
# force_decrement = 0
force_vec = vec2(0.001, 0)

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
    for body in world.bodies:
        if body not in chain1 and body not in chain2:
            for fixture in body.fixtures:
                shape = fixture.shape
                draw_vertices = [(body.transform * v) * PPM for v in shape.vertices]
                # print("*", body.transform * shape.vertices[0])
                draw_vertices = [(v[0] + H_OFFSET, SCREEN_HEIGHT - (v[1] + V_OFFSET)) for v in draw_vertices]
                pygame.draw.polygon(screen, (255, 255, 255, 255), draw_vertices)

    for index in range(len(chain1)):
        link = chain1[index]
        for fixture in link.fixtures:
            shape = fixture.shape
            # print(snowflake_body.transform * shape.pos)
            draw_pos = (link.transform * shape.pos) * PPM
            draw_pos = (draw_pos[0] + H_OFFSET, SCREEN_HEIGHT - (draw_pos[1] + V_OFFSET))
            pygame.draw.circle(screen, (255, 255, 255, 255), (int(draw_pos[0]), int(draw_pos[1])), int(shape.radius * PPM))

    for index in range(len(chain2)):
        link = chain2[index]
        for fixture in link.fixtures:
            shape = fixture.shape
            # print(snowflake_body.transform * shape.pos)
            draw_pos = (link.transform * shape.pos) * PPM
            draw_pos = (draw_pos[0] + H_OFFSET, SCREEN_HEIGHT - (draw_pos[1] + V_OFFSET))
            pygame.draw.circle(screen, (255, 255, 255, 255), (int(draw_pos[0]), int(draw_pos[1])), int(shape.radius * PPM))


    output_body(balloon1_body, frame, 0)
    output_body(balloon1_knot_body, frame, 1)
    output_chain(chain1, frame, 2)
    output_body(balloon2_body, frame, 3)
    output_body(balloon2_knot_body, frame, 4)
    output_chain(chain2, frame, 5)

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

    for body in world.bodies:
        body.ApplyForce(force_vec * force_magnitude, body.position, True)

    force_magnitude = max(0, force_magnitude - force_decrement)

    # calculate and set linear velocities of balloons
    update_body(balloon1_body, force_magnitude, frame, 0, True)
    update_body(balloon1_knot_body, force_magnitude, frame, 1, True)
    update_body(balloon2_body, force_magnitude, frame, 3, True)
    update_body(balloon2_knot_body, force_magnitude, frame, 4, True)
    
    update_chain(chain1, force_magnitude, frame, 2)
    update_chain(chain2, force_magnitude, frame, 5)

    frame = frame + 1

pygame.quit()
print('Done!')