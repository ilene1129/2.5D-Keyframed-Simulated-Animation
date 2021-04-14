import pandas as pd
import sys
import csv
import numpy as np
import math
import random

import Box2D
from Box2D.b2 import (chainShape, contactListener, pi, world, vec2, fixtureDef, polygonShape, circleShape, staticBody, dynamicBody)
import pygame
from pygame.locals import (QUIT)


CHAIN_MULTIPLIER = 5


def load_chain(world, data, dynamic):
    chain = []
    for ind in range(len(data)):
        row = data[ind]
        circle = circleShape(pos=(0,0), radius=0.001)
        if dynamic:
            new_body = world.CreateDynamicBody(position=row, userData=ind)
            new_body.CreateFixture(fixtureDef(shape=circle, density=1, friction=0.3))
            # if ind > 0:
            #     world.CreateRevoluteJoint(bodyA=chain[ind-1], bodyB=new_body, anchor=chain[ind-1].worldCenter)
        else:
            new_body = world.CreateStaticBody(position=row, shapes=[circle])
        chain.append(new_body)
    return chain

def update_body(body, force_magnitude, frame_num, stroke_num, poly):
    if poly:
        next_data = pd.read_csv("frame"+str(frame_num + 1)+"stroke" +str(stroke_num)+ ".csv", delimiter = ",", header = None)
        next_vecs = data_to_vecs_poly(next_data)
        shape = body.fixtures[0].shape
        set_lin_vel(next_vecs, body, shape, force_magnitude, poly=True)


def update_chain(chain, force_magnitude, frame_num, stroke_num):
    # print(frame_num, stroke_num)
    next_data = pd.read_csv("frame"+str(frame_num + 1)+"stroke" +str(stroke_num)+ ".csv", delimiter = ",", header = None)
    # if frame_num == 9 and stroke_num == 1:
    #     next_data = pd.read_csv("frame"+str(frame_num + 1)+"stroke" +str(2)+ ".csv", delimiter = ",", header = None)
    # elif frame_num == 9 and stroke_num == 2:
    #     next_data = pd.read_csv("frame"+str(frame_num + 1)+"stroke" +str(1)+ ".csv", delimiter = ",", header = None)
    next_vecs = data_to_vecs(next_data)[::CHAIN_MULTIPLIER]

    for ind in range(len(chain)):
        # print(len(next_vecs), len(chain))
        body = chain[ind]
        shape = body.fixtures[0].shape
        vel = tuple(map(lambda i, j: (1 - force_magnitude) * TARGET_FPS * (i - j), next_vecs[ind], body.transform * shape.pos))
        body.linearVelocity = (vel)
    

def output_chain(chain, frame_num, stroke_num):
    output = []

    for ind in range(len(chain)):
        body = chain[ind]
        shape = body.fixtures[0].shape
        if ind < len(chain) - 1:
            next_body_transform = chain[ind + 1].transform
        else:
            next_body_transform = body.transform

        for mult_ind in range(CHAIN_MULTIPLIER):
            factor = mult_ind / CHAIN_MULTIPLIER
            screen_pos = body.transform.position * factor + shape.pos + next_body_transform.position * (1 - factor)
            output.append(screen_pos)   

    with open("/Users/ilenee/Documents/2020-2021/Thesis/3_16/grass/output_frame"+str(frame_num)+"stroke"+str(stroke_num)+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(output)


def output_body(body, start_frame, frame_num, stroke_num):
    data = pd.read_csv("frame"+str(start_frame)+"stroke" +str(stroke_num)+ ".csv", delimiter = ",", header = None)
    vecs = data_to_vecs(data)
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

    with open("/Users/ilenee/Documents/2020-2021/Thesis/3_16/grass/output_frame"+str(frame_num)+"stroke"+str(stroke_num)+".csv", "w") as f:
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
PPM = 600.0  # pixels per meter
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

# construct world
world = world(gravity=(0,0), doSleep=True)

# add borders (for debugging)
border_shape_h = polygonShape(vertices = [(-0.75, 0), (-0.75, 0.01), (0.75, 0.01), (0.75, 0)])
border_shape_v = polygonShape(vertices = [(0, -0.6), (0.01, -0.6), (0.01, 0.6), (0, 0.6)])

border_body1 = world.CreateStaticBody(position=(0,0.39), shapes=border_shape_h)

ground_body = world.CreateStaticBody(position=(0,-0.4), shapes=border_shape_h)

border_body3 = world.CreateStaticBody(position=(0.49,0), shapes=border_shape_v)

border_body4 = world.CreateStaticBody(position=(-0.5,0), shapes=border_shape_v)

env = [border_body1, ground_body, border_body3, border_body4]

# load strokes
chains = []
static_chains = []
dynamic_chains = []
for ind in range(0, 73):
    # # print(ind)
    data = pd.read_csv("frame"+str(start_frame)+"stroke"+str(ind)+".csv", delimiter = ",", header = None)
    data = data_to_vecs(data)[::CHAIN_MULTIPLIER]
    chain = load_chain(world, data, dynamic=True)
    dynamic_chains.append(chain)
    # or chain[-1] ?
    # world.CreateWeldJoint(bodyA=ground_body, bodyB=chain[0], anchor=chain[0].worldCenter)
    chains.append(chain)

# env = [env_body]
# env = []

force_magnitude = 0.01
force_decrement = force_magnitude / (end_frame - start_frame - 1)
# force_decrement = 0
force_vec = vec2(0.01, 0.0)

# main game loop
running = True
frame = start_frame
PERIOD = 10
period_counter = 0

while running:
    # print("running", frame)
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT:
            # The user closed the window
            running = False 

    screen.fill((0, 0, 0, 0))

    # print("screen filled")

    for body in env:
        for fixture in body.fixtures:
            shape = fixture.shape
            draw_vertices = [(body.transform * v) * PPM for v in shape.vertices]
            # # print("*", body.transform * shape.vertices[0])
            draw_vertices = [(v[0] + H_OFFSET, SCREEN_HEIGHT - (v[1] + V_OFFSET)) for v in draw_vertices]
            pygame.draw.polygon(screen, (255, 255, 255, 255), draw_vertices)


    for chain in chains:
        for index in range(len(chain)):
            link = chain[index]
            for fixture in link.fixtures:
                shape = fixture.shape
                draw_pos = (link.transform * shape.pos) * PPM
                draw_pos = (draw_pos[0] + H_OFFSET, SCREEN_HEIGHT - (draw_pos[1] + V_OFFSET))
                pygame.draw.circle(screen, (255, 255, 255, 255), (int(draw_pos[0]), int(draw_pos[1])), int(shape.radius * PPM))

    # print("chains drawn")

    for ind in range(len(dynamic_chains)):
        chain = dynamic_chains[ind]
        output_chain(chain, frame, ind + 15)

    # Make Box2D simulate the physics of our world for one step.
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    # See the manual (Section "Simulating the World") for further discussion
    # on these parameters and their implications.
    world.Step(TIME_STEP, 10, 10)

    world.ClearForces()

    # print("time stepped")

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)
    if frame == end_frame:
        break 

    if period_counter == PERIOD:
        PERIOD = math.ceil(random.random()*5+10)
        period_counter = 0

    for chain in dynamic_chains:
        rand = random.random()
        for ind in range(len(chain)):
            body = chain[ind]
            body.ApplyForce(rand * force_vec * force_magnitude * (math.sin(frame * 2 * math.pi / PERIOD) + 1) / 2 * abs(frame) / PERIOD * abs(ind) / len(chain), body.position, True)
                
    #         # if math.floor(frame * PERIOD / 100) % 2 == 0:
            #     print("*", abs(frame - math.floor(frame / PERIOD) * PERIOD) / PERIOD)
            #     body.ApplyForce(force_vec * force_magnitude * abs(frame - math.floor(frame / PERIOD) * PERIOD) / PERIOD * abs(ind - len(chain)) / len(chain), body.position, True)
            # else:
            #     if abs(frame - math.ceil(frame / PERIOD) * PERIOD) / PERIOD == 0:
            #         print(abs(frame) / PERIOD)
            #         body.ApplyForce(force_vec * force_magnitude * abs(frame) / PERIOD * abs(ind - len(chain)) / len(chain), body.position, True)
            #     else:
            #         print(abs(frame - math.ceil(frame / PERIOD) * PERIOD) / PERIOD)
            #         body.ApplyForce(force_vec * force_magnitude * abs(frame - math.ceil(frame / PERIOD) * PERIOD) / PERIOD * abs(ind - len(chain)) / len(chain), body.position, True)

    period_counter = period_counter + 1

    force_magnitude = max(0, force_magnitude - force_decrement)

    # calculate and set linear velocities of balloons
    # update_body(balloon1_body, force_magnitude, frame, 0, True)
    # update_body(balloon1_knot_body, force_magnitude, frame, 1, True)
    # update_body(balloon2_body, force_magnitude, frame, 3, True)
    # update_body(balloon2_knot_body, force_magnitude, frame, 4, True)
    
    for ind in range(len(chains)):
        update_chain(chains[ind], force_magnitude, frame, ind)
        # update_chain(dynamic_chains[1], force_magnitude, frame, 4)

    # print("chains updated")

    frame = frame + 1

pygame.quit()
# # print('Done!')