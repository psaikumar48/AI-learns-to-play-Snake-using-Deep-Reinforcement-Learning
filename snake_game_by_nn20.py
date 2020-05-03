import pygame
import random
import numpy as np
from scipy.spatial import distance

# Snake game functions(food,display,update_snake,snake_game)
def food():
    global Food
    snake_no_grids= [i for i in grids if i not in Snake]
    Food = random.choice(snake_no_grids)
def display():
    pygame.draw.rect(screen,(0,0,0), (0,0,M*grid_size,N*grid_size))
    for i in Snake:
        pygame.draw.rect(screen,(255,255,255), (i[0]*grid_size,i[1]*grid_size,grid_size,grid_size),1)
    pygame.draw.rect(screen,(255,255,255), (Food[0]*grid_size,Food[1]*grid_size,grid_size,grid_size))
    pygame.display.update()
def update_snake():
    global snake_tail,snake_head,snake_body
    (x,y)=Snake[0]
    if action == 'Right' :
        Snake.insert(0,(x+1,y))
    elif action == 'Left' :
        Snake.insert(0,(x-1,y))
    elif action == 'Top' :
        Snake.insert(0,(x,y-1))
    elif action == 'Bottum' :
        Snake.insert(0,(x,y+1))
    snake_tail=Snake.pop()
    snake_head=Snake[0]
    snake_body=Snake[1:len(Snake)]
    display()

# state idenetification function
def state_identification():
    (x1,y1),(fx,fy)=Snake[0],Food
    r=(x1+1,y1)
    l=(x1-1,y1)
    t=(x1,y1-1)
    b=(x1,y1+1)
    state=''
    for i in [t,r,b,l]:
        if not (i in grids) or i in Snake:
            state+='0'
        else:
            state+='1'
    if x1==fx  and y1>fy:
        state+='0'
    elif x1<fx and y1>fy:
        state+='1'
    elif x1<fx and y1==fy:
        state+='2'
    elif x1<fx and y1<fy:
        state+='3'
    elif  x1==fx and y1<fy:
        state+='4'
    elif x1>fx and y1<fy:
        state+='5'
    elif x1>fx and y1==fy:
        state+='6'
    elif x1>fx and y1>fy:
        state+='7'
    if state not in state_list:
        state_id=len(state_list)
        state_list.append(state)
    else:
        state_id=state_list.index(state)
    return state_id
def predection():
    state_id=state_identification()
    global action
    if np.random.uniform(0, 1) < Epsilon:
        action = random.choice(Actions)
    else:
        action = Actions[np.argmax(Q[state_id, :])]
    action_id=Actions.index(action)
    return state_id,action_id

M,N,grid_size=40,30,20
grids=[(i,j) for i in range(M) for j in range(N)]
Episode,High_score,Snake_wait_time,Epsilon= 0,0,0,0.99
state_list,LEARNING_RATE,GAMMA = [],0.81,0.96
Q = np.zeros((200,4))
Actions=['Top','Right','Bottum','Left']
mloop=True
while mloop:
    pygame.init()
    screen = pygame.display.set_mode((M*grid_size,N*grid_size))
    Snake=[random.choice(grids)]
    food()
    loop=True
    while loop:
        pygame.time.wait(Snake_wait_time)
        distance_to_food=distance.euclidean(Snake[0], Food)
        state_id,action_id=predection()
        update_snake()
        if snake_head==Food:
            food()
            Next_state_id=state_identification()
            reward=5
            Q[state_id, action_id] = Q[state_id, action_id] + LEARNING_RATE * (reward + GAMMA * np.max(Q[Next_state_id, :]) - Q[state_id, action_id])
            Snake.append(snake_tail)
        elif snake_head not in grids or snake_head in snake_body:
            Next_state_id=state_identification()
            reward=-5
            Q[state_id, action_id] = Q[state_id, action_id] + LEARNING_RATE * (reward + GAMMA * np.max(Q[Next_state_id, :]) - Q[state_id, action_id])
            score=len(Snake)-1
            if score > High_score:
                High_score=score
            print('Episodes :',Episode,', Epsilon :',format(Epsilon, '.3f'),', High_score :',High_score,', Score :',score)
            Episode+=1
            Epsilon-=0.001
            loop=False
        else:
            new_distance_to_food=distance.euclidean(Snake[0], Food)
            if new_distance_to_food < distance_to_food:
                reward=1
            else:
                reward=-1
            Next_state_id=state_identification()
            Q[state_id, action_id] = Q[state_id, action_id] + LEARNING_RATE * (reward + GAMMA * np.max(Q[Next_state_id, :]) - Q[state_id, action_id])
        ev=pygame.event.get()
        for event in ev:
            if event.type == pygame.QUIT:
                pygame.quit()
                loop=False
                mloop=False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    Snake_wait_time+=25
                    print('Frame refresh speed : ',Snake_wait_time)
                elif event.key == pygame.K_DOWN:
                    if Snake_wait_time>=25:
                        Snake_wait_time-=25
                        print('Frame refresh speed : ',Snake_wait_time)