import pygame as p
import numpy as np
import random

p.init()
font_style = p.font.SysFont("arial", 30)

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 150)

# A class that implements snake playable by an agent
class Environment():

    def __init__(self, width, height, size, speed):
        self.width = width
        self.height = height

        # init pygame display
        self.display = p.display.set_mode((self.width, self.height))
        p.display.set_caption('Snake AI')
        self.clock = p.time.Clock()

        # snake attributes
        self.size = size
        self.speed = speed
        self.reset()

    def reset(self):
        '''
        Function that resets the snake. Used when the snake dies.
        '''

        self.iteration = 0
        self.score = 0
        self.directions = {"right": (self.size, 0), "down": (0, self.size), "left": (-self.size, 0), "up": (0, -self.size)}
        self.direction = "right"

        # End of the list is the head of the snake, start length of 5
        self.positions = [(self.width/2, self.height/2), ( (self.width/2) + self.size , self.height/2), ( (self.width/2) + (2 * self.size) , self.height/2),
                          ( (self.width/2) + (3 * self.size) , self.height/2), ( (self.width/2) + (4 * self.size) , self.height/2)]
        self.food = None
        self._create_food()

    def _create_food(self):
        '''
        Function that places the food at a new location not occupied by the snake.
        '''

        # New location
        x = random.randrange(0, self.width, self.size)
        y = random.randrange(0, self.height, self.size)
        self.food = (x, y)

        # Check if new location is equal to any of snake positions
        for pos in self.positions:
            if (pos[0] == self.food[0]) and (pos[1] == self.food[1]):
                self._create_food

    def _eat_food(self):
        '''
        Fuction that checks if the snake hits the food.
        '''

        reward = 0
        # Check if position of head of snake is equal to position of food
        if (self.positions[-1][0] == self.food[0]) and (self.positions[-1][1] == self.food[1]):
            self.score += 1
            reward = 10
            self.positions.insert(0, (self.positions[0]))
            self._create_food()
        return reward

    def _update_ui(self):
        '''
        Function that updates the pygame UI and draws the different game objects.
        '''

        self.display.fill(BLACK)

        # Draw game objects
        for pos in self.positions:
            p.draw.rect(self.display, GREEN, p.Rect(pos[0], pos[1], self.size, self.size))
        p.draw.rect(self.display, RED, p.Rect(self.food[0], self.food[1], self.size, self.size))

        # Draw text
        text = font_style.render("Score: "+ str(self.score), True, WHITE)
        self.display.blit(text, [20, 20])

        # Update UI
        p.display.update()

    def _move(self, action):
        '''
        Fucntion that moves the snake.
        '''

        # Action
        # [1, 0, 0] -> Straight
        # [0, 1, 0] -> Right
        # [0, 0, 1] -> Left

        temp = list(self.directions)
        # if(action[0] == 1):    # Go straight, dont change direction
        #     pass
        if(action[1] == 1):  # Turn right
            self.direction = temp[(temp.index(self.direction) + 1) % 4]
        elif(action[2] == 1):   # Turn left
            self.direction = temp[(temp.index(self.direction) - 1) % 4]

        length = len(self.positions) - 1
        for count, _ in enumerate(self.positions):
            # The last item in the list is the head of the snake
            if count == (length):
                # Adds two tuples together
                self.positions[count] = tuple(sum(x) for x in zip(self.positions[count], self.directions[self.direction]))
            else:
                # The current item is equal to the next item
                self.positions[count] = self.positions[count + 1]

    def _collide(self, pos=None):
        '''
        Function that checks if the snake collides with the boundaries or itslef.
        Optional: Take a position as parameter and check if it will collide
        '''
        if(pos == None):
            pos = self.positions[-1]

        # Check if snake hits right or left wall
        if(pos[0] > self.width - self.size) or (pos[0] < 0):
            return True
        # Check if snake hits bottom or upper wall
        if(pos[1] > self.height - self.size) or (pos[1] < 0):
            return True
        # Check if snake hits itself
        if(pos in self.positions[:-1]):
            return True
        return False

    def step(self, action):
        '''
        Function that playes the next action of the snake, and checks the state of the snake after the action.
        
        Return: game over (bool), self.score (int), reward (int)
        '''

        self.iteration += 1
        # Check for key presses
        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                quit()

        # Previous position of snake head
        prev_pos = self.positions[-1]

        # Move the snake
        self._move(action)

        # Reward the snake for getting closer to the food, punish otherwise
        reward = self._closer_to_food(prev_pos)

        # Check if game Over
        game_over = False
        if( self._collide() or self.iteration > 100*len(self.positions) ):
            game_over = True
            reward += -10
            return game_over, self.score, reward

        # Check if snake eats food
        reward += self._eat_food()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # Return game Over and Display Score
        return game_over, self.score, reward

    def get_state(self):
        '''
        Functions that return the current state of the snake in a list, there are 11 states.
        
        [ Danger going -Straight, -Right,  -Left,

          Snake is going -Right, -Down, -Left, -Up,
          
          Food is -Right, -Down, -Left, -Up ]
        '''
        
        # Check where the snake is going, True or False
        dir_r = self.direction == list(self.directions.keys())[0]
        dir_d = self.direction == list(self.directions.keys())[1]
        dir_l = self.direction == list(self.directions.keys())[2]
        dir_u = self.direction == list(self.directions.keys())[3]

        # Possible positions the next action could move the snake
        pos_r = tuple(sum(x) for x in zip(self.positions[-1], self.directions["right"]))
        pos_d = tuple(sum(x) for x in zip(self.positions[-1], self.directions["down"]))
        pos_l = tuple(sum(x) for x in zip(self.positions[-1], self.directions["left"]))
        pos_u = tuple(sum(x) for x in zip(self.positions[-1], self.directions["up"]))

        state = [
            # Danger to go straight
            (dir_r and self._collide(pos_r))or
            (dir_d and self._collide(pos_d))or
            (dir_l and self._collide(pos_l))or
            (dir_u and self._collide(pos_u)),

            # Danger to go right
            (dir_r and self._collide(pos_d))or
            (dir_d and self._collide(pos_l))or
            (dir_l and self._collide(pos_u))or
            (dir_u and self._collide(pos_r)),

            # Danger to go left
            (dir_r and self._collide(pos_u))or
            (dir_d and self._collide(pos_r))or
            (dir_l and self._collide(pos_d))or
            (dir_u and self._collide(pos_l)),


            dir_r, # Snake is going right
            dir_d, # Snake is going down
            dir_l, # Snake is going left
            dir_u, # Snake is going up


            self.food[0] > self.positions[-1][0], # Food is right
            self.food[1] > self.positions[-1][1], # Food is down
            self.food[0] < self.positions[-1][0], # Food is left
            self.food[1] < self.positions[-1][1] # Food is up
        ]

        # return state
        return np.array(state, dtype=int)

    def _closer_to_food(self, prev_pos):
        '''
        Function that rewards the snake for getting closer to the food and punishes it otherwise.
        '''

        # Subtract position and food tuple
        prev_dis = np.subtract(prev_pos, self.food)
        new_dis = np.subtract(self.positions[-1], self.food)

        # The absolute sum of each tuple
        prev_sum = abs(prev_dis[0]) + abs(prev_dis[1])
        new_sum = abs(new_dis[0]) + abs(new_dis[1])

        # The lowest value is the closest to the food
        if prev_sum > new_sum:
            return 1
        else:
            return -1
        

if __name__=="__main__":
    game = Environment(width=600, height=400, size=10, speed=1)

    #Game loop
    while True:
        # Random example
        actions = [ [1,0,0], [0,1,0], [0,0,1] ]
        action = random.choice(actions)
        game_over, score, reward = game.step(action)
        if(game_over == True):
            break
    print('Final Score',score)

    p.quit()
