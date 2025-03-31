import pygame as p
import random

p.init()
font_style = p.font.SysFont("arial", 30)

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 150)

class Snake():

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
        
        self.frame_iteration = 0
        self.score = 0
        self.directions = {"left": (-self.size, 0), "up": (0, -self.size), "down": (0, self.size), "right": (self.size, 0)}
        self.direction = "right"

        # End of the list is the head of the snake
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

        # Check if position of head of snake is equal to position of food
        if (self.positions[-1][0] == self.food[0]) and (self.positions[-1][1] == self.food[1]):
            self.score += 1
            self.positions.insert(0, (self.positions[0]))
            self._create_food()

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
        p.display.flip()

    def _move(self):
        '''
        Fucntion that moves the snake.
        '''

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

    def step(self):
        '''
        Function that playes the next action for the snake, and checks the state of the snake after the action.
        '''

        # Check for key presses
        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                quit()

            if event.type == p.KEYDOWN:
                if event.key == p.K_d and self.direction != "left":
                    self.direction = "right"
                if event.key == p.K_a and self.direction != "right":
                    self.direction = "left"
                if event.key == p.K_w and self.direction != "down":
                    self.direction = "up"
                if event.key == p.K_s and self.direction != "up":
                    self.direction = "down"

        # Move the snake
        self._move()

        # Check if game Over
        game_over = False 
        if(self._collide()):
            game_over = True
            return game_over, self.score

        # Check if snake eats food
        self._eat_food()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # Return game Over and Display Score
        return game_over,self.score

    
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

        return state


if __name__=="__main__":
    game = Snake(width=600, height=400, size=10, speed=15)

    # Game loop
    while True:
        game_over, score = game.step()
        if(game_over == True):
            break
    print('Final Score',score)

    p.quit()