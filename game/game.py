import pygame
from pygame.locals import *
import sys
import random
#import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Game:
    def __init__(self):
        # Initializing the game and its parameters
        pygame.init()
        self.move_time = 0
        self.width = 1000
        self.height = 600
        self._display_surface = pygame.display.set_mode([self.width, self.height])
        # Spawning the player
        self.spawn_player()
        # Instantiate the enemy group
        self.spawn_enemy_group()
        #pygame.display.set_caption("Game")

    def exit(self):
        # Exiting the game
        pygame.quit()
        sys.exit()

    def manage_characters(self, time):
        #Manages the player and the enemies

        # Propogating the player and enemy based on the time
        if (time - self.move_time) > 3:
            # Only allow movement every 3 ticks to slow game down
            self.player.update()
            if self.enemy:
                self.enemy.update()
            self.move_time = time
        else:
            # Always must draw player and enemy
            self.player.draw()
            if self.enemy:
                self.enemy.draw()
        
        # Checking out of bounds
        if self.player.out_of_bounds:
            self.exit()
        
        # Spawn an enemy if there are none
        if self.enemy == None:
            self.spawn_enemy()

        # Remove enemy if out of bounds
        if self.enemy.out_of_bounds:
            self.enemy = None

        # Check for a collision between the player and any element of the enemy group
        collision = pygame.sprite.spritecollideany(self.player, self.enemy_group)
        if collision:
            self.exit()

    def update(self, time):
        # Updating the game state every tick
        # Have to fill the screen every tick
        self._display_surface.fill(WHITE)
        # Check for events
        self._events = pygame.event.get()
        for event in self._events:
            if event.type == QUIT:
                self.exit()
        # Manage the characters' behavior
        self.manage_characters(time)

    def spawn_player(self):
        # Spawns the player
        player = Player(self._display_surface, self.width, self.height)
        self.player = player

    def propogate_player(self):
        self.player.update()

    def spawn_enemy_group(self):
        enemy_group = EnemyGroup()
        self.enemy_group = enemy_group
        self.enemy = None

    def spawn_enemy(self):
        # Spawns an enemy
        enemy = Enemy(self._display_surface)
        self.enemy = enemy
        self.enemy_group.add(enemy)

    def propogate_enemy(self):
        # Propogates an enemy
        self.enemy.update()

class Player(pygame.sprite.Sprite):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    def __init__(self, game_surface, game_width, game_height):
        super().__init__()
        # Instantiating the player object visually
        self._game_surface = game_surface
        self.game_width = game_width
        self.game_height = game_height
        self.width = 40
        self.height = 20
        self.image = pygame.Surface([self.width, self.height])
        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()
        self.rect.center = (500, 300)
        # Setting out of bounds to 0 initially
        self.out_of_bounds = 0

    def move(self):
        # Moves the player based on the pressed directional keys
        pressed_keys = self._pressed_keys
        if pressed_keys[K_UP]:
            # "UP" is the -y direction
            self.rect.move_ip(0, -1)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(1, 0)
        if pressed_keys[K_DOWN]:
            # "DOWN" is the +y direction
            self.rect.move_ip(0, 1)
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-1, 0)

    def check_bounds(self):
        # Sets player.out_of_bounds to 1 if out of bounds, 0 otherwise
        x, y = self.rect.center
        x_left  = x - self.width  / 2
        x_right = x + self.width  / 2
        # Note that towards the bottom of the screen is in the POSITIVE y direction, so
        y_top   = y - self.height / 2 # This is the lower numerical value
        y_bot   = y + self.height / 2 # This is the greater numerical value
        self.out_of_bounds = (x_left <= 0) or (x_right >= self.game_width) or (y_top <= 0) or (y_bot >= self.game_height)

    def update(self):
        # Updates the player's position and checks for collisions
        self._pressed_keys = pygame.key.get_pressed()
        self.move()
        self.check_bounds()
        self.draw()
    
    def draw(self):
        # Draws the player onto the game surface
        self._game_surface.blit(self.image, self.rect)

class EnemyGroup(pygame.sprite.Group):
    # Instantiating the enemy group. For now this will just be a sprite group
    def __init__(self):
        super().__init__()

class Enemy(pygame.sprite.Sprite):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    def __init__(self, game_surface):
        super().__init__()
        # Instantiating the Enemy object visually
        self._game_surface = game_surface
        self.width = 20
        self.height = 10
        self.image = pygame.Surface([self.width, self.height])
        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()
        # The start position of the object will be a random y cord just off the right side of the screen
        self.rect.center = (1000 + self.width / 2, random.randint(10, 590)) #Hard coding bounds for now -- have to pull from game width and height
        self.out_of_bounds = 0

    def move(self):
        # The basic enemy object moves left only
        self.rect.move_ip(-1, 0)

    def check_bounds(self):
        # Sets enemy.out_of_bounds to 1 if out of bounds, 0 otherwise
        # Note that for the basic enemy (not target seeking) the only out of bounds possibility is left
        x, y = self.rect.center
        x_left  = x - self.width  / 2
        # Note that down is the POSITIVE y direction, so
        self.out_of_bounds = (x_left <= 0)

    def draw(self):
        # Draws the enemy onto the game surface
        self._game_surface.blit(self.image, self.rect)

    def update(self):
        # Updates the enemy's position and checks for collisions
        self.move()
        self.check_bounds()
        if self.out_of_bounds:
            self.kill()
        self.draw()



game = Game()
clock = pygame.time.Clock()
time = 0

while True:
    clock.tick()
    time = pygame.time.get_ticks()
    game.update(time)
    pygame.display.update()
