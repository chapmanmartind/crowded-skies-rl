import pygame
from pygame.locals import *
import sys
import random
#import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)

class Game:
    def __init__(self):
        # Initializing the game and its parameters
        pygame.init()
        # Loading a font
        self.font = pygame.font.SysFont('Arial', 60, bold=True)
        
        # move_time is a timer we will use later
        self.move_time = 0
        self.width = 1000
        self.height = 700
        # The height offset is the vertical number of pixels reserved for the header at the top
        self.header_height = 100
        self._display_surface = pygame.display.set_mode([self.width, self.height])
        # Spawning the player
        self.spawn_player()
        # Instantiate the enemy group
        self.instantiate_enemy_group()
        #pygame.display.set_caption("Game")

    def exit(self):
        # Exiting the game
        pygame.quit()
        sys.exit()

    def manage_enemies(self, time):
        # Manages the enemies in the group

        if (random.randint(0, 299) == 0):
            self.spawn_enemy()

        # Listing all the enemies in the group
        enemy_arr = self.enemy_group.sprites()
        if (time - self.move_time) > 3:
            # Moving only every 3 ticks to slow game down
            for enemy in enemy_arr:
                enemy.update()
        else:
            # Always must draw
            for enemy in enemy_arr:
                enemy.draw()
        
        for enemy in enemy_arr:
            if enemy.out_of_bounds:
                # Deleting enemy if out of bounds
                del enemy
                
    def manage_player(self, time):
        # Manages the player
        player = self.player
        if (time - self.move_time) > 3:
            # Moving only every 3 ticks to slow game down
            player.update()
        else:
            # Always must draw
            player.draw()

        if player.out_of_bounds:
            self.exit()    
    
    def manage_characters(self, time):
        # Manages the player and the enemies

        self.manage_player(time)
        self.manage_enemies(time)
        # Propogating the player and enemy based on the time
        if (time - self.move_time) > 3:
            self.move_time = time

        # Check for a collision between the player and any element of the enemy group
        collision = pygame.sprite.spritecollideany(self.player, self.enemy_group)
        if collision:
            self.exit()

    def set_background(self):
        # Manages the background and the header
        # Filling the white background
        self._display_surface.fill(WHITE)
        # Creating the header font object text, antialias, color
        header = pygame.Surface([self.width, self.header_height])
        header.fill(BLUE)
        # Header rectangle centered at center of header section
        header_text = self.font.render("Skies to Fordow", True, RED)
        self._display_surface.blit(header, (0, 0))
        self._display_surface.blit(header_text, (275, self.header_height / 2 - 30))

    def update(self, time):
        # Updating the game state every tick
        # Have to fill the screen every tick
        self.set_background()
        # Check for events
        self._events = pygame.event.get()
        for event in self._events:
            if event.type == QUIT:
                self.exit()
        # Manage the characters' behavior
        self.manage_characters(time)

        #Update the game display. Without this, nothing will appear
        pygame.display.update()


    def spawn_player(self):
        # Spawns the player
        player = Player(self._display_surface, self.width, self.height, self.header_height)
        self.player = player

    def propogate_player(self):
        self.player.update()

    def instantiate_enemy_group(self):
        # Instantiates the enemy group
        # This is will hold all the enemy characters
        enemy_group = EnemyGroup()
        self.enemy_group = enemy_group

    def spawn_enemy(self):
        # Spawns an enemy
        enemy = Enemy(self._display_surface, self.width, self.height, self.header_height)
        self.enemy_group.add(enemy)


class Player(pygame.sprite.Sprite):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    def __init__(self, game_surface, game_width, game_height, header_height):
        super().__init__()
        # Instantiating the player object
        self._game_surface = game_surface
        self.game_width = game_width
        self.game_height = game_height
        self.header_height = header_height
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
        # The first header_height pixels are reserved for the header
        self.out_of_bounds = (x_left <= 0) or (x_right >= self.game_width) or (y_top <= self.header_height) or (y_bot >= self.game_height)

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
    def __init__(self, game_surface, game_width, game_height, header_height):
        super().__init__()
        # Instantiating the Enemy object visually
        self._game_surface = game_surface
        self.game_width = game_width
        self.game_height = game_height
        self.header_height = header_height
        self.width = 20
        self.height = 10
        self.image = pygame.Surface([self.width, self.height])
        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()
        # The start position of the object will be a random y cord just off the right side of the screen
        x_pos = self.game_width + self.width / 2
        # The y position shouldn't be out of bounds for any part of the object
        y_pos = random.randint(int(header_height + self.height / 2), int(self.game_height - self.height / 2))
        self.rect.center = (x_pos, y_pos)
        self.out_of_bounds = 0

    def move(self):
        # The basic enemy object moves left only
        self.rect.move_ip(-1, 0)

    def check_bounds(self):
        # Sets enemy.out_of_bounds to 1 if out of bounds, 0 otherwise
        # Note that for the basic enemy (not target seeking) the only out of bounds possibility is left
        x, y = self.rect.center
        x_left  = x - self.width  / 2
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