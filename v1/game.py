# This is the file which handles game logic

import pygame
from pygame.locals import QUIT, K_UP, K_RIGHT, K_DOWN, K_LEFT
import sys
import random
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT, HEADER_HEIGHT,
                       WHITE, RED, BLUE)


class Game:
    def __init__(self, human_mode, render_mode):
        # Initializing the game and its parameters
        # The game will be rendered if it is meant to be played by a human
        # For model training RENDER_MODE = False

        pygame.init()
        self.human_mode = human_mode
        self.render_mode = render_mode
        
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        # Number of pixels reserved for the header at the top
        self.header_height = HEADER_HEIGHT

        # Setting a display if we are rendering the game
        if self.render_mode:
            self.font = pygame.font.SysFont('Arial', 60, bold=True)
            self._display_surface = pygame.display.set_mode(
                [self.width, self.height])
        else:
            self._display_surface = None

        # Resetting the game state to initiate the game
        self.reset()

    def reset(self):
        # Resets the game state
        self.player = None
        self.enemy_group = None

        self.spawn_player()
        self.instantiate_enemy_group()

        # For the current version, spawn a single enemy
        # and only once
        # and at the player's position
        self.spawn_enemy(init_pos=self.player.rect.center)

    def update(self):
        # Updating the game state every tick
        if self.render_mode:
            self.render()

        # Check for events
        self._events = pygame.event.get()
        for event in self._events:
            if event.type == QUIT:
                self.exit()

        # Manage the characters' behavior
        self.update_characters()

    def render(self):
        # Renders the game and all characters
        self.set_background()

        self.player.draw()
        self.enemy_group.draw()
        # Update the game display. Without this, nothing will appear
        pygame.display.update()

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
        self._display_surface.blit(header_text,
                                   (150, self.header_height / 2 - 30))

    def update_characters(self):
        # Manages the player and the enemies

        self.player.update()
        self.enemy_group.update()

        if self.player.out_of_bounds:
            self.exit()

        # Check for a collision between the player and any member
        # of the enemy group
        collision = pygame.sprite.spritecollideany(
            self.player, self.enemy_group)
        if collision:
            self.exit()

    def exit(self):
        # Exiting the game

        pygame.quit()
        sys.exit()

    def spawn_player(self):
        # Spawns the player

        player = Player(self.human_mode, self._display_surface, self.width,
                        self.height, self.header_height)
        self.player = player

    def spawn_enemy(self, init_pos=None):
        # Spawns an enemy

        enemy = Enemy(self._display_surface, self.width, self.height,
                      self.header_height, init_pos)
        self.enemy_group.add(enemy)

    def instantiate_enemy_group(self):
        # Instantiates the enemy group
        # This is will hold all the enemy characters

        enemy_group = EnemyGroup()
        self.enemy_group = enemy_group

    def get_observation(self):
        # The model will call this to get the game state
        # What I put in here is based on what is currently in the game
        # At this point, only returning the player and enemy_group objects

        return (self.player, self.enemy_group)


class Player(pygame.sprite.Sprite):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    def __init__(self, human_mode, game_surface, game_width, game_height,
                 header_height):
        super().__init__()
        # Instantiating the player object
        # NOTE!! I AM NOT SURE IF HUMAN_MODE IS THE BEST WAY TO DEAL WITH 
        # HUMAN/MODEL MOVEMENT -- TBD
        # TODO: FIGURE THIS OUT

        self.human_mode = human_mode
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

    def update(self):
        # Updates the player's position and checks for collisions

        self.move()
        self.check_bounds()

    def move(self):
        # Moves the player based on the pressed directional keys

        if self.human_mode:
            pressed_keys = pygame.key.get_pressed()
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
        else:
            # TODO: FIGURE OUT WHAT TO DO NOT HUMAN MODE
            pass

    def check_bounds(self):
        # Sets player.out_of_bounds to 1 if out of bounds, 0 otherwise
        # The first header_height pixels are reserved for the header

        self.out_of_bounds = ((self.rect.left <= 0)
                              or (self.rect.right >= self.game_width)
                              or (self.rect.top <= self.header_height)
                              or (self.rect.bottom >= self.game_height))

    def draw(self):
        # Draws the player onto the game surface

        self._game_surface.blit(self.image, self.rect)


class EnemyGroup(pygame.sprite.Group):
    # Instantiating the enemy group. For now this will just be a sprite group
    def __init__(self):
        super().__init__()

    def update(self):
        for enemy in self.sprites():
            enemy.update()
            if enemy.out_of_bounds:
                enemy.kill()

    def draw(self):
        for enemy in self.sprites():
            enemy.draw()


class Enemy(pygame.sprite.Sprite):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    def __init__(self, game_surface, game_width, game_height, header_height,
                 init_pos=None):
        super().__init__()
        # Instantiating the Enemy object visually
        # If init_pos is a set of coordinates, initalize there
        # Otherwise, randomly

        self._game_surface = game_surface
        self.game_width = game_width
        self.game_height = game_height
        self.header_height = header_height
        self.width = 20
        self.height = 10
        self.image = pygame.Surface([self.width, self.height])

        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()

        # The start position of the object will ALWAYS be a random y cord just
        # beyond the right side of the screen
        # (regardless of init_pos)
        x_pos = self.game_width + self.width / 2
        # The y position shouldn't be out of bounds for any part of the object
        if init_pos:
            y_pos = init_pos[1]
        else:
            y_pos = random.randint(int(header_height + self.height / 2),
                                   int(self.game_height - self.height / 2))

        self.rect.center = (x_pos, y_pos)
        self.out_of_bounds = 0

    def update(self):
        # Updates the enemy's position and checks for collisions

        self.move()
        self.check_bounds()

    def move(self):
        # The basic enemy object moves left only

        self.rect.move_ip(-1, 0)

    def check_bounds(self):
        # Sets enemy.out_of_bounds to 1 if out of bounds, 0 otherwise
        # Note that for the basic enemy (not target seeking) the only out of
        # bounds possibility is left

        self.out_of_bounds = (self.rect.left <= 0)

    def draw(self):
        # Draws the enemy onto the game surface

        self._game_surface.blit(self.image, self.rect)


# Human mode describes if a human will be playing the game
HUMAN_MODE = True
# Render mode describes if the game will be rendered
RENDER_MODE = True
game = Game(HUMAN_MODE, RENDER_MODE)

while True:
    game.update()
