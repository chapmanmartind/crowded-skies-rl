# This is the file which handles game logic
# In this version of the game the game will spawn 5 large missiles one after
# the other at the player's current position
# Dodging all 5 missiles constitutes winning the game
# Being struck by the missle or going out of bounds
# constitutes losing the game
# In this simplified version of the game you can only move up or down

import pygame
from pygame.locals import QUIT, K_UP, K_DOWN
import sys
import random
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT, HEADER_HEIGHT, TITLE, BLACK, WHITE, RED, BLUE, PLAYER_ACTION_DICT,
                       PLAYER_WIDTH, PLAYER_HEIGHT, ENEMY_WIDTH, ENEMY_HEIGHT)
import numpy as np

class Game:
    def __init__(self, human_mode, render_mode, no_enemies):
        # Initializing the game and its parameters
        # The game will be rendered if it is meant to be played by a human
        # For model training RENDER_MODE = False

        pygame.init()
        self.human_mode = human_mode
        self.render_mode = render_mode
        self.no_enemies = no_enemies

        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        # Number of pixels reserved for the header at the top
        self.header_height = HEADER_HEIGHT
        self.title = TITLE

        self.player_action_dict = PLAYER_ACTION_DICT

        # Setting a display if we are rendering the game
        if self.render_mode:
            self.font = pygame.font.SysFont('Arial', 60, bold=True)
            self._display_surface = pygame.display.set_mode([self.width, self.height])
        else:
            self._display_surface = None

        # Resetting the game state to initiate the game
        self.reset()

    def reset(self):
        # Resets the game state

        self.game_over = False
        self.victory = False
        self.player = None
        self.enemy_group = None
        self.spawned_enemy_count = 0
        self.player_enemy_collision = 0

        # In this version of the game I am including frame count as a way to keep track of time
        # I don't want to use pygame.clock because that will become difficult to deal with during training
        self.frame = 0

        self.spawn_player()
        self.instantiate_enemy_group()

    def update(self, action=None):
        # Updating the game state every tick
        # Action is only None in human mode where we will
        # read the action from the keyboard

        # Check for events
        # Incrementing the frame every game update. 
        self.frame += 1
        self._events = pygame.event.get()
        for event in self._events:
            if event.type == QUIT:
                self.exit()

        if not action:
            pressed_keys = pygame.key.get_pressed()
            # In this version only 1 key can be pressed at a time
            if pressed_keys[K_UP]:
                action = self.player_action_dict['UP']
            elif pressed_keys[K_DOWN]:
                action = self.player_action_dict['DOWN']
            else:
                # No operation
                action = self.player_action_dict['NO-OP']

        # Manage the characters' behavior
        self.update_characters(action)

        if self.render_mode:
            self.render()

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
        header_text = self.font.render(self.title, True, RED)
        self._display_surface.blit(header, (0, 0))
        # Not ideal to hardcode position like this, but ok for now
        self._display_surface.blit(header_text, (150, self.header_height / 2 - 30))

    def update_characters(self, action):
        # Manages the player and the enemies

        # Spawn an enemy if there are none one the screen
        if not self.enemy_group and not self.no_enemies:
            self.spawn_enemy(self.player.rect.center)

        # Exit if survived 5 enemies
        if self.spawned_enemy_count == 6:
            self.victory = True
            self.exit()
        if (self.frame % 40 == 0):
            self.player.update(action)
        self.enemy_group.update()

        if self.player.out_of_bounds:
            self.victory = False
            self.exit()

        # Check for a collision between the player and any member
        # of the enemy group
        collision = pygame.sprite.spritecollideany(self.player, self.enemy_group)
        if collision:
            self.victory = False
            self.exit()

    def exit(self):
        # Exiting the game
        # This should not actually exit the game. We want the outside process
        # to exit the game
        # so that way it can save game information before closing

        self.game_over = True
        if self.human_mode and self.render_mode:
            # If we are playing the game during testing we can just quit here
            pygame.quit()
            sys.exit()

    def spawn_player(self):
        # Spawns the player

        player = Player(self.human_mode, self._display_surface, self.width, self.height, self.header_height)
        self.player = player

    def spawn_enemy(self, init_pos=None):
        # Spawns an enemy

        enemy = Enemy(self._display_surface, self.width, self.height, self.header_height, init_pos)
        self.enemy_group.add(enemy)
        self.spawned_enemy_count += 1

    def instantiate_enemy_group(self):
        # Instantiates the enemy group
        # This is will hold all the enemy characters

        enemy_group = EnemyGroup()
        self.enemy_group = enemy_group

    def get_observation(self):
        # The model will call this to get the game state
        # What I put in here is based on what is currently in the game

        player_pos = self.player.rect.center

        # There is a bug where sometimes there is no enemy on the board when get_observation() is called.
        # Therefore calling self.enemy_group.sprites()[0].rect.center will throw an error.
        # To fix this, only send in the true enemy position if there is an enemy on the board.
        # Otherwise, send the expected enemy position (the current player position just off the screen)
        # We always have to send something for enemy_pos
        enemy_group = self.enemy_group.sprites()
        if enemy_group:
            enemy_pos = self.enemy_group.sprites()[0].rect.center
        else:
            # The 20 is hardcoded because I don't have access to the enemy width at the moment.
            # It's not good practice but it should do for now
            enemy_pos = ((self.width + 20) / 2, self.player.rect.center[1])

        # The following isn't exactly true because an enemy could be on the
        # screen but hasn't had the chance to collide with the player
        # Regardless, the definitionis a good enough for now
        enemies_survived = self.spawned_enemy_count

        game_over = self.game_over
        # In this version we don't actually need to check
        # out of bounds or collision because going out of bounds
        # is the same as colliding -> you lose the game
        # so we can just check if self.victory is true or not
        victory = self.victory

        return [player_pos, enemy_pos, enemies_survived, game_over, victory]

class Player(pygame.sprite.Sprite):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    def __init__(self, human_mode, game_surface, game_width, game_height, header_height):
        super().__init__()
        # Instantiating the player object

        self.human_mode = human_mode
        self._game_surface = game_surface
        self.game_width = game_width
        self.game_height = game_height
        self.header_height = header_height
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT

        # For some reason "pygame.SRCALPHA" is necessary for rotation
        self.image = pygame.Surface([self.width, self.height]).convert_alpha()

        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()
        # Changed the x position from 300 -> self.game_width / 3
        self.rect.center = (self.game_width / 3, self.game_height / 2)

        # Setting out of bounds to 0 initially
        self.out_of_bounds = 0

        self.y_velocity = 0
        self.y_acceleration = 0

    def update(self, action):
        # Updates the player's position and checks for collisions

        self.move(action)
        self.check_bounds()

    def move(self, action):
        # In this version of move we are taking a more realistic physics-based approach
        # An action will now correspond to a force which accelerates the player and modifies
        # the player's velocity, which in turn modifies the player's position
        # NOTE: Throughout this game we internally consider the downward direction as the -y direction
        # and then flip the movement direction only in the final step to align with pygame's internal convention
        # of downwards being the +y direction

        # Applying gravity
        self.y_acceleration = 0
        self.y_acceleration -= .5

        if action == 0:
            # 0 corresponds to no-op
            pass
        if action == 1:
            # 1 corresponds to up
            self.y_acceleration += 1.5
            
        if action == 2:
            # 2 corresponds to down
            self.y_acceleration -= 1.5
        self.y_velocity += self.y_acceleration
        self.rect.move_ip(0, -1 * self.y_velocity)


    def check_bounds(self):
        # Sets player.out_of_bounds to 1 if out of bounds, 0 otherwise
        # the first header_height pixels are reserved for the header

        self.out_of_bounds = ((self.rect.left <= 0)
                              or (self.rect.right >= self.game_width)
                              or (self.rect.top <= self.header_height)
                              or (self.rect.bottom >= self.game_height))

    def draw(self):
        # Draws the player onto the game surface
        # In this version we will also deal with rotation

        # In real life the angle is solely in the direction of motion (and so unconcerned with acceleration)
        # However an acceleration component makes the game feel more responsive
        angle = - (0.6 * self.y_velocity + 0.4 * self.y_acceleration) * -3

        # We don't want the angle to surpass 90 degrees because the player (a missle)
        # cannot really tilt "backwards" while ascending or descending
        angle = np.clip(angle, -90, 90)
        self.surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self.surf.fill(BLACK)
        self.rot_surf = pygame.transform.rotate(self.surf, round(angle))
        new_rect = self.rot_surf.get_rect(center=self.rect.center)

        #self.image.fill(BLACK)
        #print(angle)
        #self.image = pygame.transform.rotate(self.image, angle)
        self._game_surface.blit(self.rot_surf, new_rect)


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
    def __init__(self, game_surface, game_width, game_height, header_height, init_pos=None):
        super().__init__()
        # Instantiating the Enemy object visually
        # If init_pos is a set of coordinates, initalize there. Otherwise, randomly

        self._game_surface = game_surface
        self.game_width = game_width
        self.game_height = game_height
        self.header_height = header_height
        self.width = ENEMY_WIDTH
        self.height = ENEMY_HEIGHT
        self.image = pygame.Surface([self.width, self.height])

        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()

        # The start position of the object will ALWAYS be a random y cord just beyond the right side of the screen
        # (regardless of init_pos)
        x_pos = self.game_width + self.width / 2
        # The y position shouldn't be out of bounds for any part of the object
        if init_pos:
            y_pos = init_pos[1]
        else:
            y_pos = random.randint(int(header_height + self.height / 2), int(self.game_height - self.height / 2))

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
        # Note that for the basic enemy (not target seeking) the only out of bounds possibility is left

        self.out_of_bounds = (self.rect.left <= 0)

    def draw(self):
        # Draws the enemy onto the game surface

        self._game_surface.blit(self.image, self.rect)
