# This is the file which handles game logic
# In this version of the game the game will spawn several straight missiles, then several
# parabolic missiles, then a mix of both. Surving all the missiles constitutes winning the game.
# Being struck by the missle or going out of bounds constitutes losing the game
# In this simplified version of the game you can only move up or down

import pygame
from pygame.locals import QUIT, K_UP, K_DOWN, K_SPACE, K_x
import sys
import random
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT, HEADER_HEIGHT, TITLE, BLACK, WHITE, RED, BLUE, PLAYER_ACTION_DICT,
                       GRAVITY_CONST, FORCE_CONST, PLAYER_IMG_PATH, EXHAUST_IMG_PATH, ENEMYSTRAIGHTMISSILE_IMG_PATH,
                       BACKGROUND_IMG_PATH,
                       PLAYER_WIDTH, PLAYER_HEIGHT, ENEMY_WIDTH, ENEMY_HEIGHT)
import numpy as np
from characters import Character, Player, EnemyGroup, EnemyStraightMissile, EnemyParabolaMissile


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
        self.player_action_dict = PLAYER_ACTION_DICT

        # Setting a display if we are rendering the game
        if self.render_mode:
            self.title = TITLE
            self.font = pygame.font.SysFont('Arial', 60, bold=True)
            self.subfont = pygame.font.SysFont('Arial', 30, bold=True)
            self._display_surface = pygame.display.set_mode([self.width, self.height])
            self.top_text_image = self.font.render(self.title, True, WHITE)
            self.background = pygame.transform.scale(pygame.image.load(BACKGROUND_IMG_PATH), (self.width, self.height))
            self.header = pygame.Surface([self.width, self.header_height])
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
        self.pause = False

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
        # Incrementing the frame every game update
        self.frame += 1
        self._events = pygame.event.get()
        for event in self._events:
            if event.type == QUIT:
                self.exit()
                pygame.quit()
                sys.exit()

        if not action:
            pressed_keys = pygame.key.get_pressed()
            # In this version only 1 key can be pressed at a time
            if pressed_keys[K_UP]:
                action = self.player_action_dict['UP']
            elif pressed_keys[K_DOWN]:
                action = self.player_action_dict['DOWN']
            elif pressed_keys[K_SPACE]:
                # For resetting the game in render_mode
                game.reset()
            elif pressed_keys[K_x]:
                # To have a key to terminate the key
                pygame.quit()
                sys.exit()
            else:
                # No operation
                action = self.player_action_dict['NO-OP']

        # We want to render before checking if game.pause because we don't want the movement updates to
        # go off if the game is paused
        if self.render_mode:
            self.render()
            if game.pause:
                return

        # Manage the characters' behavior
        self.update_characters(action)

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

        self._display_surface.blit(self.background, (0, self.header_height))

        # Creating the header font object text, antialias, color
        self.header.fill(BLUE)
        self._display_surface.blit(self.header, (0, 0))

        # Header text section
        if self.game_over:
            top_text = "GAME OVER - YOU WON" if self.victory else "GAME OVER - YOU LOST"
            top_text_image = self.subfont.render(top_text, True, WHITE)
            top_text_size = top_text_image.get_size()
            # We want the header text to be centered in the header
            self._display_surface.blit(top_text_image, (self.width / 2 - top_text_size[0] / 2,
                                                        self.header_height / 2 - top_text_size[1]))
            sub_text = "Press space to reset or x to exit"
            sub_text_image = self.subfont.render(sub_text, True, WHITE)
            sub_text_size = sub_text_image.get_size()
            self._display_surface.blit(sub_text_image, (self.width / 2 - sub_text_size[0] / 2,
                                                        self.header_height / 2))

        else:
            top_text_image = self.font.render(self.title, True, WHITE)
            top_text_size = top_text_image.get_size()
            self._display_surface.blit(top_text_image, (self.width / 2 - top_text_size[0] / 2,
                                                        self.header_height / 2 - top_text_size[1] / 2))

        # Not ideal to hardcode position like this, but ok for now

    def update_characters(self, action):
        # Manages the player and the enemies

        # Spawn an enemy if there are none one the screen
        self.manage_gameplay()

        # Exit if survived 5 enemies
        if (self.frame % 40 == 0):
            self.player.update(action)
        if (self.frame % 20 == 0):
            self.enemy_group.update()

        if self.player.out_of_bounds:
            self.victory = False
            self.player.crash = True
            self.exit()

        # Check for a collision between the player and any member of the enemy group
        # Note that collision is actually the enemy object which was collided with. None if no collision.
        collided = pygame.sprite.spritecollideany(self.player, self.enemy_group)
        if collided:
            self.victory = False
            self.player.crash = True
            collided.kill()
            self.exit()

    def exit(self):
        # Exiting the game
        # This should not actually exit the game. We want the outside process
        # to exit the game
        # so that way it can save game information before closing

        self.game_over = True
        self.pause = True

    def spawn_player(self):
        # Spawns the player

        player = Player(self._display_surface, self.width, self.height, self.header_height, self.human_mode)
        self.player = player

    def spawn_enemy(self, type, init_pos=None):
        # Spawns an enemy depending on the type
        # Currently 0 = straight missile, 1 = parabola

        match type:
            # Matching isn't currently necessary but may be helpful in the future if there are many enemy types
            case 0:
                enemy = EnemyStraightMissile(self._display_surface, self.width, self.height, self.header_height,
                                             init_pos)
            case 1:
                enemy = EnemyParabolaMissile(self._display_surface, self.width, self.height, self.header_height,
                                             init_pos)

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
        # In this version of the game I want to pass in
        # - player position and velocity
        # - enemy position, velocity, and type for every enemy on the screen
        # - number of frames
        # - number of enemies spawned
        # - out of bounds
        # - collision
        # - game over
        # - victory

        # The length of enemy_group is variable. However, the network cannot accept variable length input.
        # Therefore we have to allocate a 0 buffer of the maximum possible length and overwrite it according to
        # how many enemies we have
        # I don't think we can currently have more than 3 enemies on the screen at once. Let's allocate 4 to be safe.
        # [enemy position, enemy velocity, enemy type]
        # Because of bugs in model.py we will flatten the array here
        # If this seems stupid, I know. But I think this will fix the bugs dealing with deep inhomogenous length arrays

        observation = []

        # Adding the player observations
        observation.append(self.player.rect.center[0])
        observation.append(self.player.rect.center[1])
        observation.append(self.player.velocity.x)
        observation.append(self.player.velocity.y)

        # Adding the enemy observations
        enemy_group = self.enemy_group.sprites()
        # We need to add the entire enemy buffer regardless if they exist
        enemy_buffer = [0] * 20
        for i in range(len(enemy_group)):
            if i > 3:
                print("CHARCTER BUFFER OVERFLOW -- EXITING NOW")
                pygame.quit()
                sys.exit()
            enemy = enemy_group[i]
            enemy_buffer[5 * i] = enemy.rect.center[0]
            enemy_buffer[5 * i + 1] = enemy.rect.center[1]
            enemy_buffer[5 * i + 2] = enemy.velocity.x
            enemy_buffer[5 * i + 3] = enemy.velocity.y
            enemy_buffer[5 * i + 4] = enemy.type

        observation += enemy_buffer

        # Adding the general game observations
        observation.append(self.frame)
        observation.append(self.spawned_enemy_count)
        observation.append(self.player.out_of_bounds)
        observation.append(self.player.crash)
        observation.append(self.game_over)
        observation.append(self.victory)
        # In this version (unlike the previous) we want to send in out of bounds and collision separately to give the
        # network more information. This is a fairly small input for a neural network so sending more information in
        # is not an issue at the moment
        # The total observation size is 2 + 2 + 4 * 5 + 1 + 1 + 1 + 1 + 1 + 1 = 30 floats
        if len(observation) != 30:
            print("BIG FUCKING ERROR")
            print(observation)
        return observation

    def manage_gameplay(self):
        # This function will manage the gameplay - what enemies are spawned when
        # and when the game is over
        # This version of the game will have 3 sections. An all straight missile section,
        # an all parabolic missile section, and then a mixed section. After finishing all that
        # the game is over

        if self.frame < 2000:
            # Give the player some time to settle in
            return

        if self.frame < 15000:
            # The introductory section
            if not (self.frame % 1500):
                # Firing only straight missiles. It isn't ideal to hard code the value
                # used to represent types of missiles but for now probably ok
                type = 0
                self.spawn_enemy(type, self.player.rect.center)
            return

        if self.frame < 30000:
            # The middle section
            if not (self.frame % 1500):
                # Firing only parabolic missiles
                type = 1
                self.spawn_enemy(type)
            return

        if self.frame < 50000:
            # The final section
            if not (self.frame % 1250):
                # Randomly choose the type
                type = np.random.randint(0, 2)
                self.spawn_enemy(type, self.player.rect.center)
            return

        else:
            # End the game when the screen is cleared of sprites
            if not (len(self.enemy_group.sprites())):
                self.victory = True
                self.exit()
