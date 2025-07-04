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
                       GRAVITY_CONST, FORCE_CONST, PLAYER_IMG_PATH, EXHAUST_IMG_PATH, ENEMYSTRAIGHTMISSILE_IMG_PATH,
                       PLAYER_WIDTH, PLAYER_HEIGHT, ENEMY_WIDTH, ENEMY_HEIGHT)
import numpy as np
from characters import Character, Player, EnemyGroup, EnemyStraightMissile


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

        # We want to render before checking if game.pause because we don't want the movement updates to
        # go off if the game is paused
        if self.render_mode:
            self.render()
            if game.pause:
                return

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
        if self.game_over:
            self.title = "GAME OVER - YOU WON" if self.victory else "GAME OVER - YOU LOST"

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
        if (self.frame % 10 == 0):
            self.enemy_group.update()

        if self.player.out_of_bounds:
            self.victory = False
            self.exit()

        # Check for a collision between the player and any member of the enemy group
        # Note that collision is actually the enemy object which was collided with
        collided = pygame.sprite.spritecollideany(self.player, self.enemy_group)
        if collided:
            self.game_over = True
            self.victory = False
            self.player.crash = True
            collided.kill()
            self.pause = True
            #self.exit()

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

        player = Player(self._display_surface, self.width, self.height, self.header_height, self.human_mode)
        self.player = player

    def spawn_enemy(self, init_pos=None):
        # Spawns an enemy

        enemy = EnemyStraightMissile(self._display_surface, self.width, self.height, self.header_height, init_pos)
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

human_mode = True
render_mode = True
no_enemies = False
game = Game(human_mode, render_mode, no_enemies)
while True:
    game.update()
