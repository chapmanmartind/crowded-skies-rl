import pygame
from pygame.locals import QUIT, K_UP, K_DOWN
import sys
import random
from constants import (SCREEN_WIDTH, SCREEN_HEIGHT, HEADER_HEIGHT, TITLE, BLACK, WHITE, RED, BLUE, PLAYER_ACTION_DICT,
                       GRAVITY_CONST, FORCE_CONST, PLAYER_IMG_PATH, EXHAUST_IMG_PATH, ENEMYSTRAIGHTMISSILE_IMG_PATH,
                       EXPLOSION_IMG_PATH,
                       ENEMYSTRAIGHTMISSILE_X_VELOCITY, 
                       PLAYER_WIDTH, PLAYER_HEIGHT, ENEMY_WIDTH, ENEMY_HEIGHT)
import numpy as np


class Character(pygame.sprite.Sprite):
    # The base class for the player and all enemies
    # I don't know how helpful this is considering almost all the methods have to be
    # implemented by the subclasses themselves. But it seems like good practice.
    def __init__(self, game_surface, game_width, game_height, header_height):
        super().__init__()
        self._game_surface = game_surface
        self.game_width = game_width
        self.game_height = game_height
        self.header_height = header_height
        self.out_of_bounds = 0
        self.crash = False
        self.image = None
        self.rect = None

        # All characters will have a width and height but I don't know what they are here
        self.width = 0
        self.height = 0

        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = pygame.Vector2(0, 0)

    def update(self, action=None):
        # Updates the character's position according to its privately implemented movement method
        # and checks bounds. Action can be none because enemies do not recieve input from the game

        self.move(action)
        self.check_bounds()

    def move(self, action=None):
        raise NotImplementedError("Subclass must implement move")

    def check_bounds(self):
        # Sets player.out_of_bounds to 1 if out of bounds, 0 otherwise
        # the first header_height pixels are reserved for the header
        # Removing the right check for the moment because currently impossible
        # And causing issues with the enemies

        self.out_of_bounds = ((self.rect.left <= 0)
                              #or (self.rect.right >= self.game_width)
                              or (self.rect.top <= self.header_height)
                              or (self.rect.bottom >= self.game_height))

    def apply_gravity(self):
        # Applies gravity
        # For logical consistentency and for the sake of analoigy to physics
        # all characters will be affected by grabity
        # Eben if their movement doesn't reflect it (they fly straight)

        self.acceleration.y -= GRAVITY_CONST

    def draw(self):
        # This is the default implementation of draw. Angle is based off of the velocity
        # The player character will overwrite this

        # Convert the velocity vector to polar coordinates (magnitude, angle) and keep the angle
        # TODO THIS ANGLE CALCULATING ISN'T WORKING. I THINK IT IS BECAUSE OF THE INVERSION OF THE Y AXIS. FIX THIS.
        # angle = self.velocity.as_polar()[1]
        angle = np.rad2deg(np.arctan(self.velocity.y / self.velocity.x))
        rotated_image = pygame.transform.rotate(self.image, round(angle))
        new_rect = rotated_image.get_rect(center=self.rect.center)
        self._game_surface.blit(rotated_image, new_rect)


class Player(Character):
    # Using the character class for inheritance
    def __init__(self, game_surface, game_width, game_height, header_height, human_mode):
        super().__init__(game_surface, game_width, game_height, header_height)
        # Instantiating the player object

        self.human_mode = human_mode
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        # For some reason "pygame.SRCALPHA" is necessary for rotation
        self.image = pygame.transform.scale(pygame.image.load(PLAYER_IMG_PATH), (self.width, self.height))
        self.exhaust_image = pygame.transform.scale(pygame.image.load(EXHAUST_IMG_PATH),
                                                    (self.width * 1.48, self.height))

        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()
        # Changed the x position from 300 -> self.game_width / 3
        self.rect.center = (self.game_width / 3, self.game_height / 2)

        # This will be used to trigger the exhaust image
        self.acceleration_flag = 0

        # I am not sure if this is useful but it could be. Every character will have a type.
        # The type for the player is 0
        self.type = 0

    def move(self, action):
        # In this version of move we are taking a more realistic physics-based approach
        # An action will now correspond to a force which accelerates the player and modifies
        # the player's velocity, which in turn modifies the player's position
        # NOTE: Throughout this game we internally consider the downward direction as the -y direction
        # and then flip the movement direction only in the final step to align with pygame's internal convention
        # of downwards being the +y direction

        # We need to reset the acceleration every move because the acceleration is a function of force
        # which doesn't propgate across moves. A force is applied only for an instant - a single move
        self.acceleration = pygame.Vector2()
        self.acceleration_flag = 0
        # This is the acceleration constant that seems to work with the frame rate and force constant
        self.apply_gravity()

        if action == 0:
            # 0 corresponds to no-op
            pass
        if action == 1:
            # 1 corresponds to up
            self.acceleration.y += FORCE_CONST
            self.acceleration_flag = 1

        if action == 2:
            # 2 corresponds to down
            self.acceleration.y -= FORCE_CONST
            self.acceleration_flag = 1

        self.velocity += self.acceleration
        # We need to deal with the x and y velocities separately now because of the direction of the y axis
        self.rect.move_ip(self.velocity.x, -1 * self.velocity.y)

    def draw(self):
        # Draws the player onto the game surface
        # In this version we will also deal with rotation

        if self.crash:
            crash_image = pygame.transform.scale(pygame.image.load(EXPLOSION_IMG_PATH), (self.width, self.height * 3))
            self._game_surface.blit(crash_image, self.rect)
            return
        # In real life the angle is solely in the direction of motion (and so unconcerned with acceleration)
        # However an acceleration component makes the game feel more responsive
        angle = (0.6 * self.velocity.y + 0.4 * self.acceleration.y) * 3
        # We don't want the angle to surpass 90 degrees because the player (a missle)
        # cannot tilt "backwards" while ascending or descending
        angle = np.clip(angle, -90, 90)

        if not self.acceleration_flag:
            self.rotated_image = pygame.transform.rotate(self.image, round(angle))
            new_rect = self.rotated_image.get_rect(center=self.rect.center)
            self._game_surface.blit(self.rotated_image, new_rect)
        else:
            self.rotated_image = pygame.transform.rotate(self.exhaust_image, round(angle))
            new_rect = self.rotated_image.get_rect(center=(self.rect.center[0] - self.width / 4, self.rect.center[1]))
            self._game_surface.blit(self.rotated_image, new_rect)


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


class EnemyStraightMissile(Character):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    # This is the basic enemy missile which goes straight
    def __init__(self, game_surface, game_width, game_height, header_height, init_pos=None):
        super().__init__(game_surface, game_width, game_height, header_height)
        # Instantiating the Enemy object visually
        # If init_pos is a set of coordinates, initalize there. Otherwise, randomly

        self.width = ENEMY_WIDTH
        self.height = ENEMY_HEIGHT
        self.image = pygame.transform.scale(pygame.image.load(ENEMYSTRAIGHTMISSILE_IMG_PATH), (self.width, self.height))

        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()

        # The start position of the object will ALWAYS be a random y cord just beyond the right side of the screen
        # (regardless of init_pos)
        x_pos = self.game_width + self.width / 2
        # The y position shouldn't be out of bounds for any part of the object
        if init_pos:
            y_pos = init_pos[1]
        else:
            y_pos = random.randint(int(self.header_height + self.height / 2), int(self.game_height - self.height / 2))

        self.rect.center = (x_pos, y_pos)
        self.velocity.x = np.random.uniform(.5, 1) * ENEMYSTRAIGHTMISSILE_X_VELOCITY
        self.out_of_bounds = 0

        # The enemy straight missile has type 1
        self.type = 1

    def move(self, action=None):
        # In this version of the game we will take a more physics based approach to character movement
        # All characters are affected by gravity
        # This character will have a constant upward acceleration to offset gravity
        # And a constant leftward velocity set in the beginning

        # Action will always be None for this. It needs to have action=None to correctly inheret from Character

        # I understand it is redundant to apply gravity then undo it through an upward acceleration
        # However, this more closely represents physical reality than simply having no acceleration in the
        # y direction at all
        self.apply_gravity()
        self.acceleration.y += GRAVITY_CONST
        self.velocity += self.acceleration
        # Note we have to invert the y component of the velocity to reflect pygame's y coordinate convention
        self.rect.move_ip(self.velocity.x, -1 * self.velocity.y)


class EnemyParabolaMissile(Character):
    # Using the built-in pygame.sprite.Sprite class for inheritance
    # This missile follows a parabolic trajectory
    def __init__(self, game_surface, game_width, game_height, header_height, init_pos=None):
        super().__init__(game_surface, game_width, game_height, header_height)
        # Instantiating the Enemy object visually
        # If init_pos is a set of coordinates, initalize there. Otherwise, randomly

        self.width = ENEMY_WIDTH
        self.height = ENEMY_HEIGHT
        self.image = pygame.transform.scale(pygame.image.load(ENEMYSTRAIGHTMISSILE_IMG_PATH), (self.width, self.height))

        # The rect is the actual physical representation on the screen
        self.rect = self.image.get_rect()

        # This missile will come from the same position every time
        x_pos = self.game_width + self.width / 2
        y_pos = self.game_height - self.height / 2
        self.rect.center = (x_pos, y_pos)

        # However it will have randomly varying velocity and so trajectory
        # I arrived upon these ranges through trial and error. I am hard coding them for now because
        # I will not change them and changing them affects the movement and gameplay in unpredictable and
        # usually negative ways
        self.velocity.x = np.random.uniform(.6, .85) * ENEMYSTRAIGHTMISSILE_X_VELOCITY
        self.velocity.y = np.random.uniform(3, 10)
        self.out_of_bounds = 0

        # Enemy parabolic missile has type 2
        self.type = 2

    def move(self, action=None):
        # In this version of the game we will take a more physics based approach to character movement
        # All characters are affected by gravity
        # This character will have a constant upward acceleration to offset gravity
        # And a constant leftward velocity set in the beginning

        # Action will always be None for this. It needs to have action=None to correctly inheret from Character

        # I understand it is redundant to apply gravity then undo it through an upward acceleration
        # However, this more closely represents physical reality than simply having no acceleration in the
        # y direction at all
        self.apply_gravity()

        # Dividing the effect of gravity by this amount makes the parabolic motion 
        # usually intersect with the player's position
        self.velocity += self.acceleration / 160
        # Note we have to invert the y component of the velocity to reflect pygame's y coordinate convention
        self.rect.move_ip(self.velocity.x, -1 * self.velocity.y)
