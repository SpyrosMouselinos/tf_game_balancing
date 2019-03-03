import numpy as np


class BaseClass(object):
    def __init__(self, hp=None, attack=None, defense=None, speed=None):
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.current_hp = self.hp

    def set_atk(self, attack):
        self.attack = attack

    def set_def(self, defense):
        self.defense = defense

    def set_hp(self, hp):
        self.hp = hp

    def set_current_hp(self, hp):
        self.current_hp = hp

    def set_speed(self, speed):
        self.speed = speed

    def get_atk(self):
        return self.attack

    def get_def(self):
        return self.defense

    def get_hp(self):
        return self.hp

    def get_speed(self):
        return self.speed

    def get_current_hp(self):
        return self.current_hp

    def fight(self, opponent):
        if isinstance(opponent, BaseClass):
            damage = 0.5*(self.get_atk() * 0.8 + self.get_speed() * 0.2)
            if opponent.get_def() > self.get_atk():
                damage -= np.abs(opponent.get_def() - self.get_atk())*0.6
            else:
                damage -= np.abs(opponent.get_def() - self.get_atk())*0.3
            if damage >= 0:
                opponent.set_current_hp(opponent.get_current_hp() - damage)
        else:
            raise(AttributeError, 'Attack should get a second argument of GameClass!\n')

    def is_dead(self):
        return self.get_current_hp() <= 0


class GameClass(BaseClass):
    def __init__(self, hp=None, attack=None, defense=None, speed=None, name=None):
        self.name = name
        super().__init__(hp=hp, attack=attack, defense=defense, speed=speed)

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name


class Warrior(GameClass):
    def __init__(self, hp=None, attack=None, defense=None, speed=None):
        super().__init__(hp=hp, attack=attack, defense=defense, speed=speed, name='Warrior')

    def set_name(self, name):
        self.name = 'Warrior_'+str(name)

    @staticmethod
    def permute_and_give_points(total_points, stats):
        return np.random.multinomial(total_points, np.ones(stats) / stats, size=1)[0]

    @classmethod
    def make_random(cls, total_points):
        hp, attack, defense, speed = Warrior.permute_and_give_points(total_points, 4)
        return cls(hp=hp, attack=attack, defense=defense, speed=speed)


class Wizard(GameClass):
    def __init__(self, hp=None, attack=None, defense=None, speed=None):
        super().__init__(hp=hp, attack=attack, defense=defense, speed=speed, name='Wizard')

    def set_name(self, name):
        self.name = 'Wizard_' + str(name)

    @staticmethod
    def permute_and_give_points(total_points, stats):
        return np.random.multinomial(total_points, np.ones(stats) / stats, size=1)[0]

    @classmethod
    def make_random(cls, total_points):
        hp, attack, defense, speed = Wizard.permute_and_give_points(total_points, 4)
        return cls(hp=hp, attack=attack, defense=defense, speed=speed)


class Rogue(GameClass):
    def __init__(self, hp=None, attack=None, defense=None, speed=None):
        super().__init__(hp=hp, attack=attack, defense=defense, speed=speed, name='Rogue')

    def set_name(self, name):
        self.name = 'Rogue_' + str(name)

    @staticmethod
    def permute_and_give_points(total_points, stats):
        return np.random.multinomial(total_points, np.ones(stats) / stats, size=1)[0]

    @classmethod
    def make_random(cls, total_points):
        hp, attack, defense, speed = Rogue.permute_and_give_points(total_points, 4)
        return cls(hp=hp, attack=attack, defense=defense, speed=speed)
